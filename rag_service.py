"""
RAG Service Module
==================
Handles document ingestion (with chunking), embedding via Ollama,
vector storage in LanceDB, and budget-aware retrieval.
"""

import asyncio
import json
from typing import Optional

import ollama as ollama_sdk

from config import LANCEDB_URI, EMBED_MODEL, EMBED_DIM, OLLAMA_HOST
from context_manager import context_manager

# ---------------------------------------------------------------
# LanceDB lazy initialisation
# ---------------------------------------------------------------

_db = None
_TABLE_NAME = "knowledge_base"

# Maximum cosine distance for a result to be considered relevant.
# Lower = stricter.  Cosine distance ranges from 0 (identical) to 2 (opposite).
_MAX_DISTANCE = 1.2


def _get_db():
    """Lazy-connect to LanceDB and bootstrap the table if needed."""
    global _db
    if _db is None:
        import lancedb

        _db = lancedb.connect(LANCEDB_URI)
        if _TABLE_NAME not in _db.table_names():
            # Bootstrap with one dummy row so the schema is defined, then delete
            _db.create_table(
                _TABLE_NAME,
                data=[{
                    "vector": [0.0] * EMBED_DIM,
                    "text": "",
                    "metadata": "{}",
                }],
            )
            _db.open_table(_TABLE_NAME).delete("text = ''")
    return _db


# ---------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------


class RagService:
    def __init__(self):
        self._client = ollama_sdk.Client(host=OLLAMA_HOST)

    # ----------------------------------------------------------------
    # Embedding helpers
    # ----------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string using the configured Ollama embedding model."""
        response = await asyncio.to_thread(
            self._client.embed, model=EMBED_MODEL, input=text
        )
        return response.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple strings in a single Ollama call.
        Falls back to one-by-one if the batch call fails.
        """
        if not texts:
            return []

        try:
            response = await asyncio.to_thread(
                self._client.embed, model=EMBED_MODEL, input=texts
            )
            return response.embeddings
        except Exception as exc:
            print(f"[RAG] Batch embed failed ({exc}), falling back to sequential")
            results = []
            for t in texts:
                vec = await self.embed_text(t)
                results.append(vec)
            return results

    # ----------------------------------------------------------------
    # Document chunking
    # ----------------------------------------------------------------

    def chunk_text(
        self, text: str, max_tokens: int = 512, overlap_tokens: int = 64
    ) -> list[str]:
        """
        Split *text* into chunks of at most *max_tokens* tokens,
        with a sliding overlap for continuity.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative.")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens.")

        context_manager.load_tokenizer()
        tokens = context_manager.tokenizer.encode(text)

        if len(tokens) <= max_tokens:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_str = context_manager.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_str)
            # Advance by (max_tokens - overlap) so consecutive chunks share context
            start += max_tokens - overlap_tokens

        return chunks

    # ----------------------------------------------------------------
    # Document ingestion
    # ----------------------------------------------------------------

    async def add_document(
        self, text: str, metadata: str = "{}", max_chunk_tokens: int = 512
    ):
        """
        Chunk the document, batch-embed all chunks, and write them
        to LanceDB in one go.
        """
        chunks = self.chunk_text(text, max_tokens=max_chunk_tokens)
        vectors = await self.embed_batch(chunks)

        rows = [
            {"vector": vec, "text": chunk, "metadata": metadata}
            for vec, chunk in zip(vectors, chunks)
        ]

        table = _get_db().open_table(_TABLE_NAME)
        await asyncio.to_thread(table.add, rows)
        print(
            f"[RAG] Added {len(chunks)} chunk(s) from document: "
            f"{text[:60]}…"
        )

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    async def retrieve_context(
        self,
        query: str,
        budget_limit: int,
        top_k: int = 10,
        max_distance: Optional[float] = None,
    ) -> tuple[str, int]:
        """
        Retrieve the most relevant chunks that fit within *budget_limit*
        tokens.  Results beyond *max_distance* (cosine) are discarded.

        Returns (context_string, remaining_budget).
        """
        if max_distance is None:
            max_distance = _MAX_DISTANCE

        try:
            query_vector = await self.embed_text(query)
            table = _get_db().open_table(_TABLE_NAME)
            results = await asyncio.to_thread(
                lambda: table.search(query_vector).limit(top_k).to_list()
            )

            context_chunks: list[str] = []
            remaining_budget = budget_limit

            for row in results:
                # ---- Relevance filtering ----
                distance = row.get("_distance")
                if distance is not None and distance > max_distance:
                    continue

                chunk_text = row.get("text") or ""
                if not chunk_text:
                    continue

                chunk_tokens = context_manager.count_tokens(chunk_text)
                if chunk_tokens <= remaining_budget:
                    context_chunks.append(chunk_text)
                    remaining_budget -= chunk_tokens
                else:
                    break  # Budget exhausted

            if context_chunks:
                combined = (
                    "Relevant Context:\n" + "\n---\n".join(context_chunks)
                )
                return combined, remaining_budget
            return "", remaining_budget

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return "", budget_limit


# Global instance
rag_service = RagService()
