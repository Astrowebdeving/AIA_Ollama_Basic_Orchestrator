import asyncio
import ollama as ollama_sdk
from config import LANCEDB_URI, EMBED_MODEL, OLLAMA_HOST
from context_manager import context_manager

# Lazy-init: avoid import-time side effects if Ollama is unreachable
_db = None
_TABLE_NAME = "knowledge_base"


def _get_db():
    global _db
    if _db is None:
        import lancedb
        _db = lancedb.connect(LANCEDB_URI)
        if _TABLE_NAME not in _db.table_names():
            # Bootstrap the table with one dummy row, then delete it
            _db.create_table(
                _TABLE_NAME,
                data=[{"vector": [0.0] * 768, "text": "", "metadata": "{}"}],
            )
            _db.open_table(_TABLE_NAME).delete("text = ''")
    return _db


class RagService:
    def __init__(self):
        self._client = ollama_sdk.Client(host=OLLAMA_HOST)

    # ----------------------------------------------------------------
    # Embedding helper
    # ----------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single string using the Ollama embedding model."""
        response = await asyncio.to_thread(
            self._client.embed, model=EMBED_MODEL, input=text
        )
        # embed() returns {"embeddings": [[...]]} — a list of vectors
        return response.embeddings[0]

    # ----------------------------------------------------------------
    # Document ingestion
    # ----------------------------------------------------------------

    async def add_document(self, text: str, metadata: str = "{}"):
        vector = await self.embed_text(text)
        table = _get_db().open_table(_TABLE_NAME)
        await asyncio.to_thread(
            table.add,
            [{"vector": vector, "text": text, "metadata": metadata}],
        )
        print(f"[RAG] Added document: {text[:60]}…")

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    async def retrieve_context(
        self, query: str, budget_limit: int
    ) -> tuple[str, int]:
        """
        Retrieve the top-k most relevant chunks that fit within
        *budget_limit* tokens.  Returns (context_string, remaining_budget).
        """
        try:
            query_vector = await self.embed_text(query)
            table = _get_db().open_table(_TABLE_NAME)
            results = await asyncio.to_thread(
                lambda: table.search(query_vector).limit(5).to_list()
            )

            context_chunks: list[str] = []
            remaining_budget = budget_limit

            for row in results:
                chunk_text = row.get("text") or ""
                if not chunk_text:
                    continue
                chunk_tokens = context_manager.count_tokens(chunk_text)
                if chunk_tokens <= remaining_budget:
                    context_chunks.append(chunk_text)
                    remaining_budget -= chunk_tokens
                else:
                    break

            if context_chunks:
                combined = "Relevant Context:\n" + "\n---\n".join(context_chunks)
                return combined, remaining_budget
            return "", remaining_budget

        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return "", budget_limit


rag_service = RagService()
