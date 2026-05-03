"""
Context Summarizer Module
=========================
Automatically summarizes older conversation turns when the
running token count crosses the configured threshold (default 80k).

This keeps long multi-turn conversations within the context
window without simply truncating valuable earlier messages.
"""

from config import SUMMARIZE_THRESHOLD, LLM_MODEL, MAX_CONTEXT_TOKENS
from context_manager import context_manager

_SUMMARIZE_SYSTEM = (
    "You are a precise conversation summarizer. "
    "Condense the following conversation history into a concise summary "
    "that preserves all key facts, decisions, code snippets, tool results, "
    "and user preferences. Omit pleasantries and filler. "
    "Output ONLY the summary, no preamble."
)

# Keep the N most recent messages unsummarized so the model has
# immediate conversational context.
_KEEP_RECENT = 4
_MIN_SUMMARY_TOKENS = 5000
_SUMMARY_INPUT_TOKEN_BUDGET = max(4000, min(60000, MAX_CONTEXT_TOKENS // 2))
_SUMMARY_PREFIX = (
    "[CONTEXT SUMMARY -- earlier conversation condensed; "
    "use as background context only]\n"
)


class ContextSummarizer:
    """Summarizes older messages when context usage exceeds the threshold."""

    def should_summarize(self, total_tokens: int) -> bool:
        """Return True when the token count warrants summarization."""
        return total_tokens >= SUMMARIZE_THRESHOLD

    @staticmethod
    def _format_message(message: dict) -> str:
        """Convert a message dict to a text line for summarization.

        If the message carried base64 images (from ``inspect_image``),
        note the presence so the summary records that an image was
        consulted — but never include the raw base64 data.
        """
        role = message.get("role", "unknown")
        content = message.get("content", "")
        images = message.get("images")
        if images:
            content += f" [attached {len(images)} image(s) for vision analysis]"
        return f"{role}: {content}"

    @staticmethod
    def _strip_images(messages: list[dict]) -> list[dict]:
        """Remove base64 ``images`` from messages, leaving a text note.

        This is used after the LLM has already seen and responded to the
        images — the visual tokens are dead weight from that point on
        (~3-5 K tokens per high-res image).
        """
        cleaned: list[dict] = []
        for msg in messages:
            if "images" in msg:
                msg = dict(msg)  # shallow copy to avoid mutating originals
                n = len(msg.pop("images"))
                # Leave a breadcrumb so future turns know an image was here
                existing = msg.get("content", "")
                if "[image data stripped" not in existing.lower():
                    msg["content"] = existing + f" [image data stripped; {n} image(s) were shown to assistant]"
            cleaned.append(msg)
        return cleaned

    @staticmethod
    def _split_text_to_token_budget(text: str, token_budget: int) -> list[str]:
        context_manager.load_tokenizer()
        tokens = context_manager.tokenizer.encode(text)
        if len(tokens) <= token_budget:
            return [text]

        chunks: list[str] = []
        for start in range(0, len(tokens), token_budget):
            chunk_tokens = tokens[start:start + token_budget]
            chunks.append(context_manager.tokenizer.decode(chunk_tokens))
        return chunks

    def _chunk_messages(self, messages: list[dict]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for message in messages:
            text = self._format_message(message)
            text_tokens = context_manager.count_tokens(text)

            if text_tokens > _SUMMARY_INPUT_TOKEN_BUDGET:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.extend(
                    self._split_text_to_token_budget(
                        text,
                        _SUMMARY_INPUT_TOKEN_BUDGET,
                    )
                )
                continue

            if current and current_tokens + text_tokens > _SUMMARY_INPUT_TOKEN_BUDGET:
                chunks.append("\n".join(current))
                current = []
                current_tokens = 0

            current.append(text)
            current_tokens += text_tokens

        if current:
            chunks.append("\n".join(current))

        return chunks

    async def _summarize_text(self, llm_provider, text: str) -> str:
        response = await llm_provider.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": text},
            ],
            max_context=MAX_CONTEXT_TOKENS,
        )
        return (response.content or "").strip()

    async def _summarize_chunks(self, llm_provider, chunks: list[str]) -> str:
        if len(chunks) == 1:
            return await self._summarize_text(llm_provider, chunks[0])

        partial_summaries: list[str] = []
        for index, chunk in enumerate(chunks, 1):
            print(
                f"[SUMMARIZER] Summarizing chunk {index}/{len(chunks)} "
                f"({context_manager.count_tokens(chunk)} tokens) ..."
            )
            partial = await self._summarize_text(
                llm_provider,
                f"Conversation chunk {index} of {len(chunks)}:\n{chunk}",
            )
            if partial:
                partial_summaries.append(f"Chunk {index} summary:\n{partial}")

        combined = "\n\n".join(partial_summaries)
        if context_manager.count_tokens(combined) <= _SUMMARY_INPUT_TOKEN_BUDGET:
            return await self._summarize_text(
                llm_provider,
                "Combine these partial summaries into one coherent conversation "
                "summary. Preserve mission-critical facts, decisions, current "
                "task state, unresolved questions, and user preferences.\n\n"
                + combined,
            )

        return await self._summarize_chunks(
            llm_provider,
            self._split_text_to_token_budget(
                combined,
                _SUMMARY_INPUT_TOKEN_BUDGET,
            ),
        )

    async def summarize_history(
        self,
        llm_provider,
        messages: list[dict],
        keep_recent: int = _KEEP_RECENT,
    ) -> list[dict]:
        """
        Compress *messages* by summarizing all but the most recent
        *keep_recent* messages into a single context message.

        Parameters
        ----------
        llm_provider : LLMProvider
            The LLM provider instance (Ollama, AFM, or llama.cpp).
        messages : list[dict]
            The full message list (system + user/assistant/tool turns).
        keep_recent : int
            Number of tail messages to leave unsummarized.

        Returns
        -------
        list[dict]
            A shorter message list where older turns have been replaced
            by a "[CONTEXT SUMMARY]" context message.
        """
        if len(messages) <= keep_recent + 1:
            # Not enough messages to summarize (system + few turns)
            return messages

        # Split: system prompt (index 0) | old turns | recent turns
        system_msg = messages[0]
        old_turns = messages[1:-keep_recent]
        recent_turns = messages[-keep_recent:]

        if not old_turns:
            return messages

        chunks = self._chunk_messages(old_turns)

        # Count tokens to see if summarization is worthwhile
        old_tokens = sum(context_manager.count_tokens(chunk) for chunk in chunks)
        if old_tokens < _MIN_SUMMARY_TOKENS:
            # Not worth summarizing tiny histories
            return messages

        print(
            f"[SUMMARIZER] Compressing {len(old_turns)} messages "
            f"({old_tokens} tokens) ..."
        )

        try:
            summary_text = await self._summarize_chunks(llm_provider, chunks)
            if not summary_text:
                raise ValueError("summary was empty")
            summary_tokens = context_manager.count_tokens(summary_text)

            print(
                f"[SUMMARIZER] Compressed {old_tokens} -> {summary_tokens} tokens "
                f"({round((1 - summary_tokens / old_tokens) * 100)}% reduction)"
            )

            # Keep the application system prompt as the only system message.
            # Some chat backends/template adapters treat later system messages
            # as replacements or higher-priority instructions, so summaries are
            # carried as ordinary context instead of competing with the prompt.
            summary_msg = {
                "role": "user",
                "content": _SUMMARY_PREFIX + summary_text,
            }

            # Strip base64 images from surviving recent messages.
            # The LLM has already seen and responded to these images;
            # keeping the raw data wastes 3-5 K tokens per image.
            recent_turns = self._strip_images(recent_turns)

            return [system_msg, summary_msg] + recent_turns

        except Exception as exc:
            print(f"[SUMMARIZER] Failed: {exc} -- keeping original messages")
            return messages


# Global instance
context_summarizer = ContextSummarizer()
