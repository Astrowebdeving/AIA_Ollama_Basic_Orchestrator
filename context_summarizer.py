"""
Context Summarizer Module
=========================
Automatically summarizes older conversation turns when the
running token count crosses the configured threshold (default 64k).

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


class ContextSummarizer:
    """Summarizes older messages when context usage exceeds the threshold."""

    def should_summarize(self, total_tokens: int) -> bool:
        """Return True when the token count warrants summarization."""
        return total_tokens >= SUMMARIZE_THRESHOLD

    async def summarize_history(
        self,
        llm_provider,
        messages: list[dict],
        keep_recent: int = _KEEP_RECENT,
    ) -> list[dict]:
        """
        Compress *messages* by summarizing all but the most recent
        *keep_recent* messages into a single system message.

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
            by a "[CONTEXT SUMMARY]" system message.
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

        # Build a text block of the old conversation for the summarizer
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in old_turns
        )

        # Count tokens to see if summarization is worthwhile
        old_tokens = context_manager.count_tokens(history_text)
        if old_tokens < 5000:
            # Not worth summarizing tiny histories
            return messages

        print(
            f"[SUMMARIZER] Compressing {len(old_turns)} messages "
            f"({old_tokens} tokens) ..."
        )

        try:
            response = await llm_provider.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM},
                    {"role": "user", "content": history_text},
                ],
                max_context=MAX_CONTEXT_TOKENS,
            )

            summary_text = response.content or ""
            summary_tokens = context_manager.count_tokens(summary_text)

            print(
                f"[SUMMARIZER] Compressed {old_tokens} -> {summary_tokens} tokens "
                f"({round((1 - summary_tokens / old_tokens) * 100)}% reduction)"
            )

            summary_msg = {
                "role": "system",
                "content": (
                    "[CONTEXT SUMMARY -- earlier conversation condensed]\n"
                    + summary_text
                ),
            }

            return [system_msg, summary_msg] + recent_turns

        except Exception as exc:
            print(f"[SUMMARIZER] Failed: {exc} -- keeping original messages")
            return messages


# Global instance
context_summarizer = ContextSummarizer()
