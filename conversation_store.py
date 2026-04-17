"""
Single-conversation history store.

The orchestrator is used as one local EVA assistant, so conversation state is a
single JSONL file rather than a database or session table. The application
system prompt is never persisted; it is generated fresh on every request.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any


_ROOT = Path(__file__).resolve().parent
_HISTORY_PATH = _ROOT / "runtime" / "conversation_history.jsonl"
_VALID_ROLES = {"user", "assistant"}


class ConversationStore:
    """File-backed store for the single global chat history."""

    def __init__(self, path: Path = _HISTORY_PATH):
        self.path = path
        self.lock = asyncio.Lock()

    def load_history(self) -> list[dict]:
        """Load persisted user/assistant messages from JSONL."""
        if not self.path.is_file():
            return []

        messages: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    item: Any = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue

                role = item.get("role")
                content = item.get("content")
                if role not in _VALID_ROLES or not isinstance(content, str):
                    continue
                messages.append({"role": role, "content": content})
        return messages

    def replace_history(self, messages: list[dict]) -> None:
        """Atomically replace the JSONL history with sanitized messages."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".jsonl.tmp")

        with temp_path.open("w", encoding="utf-8") as handle:
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if role not in _VALID_ROLES or not isinstance(content, str):
                    continue
                record = {"role": role, "content": content}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        os.replace(temp_path, self.path)

    def reset_history(self) -> None:
        """Clear the persisted conversation history."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")


conversation_store = ConversationStore()
