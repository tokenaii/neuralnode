"""Memory module: FileMemory."""

"""NeuralNode Memory System.

Three memory backends for agent conversation history:

    ConversationMemory  – fast in-memory (lost on restart)
    FileMemory          – persists to disk as JSON
    SlidingWindowMemory – keeps only the N most recent exchanges
    SummaryMemory       – compresses old history with the LLM

Example::

    from neuralnode.memory import ConversationMemory, FileMemory, SummaryMemory

    # In-memory
    mem = ConversationMemory(max_messages=200)

    # Persistent to disk
    mem = FileMemory("chat_history.json")

    # Auto-summarise old messages with the LLM
    mem = SummaryMemory(llm=my_llm, max_messages=20)
"""

import json
import re
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

from ..core import Memory, Message, MessageRole, ensure_llm_response


class FileMemory(Memory):
    """Persistent JSON-file-backed conversation memory.

    All changes are flushed to disk immediately after every ``add()``
    and ``clear()``.

    Example::

        mem = FileMemory("agent_history.json")
        mem.add(Message.user("Hello"))
        # Reload later
        mem2 = FileMemory("agent_history.json")
        print(mem2.count())  # 1
    """

    def __init__(self, file_path: str, max_messages: int = 2000):
        self.file_path = Path(file_path)
        self.max_messages = max_messages
        self._messages: List[Message] = []
        self._load()

    # ── Core interface ─────────────────────────────────────────────────────────

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()
        self._save()

    def get(self, limit: Optional[int] = None) -> List[Message]:
        if limit:
            return self._messages[-limit:]
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._save()

    def to_list(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    # ── Extended helpers ───────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self._messages)

    def search(self, query: str, case_sensitive: bool = False) -> List[Message]:
        """Keyword search through stored messages."""
        if not case_sensitive:
            query = query.lower()
        results = []
        for msg in self._messages:
            content = msg.content if case_sensitive else msg.content.lower()
            if query in content:
                results.append(msg)
        return results

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                msg = Message(
                    role=MessageRole(d["role"]),
                    content=d["content"],
                    metadata=d.get("metadata", {}),
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                )
                self._messages.append(msg)
        except Exception:
            pass  # Start fresh on corrupt file

    def _save(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2, ensure_ascii=False)

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return
        system = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        keep = max(self.max_messages - len(system), 0)
        self._messages = system + other[-keep:]
