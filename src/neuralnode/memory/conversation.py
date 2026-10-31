"""Memory module: ConversationMemory."""

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

class ConversationMemory(Memory):
    """Fast in-memory conversation storage.

    Keeps up to *max_messages* messages. When the limit is exceeded it
    preserves all ``SYSTEM`` messages and discards the oldest
    ``USER``/``ASSISTANT`` pairs.

    Example::

        mem = ConversationMemory(max_messages=100)
        mem.add(Message.user("Hello"))
        mem.add(Message.assistant("Hi there!"))
        print(mem.count())          # 2
        results = mem.search("hello")
    """

    def __init__(self, max_messages: int = 200):
        self.max_messages = max_messages
        self._messages: List[Message] = []

    # ── Core interface ─────────────────────────────────────────────────────────

    def add(self, message: Message) -> None:
        """Add a message to memory."""
        self._messages.append(message)
        self._trim()

    def get(self, limit: Optional[int] = None) -> List[Message]:
        """Return messages, most recent *limit* if specified."""
        if limit:
            return self._messages[-limit:]
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def to_list(self) -> List[Dict[str, Any]]:
        """Export as list of dicts (JSON-serializable)."""
        return [m.to_dict() for m in self._messages]

    # ── Extended helpers ───────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of stored messages."""
        return len(self._messages)

    def last(self, n: int = 1) -> List[Message]:
        """Return the last *n* messages."""
        return self._messages[-n:]

    def search(self, query: str, case_sensitive: bool = False) -> List[Message]:
        """Keyword search through memory content.

        Args:
            query: Search string.
            case_sensitive: Whether to match case.

        Returns:
            List of matching messages.
        """
        if not case_sensitive:
            query = query.lower()
        results = []
        for msg in self._messages:
            content = msg.content if case_sensitive else msg.content.lower()
            if query in content:
                results.append(msg)
        return results

    def get_by_role(self, role: MessageRole) -> List[Message]:
        """Return only messages with a specific role."""
        return [m for m in self._messages if m.role == role]

    def summary_text(self) -> str:
        """Return memory as a single human-readable string."""
        lines = []
        for m in self._messages:
            prefix = m.role.value.capitalize()
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_to_file(self, path: str) -> None:
        """Save memory to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, path: str) -> "ConversationMemory":
        """Load memory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mem = cls()
        for d in data:
            msg = Message(
                role=MessageRole(d["role"]),
                content=d["content"],
                metadata=d.get("metadata", {}),
                timestamp=datetime.fromisoformat(d["timestamp"]),
            )
            mem._messages.append(msg)
        return mem

    # ── Internal ───────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        if len(self._messages) <= self.max_messages:
            return
        system = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        other = [m for m in self._messages if m.role != MessageRole.SYSTEM]
        keep = max(self.max_messages - len(system), 0)
        self._messages = system + other[-keep:]
