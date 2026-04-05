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


class SlidingWindowMemory(Memory):
    """Memory that keeps only the most recent N user-assistant exchanges.

    Useful for keeping context short with local models that have small
    context windows.

    Example::

        mem = SlidingWindowMemory(max_pairs=5)
    """

    def __init__(self, max_pairs: int = 5):
        self.max_pairs = max_pairs
        self._messages: List[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._trim()

    def get(self, limit: Optional[int] = None) -> List[Message]:
        if limit:
            return self._messages[-limit:]
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def to_list(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def count(self) -> int:
        return len(self._messages)

    def _trim(self) -> None:
        """Keep only the most recent max_pairs exchanges."""
        pair_count = sum(1 for m in self._messages if m.role == MessageRole.ASSISTANT)
        if pair_count <= self.max_pairs:
            return

        keep_from = 0
        found = 0
        for i, m in enumerate(reversed(self._messages)):
            if m.role == MessageRole.ASSISTANT:
                found += 1
                if found == self.max_pairs:
                    keep_from = len(self._messages) - i - 1
                    break

        system = [m for m in self._messages[:keep_from] if m.role == MessageRole.SYSTEM]
        self._messages = system + self._messages[keep_from:]


class SummaryMemory(Memory):
    """Memory that summarises old messages with the LLM when the context fills up.

    When the number of messages exceeds *max_messages*, all messages older
    than the *keep_recent* most recent ones are sent to the LLM for
    summarisation, and the result replaces them as a single ``SYSTEM``
    message. This keeps the context small while preserving important context.

    Example::

        from neuralnode.memory import SummaryMemory

        mem = SummaryMemory(llm=my_llm, max_messages=30, keep_recent=10)
        mem.add(Message.user("Who are you?"))
        # ... many more messages ...
        # When > 30 messages, older ones get auto-summarised.
    """

    def __init__(
        self,
        llm,  # BaseProvider — avoid circular import
        max_messages: int = 30,
        keep_recent: int = 10,
        summary_prompt: Optional[str] = None,
    ):
        """
        Args:
            llm: Any NeuralNode LLM provider (needs ``.chat()``).
            max_messages: Trigger summarisation when this is exceeded.
            keep_recent: How many recent messages to keep verbatim.
            summary_prompt: Custom summarisation instruction.
        """
        self.llm = llm
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt or (
            "Summarise the following conversation concisely, "
            "preserving all important facts, decisions, and context. "
            "Write in third person (e.g. 'The user asked...', 'The assistant explained...')."
        )
        self._messages: List[Message] = []
        self._summaries: List[str] = []  # History of past summaries

    # ── Core interface ─────────────────────────────────────────────────────────

    def add(self, message: Message) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._summarise()

    def get(self, limit: Optional[int] = None) -> List[Message]:
        """Return messages. Summaries are prepended as a SYSTEM message."""
        msgs = list(self._messages)
        if self._summaries:
            summary_block = "## Conversation Summary\n" + "\n---\n".join(self._summaries)
            msgs = [Message.system(summary_block)] + msgs
        if limit:
            return msgs[-limit:]
        return msgs

    def clear(self) -> None:
        self._messages.clear()
        self._summaries.clear()

    def to_list(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def count(self) -> int:
        return len(self._messages)

    def get_summaries(self) -> List[str]:
        """Return all past summaries."""
        return list(self._summaries)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _summarise(self) -> None:
        """Summarise old messages and replace them with a compact entry."""
        to_summarise = self._messages[: -self.keep_recent]
        keep = self._messages[-self.keep_recent :]

        if not to_summarise:
            return

        conv_text = "\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in to_summarise
        )
        prompt = f"{self.summary_prompt}\n\nConversation:\n{conv_text}"

        try:
            response = ensure_llm_response(self.llm.chat([Message.user(prompt)]))
            summary = response.content.strip()
            self._summaries.append(summary)
        except Exception as e:
            # Fallback: just truncate without LLM
            summary = f"[Earlier conversation — {len(to_summarise)} messages]"
            self._summaries.append(summary)

        self._messages = keep


# ── Exports ────────────────────────────────────────────────────────────────────

__all__ = [
    "ConversationMemory",
    "FileMemory",
    "SlidingWindowMemory",
    "SummaryMemory",
]
