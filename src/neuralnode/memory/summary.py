"""Memory module: SummaryMemory."""

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
