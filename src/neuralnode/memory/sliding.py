"""Memory module: SlidingWindowMemory."""

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
                    # Include the user message preceding this assistant reply.
                    keep_from = max(len(self._messages) - i - 2, 0)
                    break

        system = [m for m in self._messages[:keep_from] if m.role == MessageRole.SYSTEM]
        self._messages = system + self._messages[keep_from:]
