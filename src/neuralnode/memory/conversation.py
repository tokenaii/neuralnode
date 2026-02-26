"""Conversation memory management."""

from typing import Any, Dict, List, Optional

from ..core.types import Message, MessageRole


class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self, max_messages: int = 100):
        """
        Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self._messages: List[Message] = []
        self._metadata: Dict[str, Any] = {}

    def add(self, message: Message) -> None:
        """Add a message to memory.
        
        Args:
            message: Message object to add
        """
        self._messages.append(message)
        
        # Trim if exceeds max
        if len(self._messages) > self.max_messages:
            # Keep system messages, remove oldest non-system
            system_messages = [m for m in self._messages if m.role == MessageRole.SYSTEM]
            other_messages = [m for m in self._messages if m.role != MessageRole.SYSTEM]
            
            # Keep most recent non-system messages
            keep_count = self.max_messages - len(system_messages)
            other_messages = other_messages[-keep_count:]
            
            self._messages = system_messages + other_messages

    def add_message(self, role: MessageRole, content: str, **kwargs) -> None:
        """Add a message using role and content (convenience method).
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            **kwargs: Additional message parameters
        """
        message = Message(role=role, content=content, **kwargs)
        self.add(message)

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages from memory.
        
        Args:
            limit: Maximum number of messages to return (most recent)
            
        Returns:
            List of messages
        """
        if limit:
            return self._messages[-limit:]
        return self._messages.copy()

    def get_last_n(self, n: int) -> List[Message]:
        """Get last n messages."""
        return self._messages[-n:]

    def clear(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()
        self._metadata.clear()

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata in memory."""
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from memory."""
        return self._metadata.get(key, default)

    @property
    def message_count(self) -> int:
        """Get number of messages in memory."""
        return len(self._messages)

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "messages": [m.to_dict() for m in self._messages],
            "metadata": self._metadata,
            "max_messages": self.max_messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationMemory":
        """Restore memory from dictionary."""
        memory = cls(max_messages=data.get("max_messages", 100))
        
        for msg_data in data.get("messages", []):
            msg = Message(
                role=MessageRole(msg_data["role"]),
                content=msg_data["content"],
                name=msg_data.get("name"),
                tool_calls=msg_data.get("tool_calls"),
                tool_call_id=msg_data.get("tool_call_id"),
            )
            memory.add(msg)
        
        memory._metadata = data.get("metadata", {})
        return memory


class SlidingWindowMemory(ConversationMemory):
    """Memory that keeps only the last N message pairs."""

    def __init__(self, max_pairs: int = 5):
        """
        Initialize sliding window memory.
        
        Args:
            max_pairs: Maximum number of user-assistant pairs to keep
        """
        super().__init__(max_messages=max_pairs * 2 + 1)  # +1 for system message
        self.max_pairs = max_pairs

    def add(self, message: Message) -> None:
        """Add message with sliding window logic."""
        super().add(message)
        
        # Ensure we keep message pairs together
        # This is handled by the parent's trimming logic
