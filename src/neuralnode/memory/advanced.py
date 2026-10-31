"""
Comprehensive Memory System for NeuralNode
Supports ConversationBufferMemory, ConversationSummaryMemory, VectorStoreRetrieverMemory
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in conversation memory."""

    role: str  # system, user, assistant, tool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=(
                datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
            ),
        )


class BaseMemory(ABC):
    """Abstract base class for memory systems."""

    @abstractmethod
    def add(self, message: Message) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get(self, **kwargs) -> List[Message]:
        """Get messages from memory."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory to dict."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseMemory":
        """Deserialize memory from dict."""
        pass


class ConversationBufferMemory(BaseMemory):
    """Simple buffer memory that stores recent messages."""

    def __init__(
        self,
        max_messages: int = 10,
        return_messages: bool = True,
        memory_key: str = "history",
    ):
        self.max_messages = max_messages
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.messages: List[Message] = []

    def add(self, message: Message) -> None:
        """Add a message to the buffer."""
        self.messages.append(message)

        # Trim if exceeds max
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.add(Message(role="user", content=content))

    def add_ai_message(self, content: str) -> None:
        """Add AI message."""
        self.add(Message(role="assistant", content=content))

    def get(self, k: Optional[int] = None, **kwargs) -> List[Message]:
        """Get recent messages."""
        k = k or self.max_messages
        return self.messages[-k:]

    def get_as_string(self, k: Optional[int] = None) -> str:
        """Get messages as formatted string."""
        messages = self.get(k=k)
        parts = []
        for msg in messages:
            role_label = {"user": "Human", "assistant": "AI", "system": "System"}.get(
                msg.role, msg.role
            )
            parts.append(f"{role_label}: {msg.content}")
        return "\n\n".join(parts)

    def get_as_messages(self, k: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages as list of dicts."""
        messages = self.get(k=k)
        return [{"role": m.role, "content": m.content} for m in messages]

    def load_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for injection into prompts."""
        if self.return_messages:
            return {self.memory_key: self.get_as_messages()}
        else:
            return {self.memory_key: self.get_as_string()}

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "ConversationBufferMemory",
            "max_messages": self.max_messages,
            "memory_key": self.memory_key,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationBufferMemory":
        """Deserialize from dict."""
        memory = cls(
            max_messages=data.get("max_messages", 10), memory_key=data.get("memory_key", "history")
        )
        memory.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return memory


class ConversationBufferWindowMemory(ConversationBufferMemory):
    """Buffer memory with sliding window - only keeps last k exchanges."""

    def __init__(self, k: int = 5, max_messages: Optional[int] = None, **kwargs):
        # Each exchange is 2 messages (user + assistant)
        super().__init__(max_messages=max_messages or (k * 2), **kwargs)
        self.k = k

    def get(self, **kwargs) -> List[Message]:
        """Get last k exchanges."""
        return self.messages[-(self.k * 2) :]


class ConversationSummaryMemory(BaseMemory):
    """Memory that summarizes conversation to reduce token usage."""

    def __init__(
        self,
        llm: Any,
        max_token_limit: int = 2000,
        memory_key: str = "history",
        summarizer_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.memory_key = memory_key
        self.summarizer_prompt = (
            summarizer_prompt
            or """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
        )

        self.buffer: List[Message] = []
        self.summary: str = ""

    def add(self, message: Message) -> None:
        """Add message to buffer."""
        self.buffer.append(message)

        # Check if we need to summarize
        if self._get_buffer_length() > self.max_token_limit:
            self._summarize()

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.add(Message(role="user", content=content))

    def add_ai_message(self, content: str) -> None:
        """Add AI message."""
        self.add(Message(role="assistant", content=content))

    def _get_buffer_length(self) -> int:
        """Estimate buffer length in tokens (rough estimate)."""
        return sum(len(m.content.split()) for m in self.buffer)

    def _summarize(self) -> None:
        """Summarize the buffer."""
        if not self.buffer:
            return

        # Format new lines
        new_lines = "\n".join([f"{m.role}: {m.content}" for m in self.buffer])

        # Create prompt
        prompt = self.summarizer_prompt.format(summary=self.summary, new_lines=new_lines)

        # Get summary from LLM
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            if isinstance(response, dict):
                self.summary = response.get("content", self.summary)
            else:
                self.summary = str(response)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")

        # Clear buffer
        self.buffer = []

    def get(self, **kwargs) -> List[Message]:
        """Get memory as messages."""
        messages = []

        if self.summary:
            messages.append(
                Message(role="system", content=f"Previous conversation summary: {self.summary}")
            )

        messages.extend(self.buffer)

        return messages

    def get_as_string(self) -> str:
        """Get memory as string."""
        parts = []
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        parts.extend([f"{m.role}: {m.content}" for m in self.buffer])
        return "\n\n".join(parts)

    def load_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables."""
        return {self.memory_key: self.get_as_string()}

    def clear(self) -> None:
        """Clear memory."""
        self.buffer = []
        self.summary = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "ConversationSummaryMemory",
            "max_token_limit": self.max_token_limit,
            "memory_key": self.memory_key,
            "summary": self.summary,
            "buffer": [m.to_dict() for m in self.buffer],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm: Any = None) -> "ConversationSummaryMemory":
        """Deserialize from dict."""
        memory = cls(
            llm=llm,
            max_token_limit=data.get("max_token_limit", 2000),
            memory_key=data.get("memory_key", "history"),
        )
        memory.summary = data.get("summary", "")
        memory.buffer = [Message.from_dict(m) for m in data.get("buffer", [])]
        return memory


class VectorStoreRetrieverMemory(BaseMemory):
    """Memory that uses vector store for semantic retrieval."""

    def __init__(
        self,
        vector_store: Any,
        embeddings: Any,
        memory_key: str = "history",
        top_k: int = 4,
        input_key: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.memory_key = memory_key
        self.top_k = top_k
        self.input_key = input_key
        self.messages: List[Message] = []

    def add(self, message: Message) -> None:
        """Add message to memory and vector store."""
        self.messages.append(message)

        # Add to vector store
        try:
            if hasattr(self.vector_store, "add_texts"):
                self.vector_store.add_texts(
                    texts=[message.content],
                    metadatas=[{"role": message.role, "timestamp": message.timestamp.isoformat()}],
                )
            elif hasattr(self.vector_store, "add"):
                embedding = self.embeddings.embed([message.content])[0]
                self.vector_store.add(
                    ids=[str(len(self.messages))],
                    embeddings=[embedding],
                    documents=[message.content],
                    metadatas=[{"role": message.role}],
                )
        except Exception as e:
            logger.error(f"Failed to add to vector store: {e}")

    def add_user_message(self, content: str) -> None:
        """Add user message."""
        self.add(Message(role="user", content=content))

    def add_ai_message(self, content: str) -> None:
        """Add AI message."""
        self.add(Message(role="assistant", content=content))

    def get(self, query: Optional[str] = None, **kwargs) -> List[Message]:
        """Get relevant messages based on query."""
        if not query:
            # Return recent messages
            return self.messages[-self.top_k :]

        # Search vector store
        try:
            if hasattr(self.vector_store, "similarity_search"):
                results = self.vector_store.similarity_search(query, k=self.top_k)
            elif hasattr(self.vector_store, "search"):
                results = self.vector_store.search(query, k=self.top_k)
            else:
                results = []

            # Convert to messages
            messages = []
            for doc in results:
                if isinstance(doc, dict):
                    content = doc.get("content", doc.get("text", ""))
                    role = doc.get("metadata", {}).get("role", "assistant")
                else:
                    content = getattr(doc, "content", getattr(doc, "page_content", str(doc)))
                    role = getattr(doc, "metadata", {}).get("role", "assistant")
                messages.append(Message(role=role, content=content))

            return messages

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Get memory variables based on input."""
        query = inputs.get(self.input_key, "") if self.input_key else ""
        messages = self.get(query=query)

        return {self.memory_key: [{"role": m.role, "content": m.content} for m in messages]}

    def clear(self) -> None:
        """Clear memory."""
        self.messages = []
        # Note: vector store may need separate clearing

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "type": "VectorStoreRetrieverMemory",
            "memory_key": self.memory_key,
            "top_k": self.top_k,
            "input_key": self.input_key,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], vector_store: Any = None, embeddings: Any = None
    ) -> "VectorStoreRetrieverMemory":
        """Deserialize from dict."""
        memory = cls(
            vector_store=vector_store,
            embeddings=embeddings,
            memory_key=data.get("memory_key", "history"),
            top_k=data.get("top_k", 4),
            input_key=data.get("input_key"),
        )
        memory.messages = [Message.from_dict(m) for m in data.get("messages", [])]
        return memory


class CombinedMemory(BaseMemory):
    """Combines multiple memory types."""

    def __init__(self, memories: List[BaseMemory]):
        self.memories = memories

    def add(self, message: Message) -> None:
        """Add to all memories."""
        for memory in self.memories:
            memory.add(message)

    def add_user_message(self, content: str) -> None:
        """Add user message to all."""
        msg = Message(role="user", content=content)
        for memory in self.memories:
            memory.add(msg)

    def add_ai_message(self, content: str) -> None:
        """Add AI message to all."""
        msg = Message(role="assistant", content=content)
        for memory in self.memories:
            memory.add(msg)

    def get(self, **kwargs) -> List[Message]:
        """Get from first memory (or combine)."""
        # Combine all messages
        all_messages = []
        for memory in self.memories:
            all_messages.extend(memory.get(**kwargs))
        return all_messages

    def load_memory_variables(self) -> Dict[str, Any]:
        """Combine all memory variables."""
        result = {}
        for memory in self.memories:
            if hasattr(memory, "load_memory_variables"):
                result.update(memory.load_memory_variables())
        return result

    def clear(self) -> None:
        """Clear all memories."""
        for memory in self.memories:
            memory.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {"type": "CombinedMemory", "memories": [m.to_dict() for m in self.memories]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CombinedMemory":
        """Deserialize from dict."""
        memories = []
        for m_data in data.get("memories", []):
            m_type = m_data.get("type")
            if m_type == "ConversationBufferMemory":
                memories.append(ConversationBufferMemory.from_dict(m_data))
            elif m_type == "ConversationSummaryMemory":
                memories.append(ConversationSummaryMemory.from_dict(m_data))
            elif m_type == "VectorStoreRetrieverMemory":
                memories.append(VectorStoreRetrieverMemory.from_dict(m_data))
        return cls(memories=memories)


# Convenience functions
def create_conversation_buffer_memory(max_messages: int = 10, **kwargs) -> ConversationBufferMemory:
    """Create conversation buffer memory."""
    return ConversationBufferMemory(max_messages=max_messages, **kwargs)


def create_conversation_summary_memory(llm: Any, **kwargs) -> ConversationSummaryMemory:
    """Create conversation summary memory."""
    return ConversationSummaryMemory(llm=llm, **kwargs)


def create_vector_store_memory(
    vector_store: Any, embeddings: Any, **kwargs
) -> VectorStoreRetrieverMemory:
    """Create vector store memory."""
    return VectorStoreRetrieverMemory(vector_store=vector_store, embeddings=embeddings, **kwargs)


__all__ = [
    # Base
    "Message",
    "BaseMemory",
    # Memory Types
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationSummaryMemory",
    "VectorStoreRetrieverMemory",
    "CombinedMemory",
    # Convenience functions
    "create_conversation_buffer_memory",
    "create_conversation_summary_memory",
    "create_vector_store_memory",
]
