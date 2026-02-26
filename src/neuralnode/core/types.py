"""Core types and interfaces for NeuralNode."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    LLAMACPP = "llamacpp"
    AUTO = "auto"


@dataclass
class Message:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost(self, model: str = "gpt-4") -> float:
        """Calculate approximate cost in USD."""
        rates = {
            "gpt-4": (0.03, 0.06),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0015, 0.002),
        }
        input_rate, output_rate = rates.get(model, (0.01, 0.03))
        return (
            self.prompt_tokens * input_rate / 1000
            + self.completion_tokens * output_rate / 1000
        )


@dataclass
class LLMResponse:
    """Response from an LLM."""

    text: str
    model: str
    provider: ProviderType
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingChunk:
    """A chunk of a streaming response."""

    text: str
    is_finished: bool = False
    usage: Optional[TokenUsage] = None


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    type: str
    description: str
    enum: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by an LLM."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    handler: Optional[Callable[..., Any]] = Field(default=None, exclude=True)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.config = kwargs
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send a chat completion request."""
        pass

    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async version of chat."""
        pass

    @abstractmethod
    def complete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send a text completion request."""
        pass

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async version of complete."""
        pass

    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings."""
        pass

    @abstractmethod
    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async version of embed."""
        pass

    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a callback for events."""
        self._callbacks.append(callback)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event to all callbacks."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception:
                pass

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports tool calling."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming."""
        pass

    @property
    @abstractmethod
    def supports_structured_output(self) -> bool:
        """Whether this provider supports structured output."""
        pass


@dataclass
class ChatOptions:
    """Options for chat requests."""

    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    stream: bool = False


class StructuredOutput(BaseModel):
    """Base class for structured output schemas."""

    class Config:
        extra = "allow"
