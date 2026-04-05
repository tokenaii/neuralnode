"""Core types and base classes for NeuralNode."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Callable
from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)
    
    @classmethod
    def tool(cls, content: str, tool_name: str) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            metadata={"tool_name": tool_name}
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like accessor for provider compatibility."""
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create Message from dictionary."""
        from datetime import datetime
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        )


@dataclass
class ToolCall:
    """A tool call from the LLM."""
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id
        }


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    provider: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def text(self) -> str:
        """Get the text content (alias for content)."""
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "metadata": self.metadata
        }


def ensure_llm_response(
    response: Any,
    default_model: str = "unknown",
    default_provider: str = "unknown",
) -> "LLMResponse":
    """Normalize provider outputs to ``LLMResponse``."""
    if isinstance(response, LLMResponse):
        return response

    if isinstance(response, str):
        return LLMResponse(
            content=response,
            model=default_model,
            provider=default_provider,
        )

    if isinstance(response, dict):
        raw_tool_calls = response.get("tool_calls", []) or []
        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            if isinstance(tc, ToolCall):
                tool_calls.append(tc)
            elif isinstance(tc, dict):
                tool_calls.append(
                    ToolCall(
                        name=tc.get("name", ""),
                        arguments=tc.get("arguments", {}) or {},
                        call_id=tc.get("call_id"),
                    )
                )

        return LLMResponse(
            content=response.get("content", ""),
            model=response.get("model", default_model),
            provider=response.get("provider", default_provider),
            tool_calls=tool_calls,
            usage=response.get("usage", {}) or {},
            finish_reason=response.get("finish_reason"),
            metadata=response.get("metadata", {}) or {},
        )

    raise TypeError(f"Unsupported response type: {type(response)!r}")


def ensure_streaming_chunks(stream: Iterator[Any]) -> Iterator["StreamingChunk"]:
    """Normalize streaming iterators to ``StreamingChunk`` objects."""
    for chunk in stream:
        if isinstance(chunk, StreamingChunk):
            yield chunk
        elif isinstance(chunk, dict):
            yield StreamingChunk(
                content=chunk.get("content", ""),
                is_finished=chunk.get("done", False) or chunk.get("is_finished", False),
            )
        elif isinstance(chunk, str):
            yield StreamingChunk(content=chunk)
        else:
            yield StreamingChunk(content=str(chunk))


@dataclass
class StreamingChunk:
    """A chunk of a streaming response."""
    content: str
    is_finished: bool = False
    
    @property
    def text(self) -> str:
        """Get the text content."""
        return self.content


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, content: str, **data) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, content=content, data=data)
    
    @classmethod
    def error(cls, message: str, **data) -> "ToolResult":
        """Create an error result."""
        return cls(success=False, content=message, data=data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "content": self.content,
            "data": self.data
        }
    
    @property
    def output(self) -> Dict[str, Any]:
        """Get output as dict for backward compatibility."""
        return self.data


@dataclass
class ParameterSpec:
    """Specification for a tool parameter."""
    type: str  # string, integer, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary format."""
        result = {
            "type": self.type,
            "description": self.description
        }
        if self.enum is not None:
            result["enum"] = self.enum
        return result


class BaseProvider(ABC):
    """Abstract base class for LLM providers.
    
    This is the minimal interface that all providers must implement.
    For enhanced functionality, use BaseLLMProvider from providers.base.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.config = kwargs
        self._client = None
        self._async_client = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, bool]:
        """Return provider capabilities."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List["BaseTool"]] = None,
        stream: bool = False,
        **options
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send a chat request to the LLM."""
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List["BaseTool"]] = None,
        stream: bool = False,
        **options
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async version of chat."""
        pass
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    @abstractmethod
    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed."""
        pass
    
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return self.capabilities.get("tools", False)
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self.capabilities.get("streaming", False)
    
    def supports_embeddings(self) -> bool:
        """Check if provider supports embeddings."""
        return self.capabilities.get("embeddings", False)
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        env_var = f"{self.provider_name.upper().replace('-', '_')}_API_KEY"
        return os.getenv(env_var)


class BaseTool(ABC):
    """Abstract base class for tools."""
    
    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    
    def __init__(self):
        """Initialize the tool."""
        pass
    
    @abstractmethod
    def execute(self, **params) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    async def aexecute(self, **params) -> ToolResult:
        """Async version of execute."""
        return self.execute(**params)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        required = [
            name for name, spec in self.parameters.items()
            if spec.required
        ]
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: spec.to_dict()
                        for name, spec in self.parameters.items()
                    },
                    "required": required
                }
            }
        }
    
    def validate_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Validate parameters. Returns error message or None if valid."""
        for name, spec in self.parameters.items():
            if spec.required and name not in params:
                return f"Missing required parameter: {name}"
            
            if name in params and spec.enum is not None:
                if params[name] not in spec.enum:
                    return f"Invalid value for {name}: {params[name]}. Must be one of: {spec.enum}"
        
        return None


class BaseAgent(ABC):
    """Abstract base class for agents."""
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.max_iterations = max_iterations
    
    @abstractmethod
    def run(self, task: str) -> str:
        """Run the agent on a task."""
        pass
    
    @abstractmethod
    async def arun(self, task: str) -> str:
        """Async version of run."""
        pass
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List available tool names."""
        return list(self.tools.keys())
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool dynamically."""
        self.tools[tool.name] = tool
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool by name. Returns True if removed."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False
    
    def reset(self) -> None:
        """Reset agent state. Override in subclasses."""
        pass


class Memory(ABC):
    """Abstract base class for conversation memory."""
    
    @abstractmethod
    def add(self, message: Message) -> None:
        """Add a message to memory."""
        pass
    
    @abstractmethod
    def get(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from memory."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""
        pass
    
    @abstractmethod
    def to_list(self) -> List[Dict[str, Any]]:
        """Export memory as list of dicts."""
        pass
