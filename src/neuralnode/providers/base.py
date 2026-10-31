"""Base classes and utilities for comprehensive LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Callable, Type
import os
import json
import logging

logger = logging.getLogger(__name__)

# Import base types from core - do this first to avoid circular imports
from ..core import (
    BaseProvider,
    Message,
    MessageRole,
    LLMResponse,
    StreamingChunk,
    ToolCall,
    BaseTool,
)


class ModelType(Enum):
    """Types of language models."""

    CHAT = auto()
    TEXT_COMPLETION = auto()
    EMBEDDING = auto()
    MULTIMODAL = auto()
    CODE = auto()
    INSTRUCT = auto()


class ProviderType(Enum):
    """Types of providers."""

    CLOUD_API = auto()
    LOCAL = auto()
    SELF_HOSTED = auto()
    HYBRID = auto()


@dataclass
class ProviderCapabilities:
    """Comprehensive provider capabilities."""

    chat: bool = True
    text_completion: bool = False
    embeddings: bool = False
    streaming: bool = True
    function_calling: bool = False
    tool_use: bool = False
    vision: bool = False
    json_mode: bool = False
    system_prompt: bool = True
    context_window: int = 4096
    max_output_tokens: int = 4096
    supports_multiple_languages: bool = True
    fine_tuning: bool = False
    batch_inference: bool = False
    async_support: bool = True
    rate_limit_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about a specific model."""

    id: str
    name: str
    provider: str
    model_type: ModelType
    capabilities: ProviderCapabilities
    description: str = ""
    pricing_input_per_1k: float = 0.0
    pricing_output_per_1k: float = 0.0
    release_date: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)


class BaseLLMProvider(BaseProvider):
    """Enhanced base class for all LLM providers.

    Extends BaseProvider with additional capabilities for comprehensive provider support.
    """

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Send chat completion request."""
        pass

    @abstractmethod
    async def achat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """Async chat completion."""
        pass

    def complete(
        self, prompt: str, model: Optional[str] = None, **kwargs
    ) -> Union[str, Iterator[str]]:
        """Text completion (non-chat)."""
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, model=model, **kwargs)

        if isinstance(response, dict):
            return response.get("content", "")
        return (chunk.get("content", "") for chunk in response)

    async def acomplete(
        self, prompt: str, model: Optional[str] = None, **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Async text completion."""
        messages = [{"role": "user", "content": prompt}]
        response = await self.achat(messages, model=model, **kwargs)

        if isinstance(response, dict):
            return response.get("content", "")
        return (chunk.get("content", "") async for chunk in response)

    @abstractmethod
    def embed(
        self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings."""
        pass

    @abstractmethod
    async def aembed(
        self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Async embeddings."""
        pass

    def batch_embed(
        self, texts: List[str], model: Optional[str] = None, batch_size: int = 100, **kwargs
    ) -> List[List[float]]:
        """Batch embedding with automatic chunking."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embed(batch, model=model, **kwargs)
            if isinstance(embeddings[0], float):
                all_embeddings.append(embeddings)
            else:
                all_embeddings.extend(embeddings)
        return all_embeddings

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models. Override in subclass."""
        return []

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get info for specific model."""
        for model in self.get_available_models():
            if model.id == model_id:
                return model
        return None

    def validate_config(self) -> bool:
        """Validate provider configuration."""
        if self.provider_type == ProviderType.CLOUD_API:
            return self.api_key is not None or self._get_api_key_from_env() is not None
        return True

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        env_var = f"{self.provider_name.upper().replace('-', '_')}_API_KEY"
        return os.getenv(env_var)

    def _handle_error(self, error: Exception) -> None:
        """Handle provider errors."""
        logger.error(f"{self.provider_name} error: {error}")
        raise


class BaseChatProvider(BaseLLMProvider):
    """Base class for chat model providers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.model or getattr(self, "default_model", None)

    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for provider. Override if needed."""
        return messages

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse provider response into standard format."""
        return {
            "content": response.get("content", ""),
            "model": response.get("model", self.model),
            "provider": getattr(self, "provider_name", "unknown"),
            "usage": response.get("usage", {}),
            "finish_reason": response.get("finish_reason"),
            "tool_calls": response.get("tool_calls", []),
        }

    def embed(
        self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Generate embeddings (not supported by default for chat providers)."""
        raise NotImplementedError(
            f"Embeddings are not supported by {self.provider_name or self.__class__.__name__}"
        )

    async def aembed(
        self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """Async embeddings (not supported by default for chat providers)."""
        raise NotImplementedError(
            f"Async embeddings are not supported by {self.provider_name or self.__class__.__name__}"
        )


class ProviderRegistry:
    """Registry for managing all providers."""

    _providers: Dict[str, Type[BaseLLMProvider]] = {}
    _instances: Dict[str, BaseLLMProvider] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[BaseLLMProvider]) -> None:
        """Register a provider class."""
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered provider: {name}")

    @classmethod
    def get_provider_class(cls, name: str) -> Optional[Type[BaseLLMProvider]]:
        """Get provider class by name."""
        return cls._providers.get(name.lower())

    @classmethod
    def create_provider(cls, name: str, cache: bool = True, **kwargs) -> BaseLLMProvider:
        """Create provider instance."""
        provider_class = cls.get_provider_class(name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {name}. Available: {list(cls._providers.keys())}")

        instance = provider_class(**kwargs)
        if cache:
            cls._instances[name.lower()] = instance
        return instance

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def get_provider_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider."""
        provider_class = cls.get_provider_class(name)
        if not provider_class:
            return None

        # Lazy info without instantiation
        info = {"name": name}
        for attr in ("provider_type", "default_model"):
            val = getattr(provider_class, attr, None)
            if isinstance(val, property):
                continue
            if val is not None:
                info[attr] = getattr(val, "name", val)
        return info


# Convenience functions
def create_provider(name: str, cache: bool = True, **kwargs) -> BaseLLMProvider:
    """Create a provider instance by name."""
    return ProviderRegistry.create_provider(name, cache=cache, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of all available providers."""
    return ProviderRegistry.list_providers()


def get_provider_info(name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a provider."""
    return ProviderRegistry.get_provider_info(name)


def validate_provider_config(name: str, **kwargs) -> bool:
    """Validate configuration for a provider."""
    try:
        provider = create_provider(name, **kwargs)
        return provider.validate_config()
    except Exception as e:
        logger.error(f"Validation error for {name}: {e}")
        return False


def auto_select_provider(prefer_local: bool = False) -> BaseLLMProvider:
    """Automatically select best available provider."""
    # Priority order
    providers_to_try = [
        ("anthropic", {"model": "claude-3-haiku-20240307"}),
        ("google", {"model": "gemini-1.5-flash"}),
        ("groq", {"model": "llama-3.1-8b-instant"}),
        ("ollama", {"model": "llama3.2"}),
    ]

    if prefer_local:
        providers_to_try.insert(0, ("ollama", {"model": "llama3.2"}))

    for name, config in providers_to_try:
        try:
            if validate_provider_config(name, **config):
                return create_provider(name, **config)
        except Exception:
            continue

    raise RuntimeError("No LLM provider available. Please set API key or start Ollama.")


# Backwards compatibility - re-export core types
__all__ = [
    "BaseLLMProvider",
    "BaseChatProvider",
    "ProviderCapabilities",
    "ProviderType",
    "ModelType",
    "ModelInfo",
    "ProviderRegistry",
    "create_provider",
    "get_available_providers",
    "get_provider_info",
    "validate_provider_config",
    "auto_select_provider",
    # Re-export from core for convenience
    "BaseProvider",
    "Message",
    "MessageRole",
    "LLMResponse",
    "StreamingChunk",
    "ToolCall",
    "BaseTool",
]
