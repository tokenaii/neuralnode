"""Plugin SDK for custom LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
import json
import importlib
import inspect
from pathlib import Path

from ..core.types import Message, LLMResponse, StreamingChunk, ToolDefinition


@dataclass
class ProviderMetadata:
    """Metadata for a custom provider."""
    name: str
    version: str
    description: str
    author: str
    requires_auth: bool = True
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_vision: bool = False


class BaseCustomProvider(ABC):
    """Base class for custom LLM providers."""
    
    metadata: ProviderMetadata
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the provider."""
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Send chat completion request."""
        pass
    
    @abstractmethod
    def stream_chat(
        self,
        messages: List[Message],
        **kwargs: Any,
    ):
        """Stream chat completion."""
        pass
    
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Text completion (default: converts to chat)."""
        messages = [Message(role="user", content=prompt)]
        return self.chat(messages, **kwargs)
    
    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Generate embeddings (optional)."""
        raise NotImplementedError("Embedding not supported")
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        return True
    
    def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        return {"status": "unknown", "latency_ms": 0}


class ProviderRegistry:
    """Registry for custom providers."""
    
    _providers: Dict[str, Type[BaseCustomProvider]] = {}
    _instances: Dict[str, BaseCustomProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[BaseCustomProvider]) -> None:
        """Register a provider class."""
        cls._providers[name.lower()] = provider_class
        print(f"Registered provider: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseCustomProvider]]:
        """Get provider class by name."""
        return cls._providers.get(name.lower())
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict] = None) -> Optional[BaseCustomProvider]:
        """Create provider instance."""
        provider_class = cls.get(name)
        if not provider_class:
            return None
        
        instance = provider_class(config)
        cls._instances[name.lower()] = instance
        return instance
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def load_from_directory(cls, directory: str) -> int:
        """Load all providers from directory."""
        count = 0
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return 0
        
        for file in dir_path.glob("*.py"):
            try:
                # Load module
                spec = importlib.util.spec_from_file_location(file.stem, file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find provider classes
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseCustomProvider) and
                        obj != BaseCustomProvider):
                        cls.register(name, obj)
                        count += 1
            
            except Exception as e:
                print(f"Error loading provider from {file}: {e}")
        
        return count


def register_provider(name: str):
    """Decorator to register a provider."""
    def decorator(cls: Type[BaseCustomProvider]):
        ProviderRegistry.register(name, cls)
        return cls
    return decorator


# Example implementations
@register_provider("custom_api")
class CustomAPIProvider(BaseCustomProvider):
    """Example custom API provider."""
    
    metadata = ProviderMetadata(
        name="custom_api",
        version="1.0.0",
        description="Custom REST API provider",
        author="User",
        requires_auth=True,
        supports_streaming=True,
    )
    
    def initialize(self) -> bool:
        """Initialize with API key."""
        self.api_key = self.config.get("api_key")
        self.base_url = self.config.get("base_url", "https://api.example.com")
        self._initialized = bool(self.api_key)
        return self._initialized
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "model": kwargs.get("model", "default"),
        }
        
        response = requests.post(
            f"{self.base_url}/chat",
            headers=headers,
            json=data,
        )
        
        result = response.json()
        
        return LLMResponse(
            text=result.get("text", ""),
            model=result.get("model", "unknown"),
            provider="custom_api",
        )
    
    def stream_chat(self, messages: List[Message], **kwargs):
        """Stream chat response."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        data = {
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "stream": True,
        }
        
        response = requests.post(
            f"{self.base_url}/chat",
            headers=headers,
            json=data,
            stream=True,
        )
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                yield StreamingChunk(
                    text=chunk.get("text", ""),
                    is_finished=chunk.get("done", False),
                )


@register_provider("local_model")
class LocalModelProvider(BaseCustomProvider):
    """Provider for local custom models."""
    
    metadata = ProviderMetadata(
        name="local_model",
        version="1.0.0",
        description="Local custom model provider",
        author="User",
        requires_auth=False,
        supports_streaming=False,
    )
    
    def initialize(self) -> bool:
        """Load local model."""
        model_path = self.config.get("model_path")
        if not model_path:
            return False
        
        # Load model (simplified)
        self.model_path = model_path
        self._initialized = True
        return True
    
    def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Run local model inference."""
        # Simplified - would load and run actual model
        prompt = "\n".join([f"{m.role.value}: {m.content}" for m in messages])
        
        return LLMResponse(
            text=f"Local model response to: {prompt[:50]}...",
            model="local_custom",
            provider="local_model",
        )
    
    def stream_chat(self, messages: List[Message], **kwargs):
        """Streaming not supported for local model."""
        raise NotImplementedError("Streaming not supported")


class PluginManager:
    """Manage provider plugins."""
    
    def __init__(self, plugin_dir: Optional[str] = None):
        self.plugin_dir = plugin_dir or str(Path.home() / ".neuralnode" / "plugins")
        Path(self.plugin_dir).mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
    
    def install_plugin(self, plugin_path: str) -> bool:
        """Install a plugin from file or URL."""
        try:
            import shutil
            dest = Path(self.plugin_dir) / Path(plugin_path).name
            shutil.copy(plugin_path, dest)
            return True
        except Exception as e:
            print(f"Error installing plugin: {e}")
            return False
    
    def load_all(self) -> int:
        """Load all installed plugins."""
        return ProviderRegistry.load_from_directory(self.plugin_dir)
    
    def get_plugin_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get plugin information."""
        provider_class = ProviderRegistry.get(name)
        if not provider_class:
            return None
        
        return {
            "name": provider_class.metadata.name,
            "version": provider_class.metadata.version,
            "description": provider_class.metadata.description,
            "author": provider_class.metadata.author,
            "supports": {
                "streaming": provider_class.metadata.supports_streaming,
                "tools": provider_class.metadata.supports_tools,
                "vision": provider_class.metadata.supports_vision,
            },
        }
    
    def list_installed(self) -> List[str]:
        """List installed plugins."""
        return ProviderRegistry.list_providers()


# Convenience functions
def create_custom_provider(name: str, config: Dict[str, Any]) -> Optional[BaseCustomProvider]:
    """Quick provider creation."""
    return ProviderRegistry.create(name, config)


def list_available_providers() -> List[str]:
    """List all available providers."""
    return ProviderRegistry.list_providers()
