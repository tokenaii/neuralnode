"""Auto-provider selection based on available API keys and hardware."""

import os
from typing import Optional

from ..core.types import BaseProvider, ProviderType
from ..local.hardware import HardwareDetector


class AutoProvider:
    """Automatically select the best available provider."""

    PRIORITY_ORDER = [
        ProviderType.OPENAI,
        ProviderType.ANTHROPIC,
        ProviderType.GOOGLE,
        ProviderType.OLLAMA,
        ProviderType.HUGGINGFACE,
    ]

    @classmethod
    def select(
        cls,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> BaseProvider:
        """
        Automatically select the best available provider.
        
        Priority:
        1. Cloud providers with API keys (OpenAI, Anthropic, Google)
        2. Local providers (Ollama, HuggingFace)
        
        Args:
            model: Preferred model name
            api_key: Optional API key to use
            
        Returns:
            Configured provider instance
        """
        # Check for cloud providers with API keys
        if api_key or os.getenv("OPENAI_API_KEY"):
            from ..providers.openai import OpenAIProvider
            return OpenAIProvider(model=model or "gpt-4", api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        if os.getenv("ANTHROPIC_API_KEY"):
            from ..providers.anthropic import AnthropicProvider
            return AnthropicProvider(model=model or "claude-3-sonnet-20240229")
        
        if os.getenv("GOOGLE_API_KEY"):
            from ..providers.google import GoogleProvider
            return GoogleProvider(model=model or "gemini-pro")
        
        # Check for local providers
        if cls._is_ollama_available():
            from ..providers.ollama import OllamaProvider
            return OllamaProvider(model=model or "llama3")
        
        # Fallback to HuggingFace local
        from ..providers.huggingface import HuggingFaceProvider
        return HuggingFaceProvider(model=model or "HuggingFaceH4/zephyr-7b-beta")

    @classmethod
    def _is_ollama_available(cls) -> bool:
        """Check if Ollama is running locally."""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def recommend_model(cls, hardware: Optional[HardwareDetector] = None) -> str:
        """
        Recommend a model based on available hardware.
        
        Args:
            hardware: Hardware detector instance
            
        Returns:
            Recommended model name
        """
        if hardware is None:
            hardware = HardwareDetector()
        
        # Check for GPU availability
        if hardware.has_cuda:
            vram_gb = hardware.vram_gb or 0
            if vram_gb >= 24:
                return "llama3.1:70b"
            elif vram_gb >= 16:
                return "llama3.1:8b"
            elif vram_gb >= 8:
                return "qwen2.5:7b"
        
        # CPU fallback
        ram_gb = hardware.ram_gb or 8
        if ram_gb >= 32:
            return "llama3.1:8b"
        elif ram_gb >= 16:
            return "qwen2.5:4b"
        else:
            return "phi3:mini"

    @classmethod
    def get_available_providers(cls) -> list[ProviderType]:
        """Get list of available providers."""
        available = []
        
        if os.getenv("OPENAI_API_KEY"):
            available.append(ProviderType.OPENAI)
        if os.getenv("ANTHROPIC_API_KEY"):
            available.append(ProviderType.ANTHROPIC)
        if os.getenv("GOOGLE_API_KEY"):
            available.append(ProviderType.GOOGLE)
        if cls._is_ollama_available():
            available.append(ProviderType.OLLAMA)
        
        # HuggingFace is always available (local)
        available.append(ProviderType.HUGGINGFACE)
        
        return available
