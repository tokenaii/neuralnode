"""DeepSeek Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class DeepSeekChatProvider(BaseLLMProvider):
    """DeepSeek provider with embeddings support."""
    
    provider_name = "deepseek"
    provider_type = ProviderType.CLOUD_API
    default_model = "deepseek-chat"
    
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ]
    
    EMBEDDING_MODELS = [
        "deepseek-embedding",
    ]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            json_mode=True,
            embeddings=True,  # Now supported
            context_window=64000,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable.")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to DeepSeek."""
        model = model or self.model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "model": data.get("model", model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
            "finish_reason": data["choices"][0].get("finish_reason"),
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using DeepSeek."""
        if isinstance(texts, str):
            texts = [texts]
        
        model = model or "deepseek-embedding"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "input": texts,
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


# Alias for backward compatibility
DeepSeekProvider = DeepSeekChatProvider
