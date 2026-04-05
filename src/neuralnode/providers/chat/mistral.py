"""Mistral AI Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class MistralChatProvider(BaseLLMProvider):
    """Mistral AI provider."""
    
    provider_name = "mistral"
    provider_type = ProviderType.CLOUD_API
    default_model = "mistral-tiny"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            embeddings=True,
            streaming=True,
            function_calling=True,
            context_window=32768,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Mistral."""
        model = model or self.model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [{"role": m.get("role"), "content": m.get("content")} for m in messages if m.get("role") != "system"],
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        # Add system message if present
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        if system_msg:
            payload["system"] = system_msg.get("content")
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
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
    
    def embed(self, texts, **kwargs):
        """Generate embeddings using Mistral."""
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "mistral-embed",
            "input": texts,
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
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
MistralProvider = MistralChatProvider
