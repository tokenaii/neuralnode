"""Cohere Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class CohereChatProvider(BaseLLMProvider):
    """Cohere command models provider."""
    
    provider_name = "cohere"
    provider_type = ProviderType.CLOUD_API
    default_model = "command-r"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            embeddings=True,
            streaming=True,
            context_window=128000,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key required. Set COHERE_API_KEY environment variable.")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Cohere."""
        model = model or self.model
        
        # Cohere uses different format - extract last user message and chat history
        chat_history = []
        message = ""
        
        for msg in messages:
            if msg.get("role") == "user":
                if not message:  # Last user message
                    message = msg.get("content", "")
                else:
                    chat_history.append({"role": "USER", "message": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                chat_history.append({"role": "CHATBOT", "message": msg.get("content", "")})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "message": message or "Hello",
            "chat_history": chat_history,
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            "https://api.cohere.com/v1/chat",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("text", ""),
            "model": model,
            "provider": self.provider_name,
            "usage": {
                "prompt_tokens": data.get("token_count", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("token_count", {}).get("completion_tokens", 0),
            },
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Cohere."""
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model or "embed-english-v3.0",
            "texts": texts,
            "input_type": "search_document",
        }
        
        response = requests.post(
            "https://api.cohere.com/v1/embed",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = data.get("embeddings", [])
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


# Alias for backward compatibility
CohereProvider = CohereChatProvider
