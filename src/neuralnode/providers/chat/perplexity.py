"""Perplexity AI Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class PerplexityChatProvider(BaseLLMProvider):
    """Perplexity AI provider."""

    provider_name = "perplexity"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-3.1-sonar-small-128k-online"

    AVAILABLE_MODELS = [
        "llama-3.1-sonar-huge-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-chat",
        "llama-3.1-sonar-small-128k-chat",
        "llama-3.1-8b-instruct",
        "llama-3.1-70b-instruct",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=128000,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("PERPLEXITY_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY environment variable."
            )

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Perplexity."""
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
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
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
            "citations": data.get("citations", []),
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, **kwargs):
        raise NotImplementedError("Perplexity does not provide embeddings API")

    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("Perplexity does not provide embeddings API")


PerplexityProvider = PerplexityChatProvider
