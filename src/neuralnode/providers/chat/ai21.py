"""AI21 Labs Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class AI21ChatProvider(BaseLLMProvider):
    """AI21 Labs Jurassic models provider."""

    provider_name = "ai21"
    provider_type = ProviderType.CLOUD_API
    default_model = "jamba-1.5-large"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=256000,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("AI21_API_KEY")

        if not self.api_key:
            raise ValueError("AI21 API key required. Set AI21_API_KEY environment variable.")

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to AI21."""
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
            "https://api.ai21.com/studio/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "content": data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": data.get("model", model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, **kwargs):
        raise NotImplementedError("AI21 does not provide embeddings API")

    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("AI21 does not provide embeddings API")


AI21Provider = AI21ChatProvider
