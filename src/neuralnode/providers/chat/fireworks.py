"""Fireworks AI Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class FireworksChatProvider(BaseLLMProvider):
    """Fireworks AI provider."""

    provider_name = "fireworks"
    provider_type = ProviderType.CLOUD_API
    default_model = "accounts/fireworks/models/llama-v3p1-8b-instruct"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            context_window=131072,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("FIREWORKS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Fireworks API key required. Set FIREWORKS_API_KEY environment variable."
            )

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Fireworks."""
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
            "https://api.fireworks.ai/inference/v1/chat/completions",
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
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, **kwargs):
        raise NotImplementedError("Fireworks does not provide embeddings API")

    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("Fireworks does not provide embeddings API")


FireworksProvider = FireworksChatProvider
