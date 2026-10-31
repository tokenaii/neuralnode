"""Together AI Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class TogetherChatProvider(BaseLLMProvider):
    """Together AI provider - hosts many open source models."""

    provider_name = "together"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

    POPULAR_MODELS = [
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "databricks/dbrx-instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            json_mode=True,
            context_window=128000,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("TOGETHER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Together AI API key required. Set TOGETHER_API_KEY environment variable."
            )

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Together AI."""
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
            "https://api.together.xyz/v1/chat/completions",
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

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Together AI."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or "togethercomputer/m2-bert-80M-8k-retrieval",
            "input": texts,
        }

        response = requests.post(
            "https://api.together.xyz/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


TogetherProvider = TogetherChatProvider
