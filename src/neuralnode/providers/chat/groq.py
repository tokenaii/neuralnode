"""Groq Fast Inference Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class GroqChatProvider(BaseLLMProvider):
    """Groq fast inference provider."""

    provider_name = "groq"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-3.1-8b-instant"

    AVAILABLE_MODELS = [
        "llama-3.1-405b-reasoning",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-groq-8b-8192-tool-use-preview",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "gemma-7b-it",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            tool_use=True,
            json_mode=True,
            context_window=131072,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        stream=False,
        tools=None,
        **kwargs,
    ):
        """Send chat request to Groq."""
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

        if stream:
            payload["stream"] = True

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
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
            "tool_calls": message.get("tool_calls", []),
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, **kwargs):
        raise NotImplementedError("Groq does not provide embeddings API")

    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("Groq does not provide embeddings API")


# Alias for backward compatibility
GroqProvider = GroqChatProvider
