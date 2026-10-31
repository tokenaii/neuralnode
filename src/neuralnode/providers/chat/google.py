"""Google Gemini Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class GoogleChatProvider(BaseLLMProvider):
    """Google Gemini models provider."""

    provider_name = "google"
    provider_type = ProviderType.CLOUD_API
    default_model = "gemini-1.5-flash"

    GEMINI_MODELS = [
        "gemini-1.5-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.0-pro",
        "gemini-1.0-pro-vision-latest",
        "gemini-ultra",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            streaming=True,
            function_calling=True,
            tool_use=True,
            vision=True,
            json_mode=True,
            system_prompt=True,
            context_window=1000000 if "1.5" in self.model else 32768,
            max_output_tokens=8192,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable.")

    def _format_messages(self, messages):
        """Format messages for Gemini API."""
        gemini_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                metadata = msg.get("metadata") or {}
                parts = [{"text": content}]
                media_type = metadata.get("image_media_type", "image/jpeg")
                parts.extend(
                    {"inline_data": {"mime_type": media_type, "data": image}}
                    for image in metadata.get("images", [])
                )
                if metadata.get("image_url"):
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": media_type,
                                "file_uri": metadata["image_url"],
                            }
                        }
                    )
                gemini_messages.append({"role": "user", "parts": parts})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [{"text": content}]})
            elif role == "system":
                # System prompt is handled separately in Gemini
                pass

        return gemini_messages

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """Send chat request to Gemini."""
        model = model or self.model
        gemini_messages = self._format_messages(messages)

        # Extract system instruction if present
        system_instruction = None
        for msg in messages:
            if msg.get("role") == "system":
                system_instruction = msg.get("content")
                break

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or 4096,
            },
        }

        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        params = {"key": self.api_key}

        response = requests.post(url, params=params, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()

        content = ""
        if data.get("candidates"):
            parts = data["candidates"][0].get("content", {}).get("parts", [])
            content = "".join(part.get("text", "") for part in parts)

        return {
            "content": content,
            "model": model,
            "provider": self.provider_name,
            "usage": {
                "prompt_tokens": data.get("usageMetadata", {}).get("promptTokenCount", 0),
                "completion_tokens": data.get("usageMetadata", {}).get("candidatesTokenCount", 0),
            },
            "finish_reason": data.get("candidates", [{}])[0].get("finishReason"),
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Gemini."""
        if isinstance(texts, str):
            texts = [texts]

        model = model or "models/embedding-001"
        url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"

        embeddings = []
        for text in texts:
            payload = {"content": {"parts": [{"text": text}]}}
            params = {"key": self.api_key}

            response = requests.post(url, params=params, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", {}).get("values", [])
            embeddings.append(embedding)

        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


# Alias for backward compatibility
GoogleProvider = GoogleChatProvider
