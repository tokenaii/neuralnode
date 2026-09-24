"""Client for H-Preview served through the NeuralNode model gateway."""

from __future__ import annotations

import os
import json
from typing import Any, Iterator

import requests

from .openai import OpenAICompatibleProvider
from ..base import ProviderCapabilities


class HPreviewProvider(OpenAICompatibleProvider):
    """OpenAI-compatible NeuralNode gateway client for the hosted H-Preview model."""

    provider_name = "h-preview"
    default_model = "h-preview"
    api_key_env_var = "NEURALNODE_API_KEY"
    base_url = "http://127.0.0.1:8000/v1"
    available_models = ["h-preview"]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True, text_completion=True, streaming=True, function_calling=True,
            tool_use=True, vision=False, json_mode=True, embeddings=False,
            system_prompt=True, context_window=131072, max_output_tokens=8192,
            async_support=True,
        )

    def __init__(self, **kwargs: Any) -> None:
        base_url = kwargs.pop("base_url", None) or os.getenv("NEURALNODE_API_BASE_URL")
        super().__init__(**kwargs)
        if base_url:
            self.base_url = base_url.rstrip("/")

    @staticmethod
    def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
        normalized = []
        for message in messages:
            if isinstance(message, dict):
                item = dict(message)
                role = item.get("role", "user")
                item["role"] = getattr(role, "value", role)
            else:
                item = {"role": getattr(message.role, "value", message.role), "content": message.content}
                item.update(message.metadata.get("tool_call") or {})
            normalized.append(item)
        return normalized

    def chat(self, messages: Any, stream: bool = False, **kwargs: Any) -> Any:
        normalized = self._normalize_messages(messages)
        if not stream:
            return super().chat(normalized, stream=False, **kwargs)

        payload = {
            "model": kwargs.pop("model", None) or self.model or self.default_model,
            "messages": normalized,
            "temperature": kwargs.pop("temperature", 0.7),
            "max_tokens": kwargs.pop("max_tokens", None) or 4096,
            "stream": True,
        }
        tools = kwargs.pop("tools", None)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        payload.update(kwargs)
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        response.raise_for_status()

        def chunks() -> Iterator[dict[str, Any]]:
            try:
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        yield {"content": delta["content"]}
            finally:
                response.close()

        return chunks()

    def embed(self, texts: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("H-Preview gateway currently exposes chat completions only.")

    async def aembed(self, texts: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("H-Preview gateway currently exposes chat completions only.")


__all__ = ["HPreviewProvider"]
