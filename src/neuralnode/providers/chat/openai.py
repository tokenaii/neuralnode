"""OpenAI-compatible chat providers for NeuralNode."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class OpenAICompatibleProvider(BaseLLMProvider):
    """Base class for OpenAI-compatible chat completion APIs."""

    provider_name = "openai-compatible"
    provider_type = ProviderType.CLOUD_API
    default_model = ""
    api_key_env_var = ""
    base_url = ""
    available_models: List[str] = []

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            streaming=False,
            function_calling=True,
            tool_use=True,
            json_mode=True,
            embeddings=True,
            system_prompt=True,
            context_window=128000,
            max_output_tokens=4096,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv(self.api_key_env_var)
        self.base_url = (self.base_url or kwargs.get("base_url") or "").rstrip("/")
        if not self.api_key:
            raise ValueError(
                f"{self.provider_name} API key required. Set {self.api_key_env_var} "
                "or pass api_key=..."
            )

    def get_custom_headers(self) -> Dict[str, str]:
        return {}

    @staticmethod
    def _payload_messages(messages):
        """Convert NeuralNode messages to JSON-safe OpenAI chat messages."""
        payload = []
        for message in messages:
            if hasattr(message, "to_dict"):
                message = message.to_dict()
            role = message.get("role", "user")
            content = message.get("content", "")
            metadata = message.get("metadata") or {}
            images = metadata.get("images") or []
            image_url = metadata.get("image_url")

            if images or image_url:
                content_parts = [{"type": "text", "text": content}]
                media_type = metadata.get("image_media_type", "image/jpeg")
                content_parts.extend(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image}"},
                    }
                    for image in images
                )
                if image_url:
                    content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                content = content_parts

            payload.append({"role": role, "content": content})
        return payload

    @staticmethod
    def _payload_tools(tools):
        """Convert NeuralNode tools to OpenAI-compatible function schemas."""
        return [tool if isinstance(tool, dict) else tool.get_schema() for tool in tools]

    @staticmethod
    def _normalize_tool_calls(tool_calls):
        """Normalize OpenAI tool-call objects for NeuralNode's response format."""
        normalized = []
        for tool_call in tool_calls or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", tool_call.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            normalized.append(
                {
                    "name": function.get("name") or tool_call.get("name", ""),
                    "arguments": arguments or {},
                    "call_id": tool_call.get("id") or tool_call.get("call_id"),
                }
            )
        return normalized

    def chat(
        self,
        messages,
        model=None,
        temperature=0.7,
        max_tokens=None,
        stream=False,
        tools=None,
        tool_choice=None,
        response_format=None,
        **kwargs,
    ):
        if stream:
            raise NotImplementedError(f"Streaming is not implemented for {self.provider_name} yet.")

        selected_model = model or self.model or self.default_model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.get_custom_headers(),
        }
        payload: Dict[str, Any] = {
            "model": selected_model,
            "messages": self._payload_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        if tools:
            payload["tools"] = self._payload_tools(tools)
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return {
            "content": message.get("content") or "",
            "model": data.get("model", selected_model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": self._normalize_tool_calls(message.get("tool_calls", [])),
        }

    async def achat(self, messages, **kwargs):
        import httpx

        selected_model = kwargs.pop("model", None) or self.model or self.default_model
        payload = {
            "messages": self._payload_messages(messages),
            "model": selected_model,
            **kwargs,
        }
        if payload.get("tools"):
            payload["tools"] = self._payload_tools(payload["tools"])
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.get_custom_headers(),
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return self._parse_completion(data)

    def embed(self, texts, model=None, **kwargs):
        values = [texts] if isinstance(texts, str) else list(texts)
        selected_model = model or kwargs.pop("embedding_model", "text-embedding-3-small")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.get_custom_headers(),
        }
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json={"model": selected_model, "input": values, **kwargs},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if isinstance(texts, str) and embeddings else embeddings

    async def aembed(self, texts, model=None, **kwargs):
        import httpx

        values = [texts] if isinstance(texts, str) else list(texts)
        selected_model = model or kwargs.pop("embedding_model", "text-embedding-3-small")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.get_custom_headers(),
        }
        payload = {"model": selected_model, "input": values, **kwargs}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if isinstance(texts, str) and embeddings else embeddings


class OpenAIChatProvider(OpenAICompatibleProvider):
    provider_name = "openai"
    default_model = "gpt-4o-mini"
    api_key_env_var = "OPENAI_API_KEY"
    base_url = "https://api.openai.com/v1"
    available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]


class OpenRouterChatProvider(OpenAICompatibleProvider):
    provider_name = "openrouter"
    default_model = "auto"
    api_key_env_var = "OPENROUTER_API_KEY"
    base_url = "https://openrouter.ai/api/v1"
    available_models = [
        "auto",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-pro-1.5",
        "meta-llama/llama-3.1-70b-instruct",
        "openai/gpt-4o",
    ]

    def get_custom_headers(self) -> Dict[str, str]:
        return {
            "HTTP-Referer": "https://github.com/tokenaillc/neuralnode",
            "X-Title": "NeuralNode Framework",
        }


class AgentRouterChatProvider(OpenAICompatibleProvider):
    provider_name = "agentrouter"
    default_model = "auto"
    api_key_env_var = "AGENTROUTER_API_KEY"
    base_url = "https://agentrouter.org/v1"
    available_models = ["auto"]


OpenAIProvider = OpenAIChatProvider
OpenRouterProvider = OpenRouterChatProvider
AgentRouterProvider = AgentRouterChatProvider


__all__ = [
    "OpenAICompatibleProvider",
    "OpenAIChatProvider",
    "OpenRouterChatProvider",
    "AgentRouterChatProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "AgentRouterProvider",
]
