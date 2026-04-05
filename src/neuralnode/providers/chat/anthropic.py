"""Anthropic Claude Chat Provider for NeuralNode."""

import os
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator

import requests

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType


class AnthropicChatProvider(BaseLLMProvider):
    """Anthropic Claude models provider."""
    
    provider_name = "anthropic"
    provider_type = ProviderType.CLOUD_API
    default_model = "claude-3-haiku-20240307"
    
    CLAUDE3_MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            streaming=True,
            function_calling=True,
            tool_use=True,
            vision="opus" in self.model or "sonnet" in self.model,
            json_mode=False,
            system_prompt=True,
            context_window=200000,
            max_output_tokens=4096,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_url = self.base_url or "https://api.anthropic.com"
        
        if not self.api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")
    
    def _prepare_messages(self, messages: List[Dict[str, Any]]) -> tuple:
        """Extract system message and format rest for Anthropic."""
        system = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            elif msg.get("role") in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system, anthropic_messages
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, tools=None, **kwargs):
        """Send chat request to Anthropic."""
        model = model or self.model
        system, anthropic_messages = self._prepare_messages(messages)
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
            "temperature": temperature,
        }
        
        if system:
            payload["system"] = system
        
        if stream:
            payload["stream"] = True
        
        response = requests.post(
            f"{self.base_url}/v1/messages",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        content = ""
        if data.get("content"):
            content = data["content"][0].get("text", "")
        
        return {
            "content": content,
            "model": data.get("model", model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
            "finish_reason": data.get("stop_reason"),
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("Anthropic does not provide embeddings API")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("Anthropic does not provide embeddings API")


# Alias for backward compatibility
AnthropicProvider = AnthropicChatProvider
