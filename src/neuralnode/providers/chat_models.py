"""
Comprehensive Chat Model Providers for NeuralNode
Cloud providers with full feature support
"""

import os
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator
from abc import abstractmethod

from .base import BaseLLMProvider, ProviderCapabilities, ProviderType, ModelType, ModelInfo, ProviderRegistry

logger = logging.getLogger(__name__)


class BaseChatProvider(BaseLLMProvider):
    """Base class for chat model providers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.model or getattr(self, "default_model", None)
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for provider. Override if needed."""
        return messages
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse provider response into standard format."""
        return {
            "content": response.get("content", ""),
            "model": response.get("model", self.model),
            "provider": getattr(self, "provider_name", "unknown"),
            "usage": response.get("usage", {}),
            "finish_reason": response.get("finish_reason"),
            "tool_calls": response.get("tool_calls", []),
        }



class AnthropicChatProvider(BaseChatProvider):
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


class GoogleChatProvider(BaseChatProvider):
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
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
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
            }
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





class CohereChatProvider(BaseChatProvider):
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


class MistralChatProvider(BaseChatProvider):
    """Mistral AI provider."""
    
    provider_name = "mistral"
    provider_type = ProviderType.CLOUD_API
    default_model = "mistral-tiny"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            embeddings=True,
            streaming=True,
            function_calling=True,
            context_window=32768,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        
        if not self.api_key:
            raise ValueError("Mistral API key required. Set MISTRAL_API_KEY environment variable.")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Mistral."""
        model = model or self.model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [{"role": m.get("role"), "content": m.get("content")} for m in messages if m.get("role") != "system"],
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        # Add system message if present
        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        if system_msg:
            payload["system"] = system_msg.get("content")
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
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
        """Generate embeddings using Mistral."""
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "mistral-embed",
            "input": texts,
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class GroqChatProvider(BaseChatProvider):
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
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, tools=None, **kwargs):
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
            timeout=self.timeout
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


class DeepSeekChatProvider(BaseChatProvider):
    """DeepSeek provider."""
    
    provider_name = "deepseek"
    provider_type = ProviderType.CLOUD_API
    default_model = "deepseek-chat"
    
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-coder",
        "deepseek-reasoner",
    ]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            json_mode=True,
            context_window=64000,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DeepSeek API key required. Set DEEPSEEK_API_KEY environment variable.")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to DeepSeek."""
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
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
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
        raise NotImplementedError("DeepSeek does not provide embeddings API")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("DeepSeek does not provide embeddings API")


class PerplexityChatProvider(BaseChatProvider):
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
            raise ValueError("Perplexity API key required. Set PERPLEXITY_API_KEY environment variable.")
    
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
            timeout=self.timeout
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


class AI21ChatProvider(BaseChatProvider):
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
            f"https://api.ai21.com/studio/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
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


class TogetherChatProvider(BaseChatProvider):
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
            raise ValueError("Together AI API key required. Set TOGETHER_API_KEY environment variable.")
    
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
            timeout=self.timeout
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
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class FireworksChatProvider(BaseChatProvider):
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
            raise ValueError("Fireworks API key required. Set FIREWORKS_API_KEY environment variable.")
    
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
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "model": data.get("model", model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("Fireworks embeddings not implemented")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("Fireworks embeddings not implemented")


# Register all providers
ProviderRegistry.register("anthropic", AnthropicChatProvider)
ProviderRegistry.register("cohere", CohereChatProvider)
ProviderRegistry.register("mistral", MistralChatProvider)
ProviderRegistry.register("groq", GroqChatProvider)
ProviderRegistry.register("deepseek", DeepSeekChatProvider)
ProviderRegistry.register("perplexity", PerplexityChatProvider)
ProviderRegistry.register("ai21", AI21ChatProvider)
ProviderRegistry.register("together", TogetherChatProvider)
ProviderRegistry.register("fireworks", FireworksChatProvider)


# Additional providers to implement (placeholders for now)
class AlephAlphaChatProvider(BaseChatProvider):
    provider_name = "alephalpha"
    provider_type = ProviderType.CLOUD_API
    default_model = "luminous-base"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("AlephAlpha implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class AnyscaleChatProvider(BaseChatProvider):
    provider_name = "anyscale"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta-llama/Llama-2-70b-chat-hf"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Anyscale implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class OpenRouterChatProvider(BaseChatProvider):
    provider_name = "openrouter"
    provider_type = ProviderType.CLOUD_API
    default_model = "anthropic/claude-3-haiku"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("OpenRouter implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class HyperbolicChatProvider(BaseChatProvider):
    provider_name = "hyperbolic"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta-llama/Meta-Llama-3.1-405B-Instruct"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Hyperbolic implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class NovitaChatProvider(BaseChatProvider):
    provider_name = "novita"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta-llama/llama-3.1-8b-instruct"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Novita implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class ReplicateChatProvider(BaseChatProvider):
    provider_name = "replicate"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta/meta-llama-3-70b-instruct"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Replicate implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class BedrockChatProvider(BaseChatProvider):
    """AWS Bedrock provider."""
    provider_name = "bedrock"
    provider_type = ProviderType.CLOUD_API
    default_model = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            embeddings=True,
            context_window=200000,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import boto3
            self.session = boto3.Session(
                aws_access_key_id=kwargs.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=kwargs.get("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=kwargs.get("region_name") or os.getenv("AWS_REGION", "us-east-1"),
            )
            self.client = self.session.client("bedrock-runtime")
        except ImportError:
            raise ImportError("boto3 required for AWS Bedrock. Install: pip install boto3")
        except Exception as e:
            raise ValueError(f"Failed to initialize AWS Bedrock client: {e}")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to AWS Bedrock."""
        model = model or self.model
        
        # Format messages for Bedrock
        bedrock_messages = []
        system_prompt = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                bedrock_messages.append({
                    "role": msg.get("role"),
                    "content": [{"type": "text", "text": msg.get("content", "")}]
                })
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or 4096,
            "messages": bedrock_messages,
            "temperature": temperature,
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        import json
        response = self.client.invoke_model(
            modelId=model,
            body=json.dumps(body)
        )
        
        result = json.loads(response["body"].read())
        
        content = ""
        if result.get("content"):
            content = result["content"][0].get("text", "")
        
        return {
            "content": content,
            "model": model,
            "provider": self.provider_name,
            "usage": result.get("usage", {}),
            "finish_reason": result.get("stop_reason"),
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, model=None, **kwargs):
        model = model or "amazon.titan-embed-text-v1"
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            body = {"inputText": text}
            import json
            response = self.client.invoke_model(
                modelId=model,
                body=json.dumps(body)
            )
            result = json.loads(response["body"].read())
            embeddings.append(result.get("embedding", []))
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class SageMakerChatProvider(BaseChatProvider):
    provider_name = "sagemaker"
    provider_type = ProviderType.CLOUD_API
    default_model = "huggingface-llm-falcon-7b-instruct-bf16"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("SageMaker implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class VertexAIChatProvider(BaseChatProvider):
    """Google Cloud Vertex AI provider."""
    provider_name = "vertexai"
    provider_type = ProviderType.CLOUD_API
    default_model = "gemini-1.5-flash"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            vision=True,
            embeddings=True,
            context_window=1000000,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import google.cloud.aiplatform as aiplatform
            project_id = kwargs.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = kwargs.get("location") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            aiplatform.init(project=project_id, location=location)
            self.project_id = project_id
        except ImportError:
            raise ImportError("google-cloud-aiplatform required. Install: pip install google-cloud-aiplatform")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Vertex AI."""
        try:
            from vertexai.generative_models import GenerativeModel
            
            model = model or self.model
            gemini = GenerativeModel(model)
            
            # Convert messages
            chat_history = []
            current_message = ""
            
            for msg in messages:
                if msg.get("role") == "user":
                    current_message = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    chat_history.append({"role": "model", "parts": [msg.get("content", "")]})
            
            chat = gemini.start_chat(history=chat_history)
            response = chat.send_message(
                current_message,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 4096,
                }
            )
            
            return {
                "content": response.text,
                "model": model,
                "provider": self.provider_name,
            }
        except Exception as e:
            raise RuntimeError(f"Vertex AI error: {e}")
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        try:
            from vertexai.language_models import TextEmbeddingModel
            
            if isinstance(texts, str):
                texts = [texts]
            
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
            embeddings = model.get_embeddings(texts)
            
            result = [emb.values for emb in embeddings]
            return result[0] if len(result) == 1 else result
        except Exception as e:
            raise RuntimeError(f"Vertex AI embeddings error: {e}")
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class WatsonChatProvider(BaseChatProvider):
    provider_name = "watson"
    provider_type = ProviderType.CLOUD_API
    default_model = "ibm/granite-13b-chat-v2"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Watson implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class NLPCloudChatProvider(BaseChatProvider):
    provider_name = "nlpcloud"
    provider_type = ProviderType.CLOUD_API
    default_model = "finetuned-llama-3-70b"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("NLPCloud implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class GooseAIChatProvider(BaseChatProvider):
    provider_name = "gooseai"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt-neo-20b"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("GooseAI implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class PetalsChatProvider(BaseChatProvider):
    provider_name = "petals"
    provider_type = ProviderType.SELF_HOSTED
    default_model = "petals-team/StableBeluga2"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Petals implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class BasetenChatProvider(BaseChatProvider):
    provider_name = "baseten"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-2-7b"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Baseten implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class ModalChatProvider(BaseChatProvider):
    provider_name = "modal"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-2-70b"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Modal implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class VercelChatProvider(BaseChatProvider):
    provider_name = "vercel"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt-4-turbo"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Vercel AI SDK implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class GradientChatProvider(BaseChatProvider):
    provider_name = "gradient"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama2-7b"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Gradient AI implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class SambaNovaChatProvider(BaseChatProvider):
    provider_name = "sambanova"
    provider_type = ProviderType.CLOUD_API
    default_model = "Meta-Llama-3.1-405B-Instruct"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("SambaNova implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class FriendliChatProvider(BaseChatProvider):
    provider_name = "friendli"
    provider_type = ProviderType.CLOUD_API
    default_model = "meta-llama-3-8b-instruct"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Friendli implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class ClarifaiChatProvider(BaseChatProvider):
    provider_name = "clarifai"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama2-7b-chat"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Clarifai implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class SegmindChatProvider(BaseChatProvider):
    provider_name = "segmind"
    provider_type = ProviderType.CLOUD_API
    default_model = "sdxl"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("Segmind implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


class AIMLAPIChatProvider(BaseChatProvider):
    provider_name = "aimlapi"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt-4"
    
    def chat(self, messages, **kwargs):
        raise NotImplementedError("AIMLAPI implementation pending")
    async def achat(self, messages, **kwargs):
        raise NotImplementedError()
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()
    @property
    def capabilities(self):
        return ProviderCapabilities()


# Register only providers with working implementations.
for provider_class in [
    BedrockChatProvider,
    VertexAIChatProvider,
]:
    ProviderRegistry.register(provider_class.provider_name, provider_class)


__all__ = [
    "BaseChatProvider",
    "AnthropicChatProvider",
    "GoogleChatProvider",
    "CohereChatProvider",
    "MistralChatProvider",
    "GroqChatProvider",
    "DeepSeekChatProvider",
    "PerplexityChatProvider",
    "AI21ChatProvider",
    "TogetherChatProvider",
    "FireworksChatProvider",
    "BedrockChatProvider",
    "VertexAIChatProvider",
]
