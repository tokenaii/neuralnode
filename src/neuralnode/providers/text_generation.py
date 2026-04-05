"""
Text Generation Model Providers for NeuralNode
15+ providers for non-chat text completion models
"""

import os
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator
from abc import abstractmethod

from .base import BaseLLMProvider, ProviderCapabilities, ProviderType, ModelType, ModelInfo, ProviderRegistry

logger = logging.getLogger(__name__)


class BaseTextGenerationProvider(BaseLLMProvider):
    """Base class for text generation (non-chat) providers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.model or self.default_model
    
    @property
    @abstractmethod
    def supports_chat_format(self) -> bool:
        """Whether this provider supports chat-style messages."""
        pass
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert chat messages to a prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)


class HuggingFaceTextProvider(BaseTextGenerationProvider):
    """Hugging Face Inference API for text generation models."""
    
    provider_name = "huggingface-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt2"
    
    POPULAR_MODELS = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-40b",
        "bigscience/bloom",
        "bigscience/bloomz",
        "EleutherAI/gpt-j-6b",
        "EleutherAI/gpt-neox-20b",
        "EleutherAI/pythia-6.9b",
    ]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
        self.use_inference_api = kwargs.get("use_inference_api", True)
        
        if self.use_inference_api and not self.api_key:
            raise ValueError("HuggingFace API key required for Inference API")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text completion using HuggingFace."""
        model = model or self.model
        
        if self.use_inference_api:
            return self._inference_api_complete(prompt, model, temperature, max_tokens, **kwargs)
        else:
            raise NotImplementedError("Local HuggingFace transformers support via separate module")
    
    def _inference_api_complete(self, prompt, model, temperature, max_tokens, **kwargs):
        """Use HuggingFace Inference API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "do_sample": True,
            }
        }
        
        # Add optional parameters
        if "top_p" in kwargs:
            payload["parameters"]["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["parameters"]["top_k"] = kwargs["top_k"]
        if "repetition_penalty" in kwargs:
            payload["parameters"]["repetition_penalty"] = kwargs["repetition_penalty"]
        
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # HF returns a list of generated texts
        if isinstance(data, list) and len(data) > 0:
            generated_text = data[0].get("generated_text", "")
        else:
            generated_text = ""
        
        return {
            "content": generated_text,
            "model": model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        """Convert chat to text completion."""
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("Use HuggingFaceEmbeddingProvider for embeddings")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class AI21TextProvider(BaseTextGenerationProvider):
    """AI21 Labs Jurassic models for text generation."""
    
    provider_name = "ai21-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "j2-ultra"
    
    AVAILABLE_MODELS = ["j2-ultra", "j2-mid", "j2-light", "j2-grande-instruct", "j2-jumbo-instruct"]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=8192,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("AI21_API_KEY")
        
        if not self.api_key:
            raise ValueError("AI21 API key required. Set AI21_API_KEY environment variable.")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using AI21."""
        model = model or self.model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
        
        # Optional parameters
        if "top_p" in kwargs:
            payload["topP"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["topKReturn"] = kwargs["top_k"]
        if "num_results" in kwargs:
            payload["numResults"] = kwargs["num_results"]
        
        response = requests.post(
            f"https://api.ai21.com/studio/v1/{model}/complete",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        completions = data.get("completions", [])
        text = completions[0].get("data", {}).get("text", "") if completions else ""
        
        return {
            "content": text,
            "model": model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class CohereTextProvider(BaseTextGenerationProvider):
    """Cohere Generate API for text generation."""
    
    provider_name = "cohere-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "command"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Cohere API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using Cohere Generate API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if "p" in kwargs:
            payload["p"] = kwargs["p"]
        if "k" in kwargs:
            payload["k"] = kwargs["k"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        
        response = requests.post(
            "https://api.cohere.com/v1/generate",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        generations = data.get("generations", [])
        text = generations[0].get("text", "") if generations else ""
        
        return {
            "content": text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("Use CohereEmbeddingProvider")
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class AlephAlphaTextProvider(BaseTextGenerationProvider):
    """Aleph Alpha Luminous models for text generation."""
    
    provider_name = "alephalpha-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "luminous-base"
    
    LUMINOUS_MODELS = ["luminous-base", "luminous-extended", "luminous-supreme", "luminous-supreme-control"]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=2048,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("ALEPHALPHA_API_KEY")
        
        if not self.api_key:
            raise ValueError("AlephAlpha API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.0, maximum_tokens=64, **kwargs):
        """Generate text using AlephAlpha."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "temperature": temperature,
            "maximum_tokens": maximum_tokens,
        }
        
        response = requests.post(
            "https://api.aleph-alpha.com/complete",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        completions = data.get("completions", [])
        text = completions[0].get("completion", "") if completions else ""
        
        return {
            "content": text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class NLPCloudTextProvider(BaseTextGenerationProvider):
    """NLP Cloud text generation API."""
    
    provider_name = "nlpcloud-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "finetuned-llama-2-70b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("NLPCLOUD_API_KEY")
        
        if not self.api_key:
            raise ValueError("NLPCloud API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using NLPCloud."""
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": prompt,
        }
        
        response = requests.post(
            f"https://api.nlpcloud.io/v1/gpu/{model or self.model}/generation",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("generated_text", ""),
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class GooseAITextProvider(BaseTextGenerationProvider):
    """GooseAI text generation API."""
    
    provider_name = "gooseai-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt-neo-20b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            streaming=True,
            context_window=2048,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("GOOSEAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GooseAI API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using GooseAI."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            "https://api.goose.ai/v1/engines/{}/completions".format(model or self.model),
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        choices = data.get("choices", [])
        text = choices[0].get("text", "") if choices else ""
        
        return {
            "content": text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class PetalsTextProvider(BaseTextGenerationProvider):
    """Petals distributed inference for open models."""
    
    provider_name = "petals-text"
    provider_type = ProviderType.SELF_HOSTED
    default_model = "petals-team/StableBeluga2"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = kwargs.get("petals_host", "http://localhost:8080")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using Petals."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
            }
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("generated_text", ""),
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class TextSynthProvider(BaseTextGenerationProvider):
    """TextSynth API provider."""
    
    provider_name = "textsynth"
    provider_type = ProviderType.CLOUD_API
    default_model = "gptj_6B"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=2048,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("TEXTSYNTH_API_KEY")
        
        if not self.api_key:
            raise ValueError("TextSynth API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using TextSynth."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        response = requests.post(
            f"https://api.textsynth.com/v1/engines/{model or self.model}/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("text", ""),
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class InferKitProvider(BaseTextGenerationProvider):
    """InferKit API provider."""
    
    provider_name = "inferkit"
    provider_type = ProviderType.CLOUD_API
    default_model = "default"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=3000,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("INFERKIT_API_KEY")
        
        if not self.api_key:
            raise ValueError("InferKit API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        """Generate text using InferKit."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "prompt": {"text": prompt},
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
        
        response = requests.post(
            "https://api.inferkit.com/v1/models/default/generate",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("data", {}).get("text", ""),
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class ForefrontProvider(BaseTextGenerationProvider):
    """Forefront AI provider."""
    
    provider_name = "forefront"
    provider_type = ProviderType.CLOUD_API
    default_model = "forefront-llama-2-7b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("FOREFRONT_API_KEY")
        
        if not self.api_key:
            raise ValueError("Forefront API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("Forefront implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class MosaicMLTextProvider(BaseTextGenerationProvider):
    """MosaicML text generation API (now part of Databricks)."""
    
    provider_name = "mosaicml"
    provider_type = ProviderType.CLOUD_API
    default_model = "mpt-7b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("MOSAICML_API_KEY")
        
        if not self.api_key:
            raise ValueError("MosaicML API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("MosaicML implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class RunPodTextProvider(BaseTextGenerationProvider):
    """RunPod serverless text generation."""
    
    provider_name = "runpod-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-2-7b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = kwargs.get("endpoint_id")
        
        if not self.api_key:
            raise ValueError("RunPod API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("RunPod implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class CoreWeaveTextProvider(BaseTextGenerationProvider):
    """CoreWeave cloud text generation."""
    
    provider_name = "coreweave-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-2-70b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("COREWEAVE_API_KEY")
        
        if not self.api_key:
            raise ValueError("CoreWeave API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("CoreWeave implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class LambdaTextProvider(BaseTextGenerationProvider):
    """Lambda Labs text generation API."""
    
    provider_name = "lambda-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "llama-2-70b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=4096,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("LAMBDA_API_KEY")
        
        if not self.api_key:
            raise ValueError("Lambda API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("Lambda implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class PaperspaceTextProvider(BaseTextGenerationProvider):
    """Paperspace Gradient text generation."""
    
    provider_name = "paperspace-text"
    provider_type = ProviderType.CLOUD_API
    default_model = "gpt2"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=False,
            text_completion=True,
            context_window=1024,
            async_support=True,
        )
    
    @property
    def supports_chat_format(self) -> bool:
        return False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("PAPERSPACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Paperspace API key required")
    
    def complete(self, prompt: str, model=None, temperature=0.7, max_tokens=256, **kwargs):
        raise NotImplementedError("Paperspace implementation pending")
    
    def chat(self, messages, **kwargs):
        prompt = self._messages_to_prompt(messages)
        return self.complete(prompt, **kwargs)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    async def acomplete(self, prompt, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(prompt, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


# Register only providers with working implementations.
for provider_class in [
    HuggingFaceTextProvider, AI21TextProvider, CohereTextProvider,
    AlephAlphaTextProvider, NLPCloudTextProvider, GooseAITextProvider,
    PetalsTextProvider, TextSynthProvider, InferKitProvider
]:
    ProviderRegistry.register(provider_class.provider_name, provider_class)


__all__ = [
    "BaseTextGenerationProvider",
    "HuggingFaceTextProvider",
    "AI21TextProvider",
    "CohereTextProvider",
    "AlephAlphaTextProvider",
    "NLPCloudTextProvider",
    "GooseAITextProvider",
    "PetalsTextProvider",
    "TextSynthProvider",
    "InferKitProvider",
]
