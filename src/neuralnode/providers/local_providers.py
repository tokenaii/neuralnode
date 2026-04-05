"""
Comprehensive Local Model Providers for NeuralNode
15+ local LLM backends and inference engines
"""

import os
import json
import logging
import requests
import subprocess
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator
from abc import abstractmethod
from pathlib import Path

from .base import BaseLLMProvider, ProviderCapabilities, ProviderType, ModelType, ModelInfo, ProviderRegistry
from .chat_models import BaseChatProvider
from ..turboquant import TURBOQUANT_AVAILABLE, TurboQuant, TurboQuantConfig

logger = logging.getLogger(__name__)


class LocalProviderMixin:
    """Mixin for local model providers."""
    
    def check_server_running(self, url: str) -> bool:
        """Check if local server is running."""
        try:
            response = requests.get(f"{url}/health" if not url.endswith('/health') else url, timeout=5)
            return response.status_code == 200
        except:
            return False


class OllamaProvider(BaseChatProvider, LocalProviderMixin):
    """Ollama local LLM server."""
    
    provider_name = "ollama"
    provider_type = ProviderType.LOCAL
    default_model = "llama3.2"
    
    POPULAR_MODELS = [
        "llama3.2", "llama3.1", "llama3", "llama2",
        "mistral", "mixtral", "codellama", "llava",
        "phi3", "qwen2.5", "gemma2", "nemotron",
    ]
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            embeddings=True,
            context_window=128000,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = self.model or self.default_model
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"Ollama server not running at {self.base_url}. Start with: ollama serve")
    
    def _format_messages(self, messages):
        """Format messages for Ollama."""
        system = ""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system = content
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        result = {"prompt": "\n".join(prompt_parts)}
        if system:
            result["system"] = system
        
        return result
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """Send chat request to Ollama."""
        model = model or self.model
        formatted = self._format_messages(messages)
        
        payload = {
            "model": model,
            "prompt": formatted["prompt"],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,
            }
        }
        
        if "system" in formatted:
            payload["system"] = formatted["system"]
        
        if stream:
            return self._stream_chat(payload)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("response", ""),
            "model": model,
            "provider": self.provider_name,
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            "finish_reason": "stop" if data.get("done") else None,
        }
    
    def _stream_chat(self, payload):
        """Stream chat from Ollama."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    text = data.get("response", "")
                    is_done = data.get("done", False)
                    
                    yield {"content": text, "done": is_done}
                    
                    if is_done:
                        break
                except json.JSONDecodeError:
                    continue
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]
        
        model = model or "nomic-embed-text"
        embeddings = []
        
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data.get("embedding", []))
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))
    
    def list_local_models(self) -> List[str]:
        """List models available locally in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False


class LlamaCppProvider(BaseChatProvider, LocalProviderMixin):
    """llama.cpp server provider for GGUF models."""
    
    provider_name = "llamacpp"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("LLAMACPP_URL", "http://localhost:8080")
        self.model_path = kwargs.get("model_path")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"llama.cpp server not running at {self.base_url}")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """Send chat request to llama.cpp server."""
        headers = {"Content-Type": "application/json"}
        
        # llama.cpp server expects standard API format
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        
        if stream:
            return self._stream_chat(headers, payload)
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "model": data.get("model", self.model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
        }
    
    def _stream_chat(self, headers, payload):
        """Stream from llama.cpp."""
        payload["stream"] = True
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        yield {"content": "", "done": True}
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        yield {"content": content}
                    except:
                        continue
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        """Generate embeddings using llama.cpp (if server supports it)."""
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {"Content-Type": "application/json"}
        embeddings = []
        
        for text in texts:
            payload = {
                "input": text,
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/embedding",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embedding", []))
            except Exception as e:
                logger.error(f"llama.cpp embedding failed: {e}")
                raise
        
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class TransformersProvider(BaseChatProvider):
    """Direct HuggingFace Transformers inference (local)."""
    
    provider_name = "transformers"
    provider_type = ProviderType.LOCAL
    default_model = "meta-llama/Llama-2-7b-chat-hf"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        self.device = kwargs.get("device", "auto")
        self.torch_dtype = kwargs.get("torch_dtype", None)
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.load_in_4bit = kwargs.get("load_in_4bit", False)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.turboquant = kwargs.get("turboquant", False)
        self.turboquant_bits = kwargs.get("turboquant_bits", 4)
        self.turboquant_key_bits = kwargs.get("turboquant_key_bits")
        self.turboquant_value_bits = kwargs.get("turboquant_value_bits")
        self.turboquant_protected_layers = kwargs.get("turboquant_protected_layers")
        self._turboquant_runtime = None
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,  # Pipeline doesn't support streaming easily
            context_window=4096,
            async_support=False,
        )
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._pipeline is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
                
                logger.info(f"Loading model {self.model}...")
                
                # Quantization config
                quantization_config = None
                if self.load_in_4bit:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                elif self.load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model,
                    trust_remote_code=self.trust_remote_code,
                )
                
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # Load model
                model_kwargs = {
                    "trust_remote_code": self.trust_remote_code,
                }
                
                if quantization_config:
                    model_kwargs["quantization_config"] = quantization_config
                elif self.torch_dtype:
                    model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
                
                if self.device != "auto":
                    model_kwargs["device_map"] = self.device
                else:
                    model_kwargs["device_map"] = "auto"
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    use_safetensors=True,
                    **model_kwargs
                )
                
                # Create pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self._model,
                    tokenizer=self._tokenizer,
                )
                
                logger.info(f"Model {self.model} loaded successfully")
                
            except ImportError:
                raise ImportError(
                    "transformers, torch, and accelerate required. "
                    "Install: pip install transformers torch accelerate"
                )

    def _build_turboquant_runtime(self):
        config = self.turboquant
        if not config:
            return None
        if not TURBOQUANT_AVAILABLE:
            logger.warning("TurboQuant requested for TransformersProvider, but turboquant is not installed.")
            return None
        if isinstance(config, TurboQuant):
            return config
        if isinstance(config, TurboQuantConfig):
            return TurboQuant(
                bits=config.bits or self.turboquant_bits,
                key_bits=config.key_bits,
                value_bits=config.value_bits,
                protected_layers=config.protected_layers,
                use_polar_quant=config.use_polar_quant,
                use_qjl=config.use_qjl,
            )
        if isinstance(config, dict):
            return TurboQuant(**config)
        if config is True:
            return TurboQuant(
                bits=self.turboquant_bits,
                key_bits=self.turboquant_key_bits,
                value_bits=self.turboquant_value_bits,
                protected_layers=self.turboquant_protected_layers,
            )
        return None

    def _get_generation_cache(self):
        if self._turboquant_runtime is None:
            self._turboquant_runtime = self._build_turboquant_runtime()
        if self._turboquant_runtime is None:
            return None
        try:
            return self._turboquant_runtime.create_cache()
        except Exception as exc:
            logger.warning(
                "TurboQuant cache creation failed for TransformersProvider; falling back to standard cache. %s",
                exc,
            )
            return None
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Generate response using local transformers model."""
        self._load_model()
        
        # Apply chat template if available
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Manual formatting
            prompt_parts = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            prompt_parts.append("Assistant:")
            prompt = "\n\n".join(prompt_parts)
        
        generation_kwargs = dict(kwargs)
        generation_cache = self._get_generation_cache()
        if generation_cache is None:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_tokens or 512,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
                **generation_kwargs
            )
            generated_text = outputs[0]["generated_text"]
        else:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens or 512,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    use_cache=True,
                    past_key_values=generation_cache,
                    **generation_kwargs,
                )
            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self._tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        
        return {
            "content": generated_text,
            "model": self.model,
            "provider": self.provider_name,
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("Use TransformersEmbeddingProvider for embeddings")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class VLLMProvider(BaseChatProvider, LocalProviderMixin):
    """vLLM high-performance inference engine."""
    
    provider_name = "vllm"
    provider_type = ProviderType.LOCAL
    default_model = "meta-llama/Llama-2-7b-hf"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            context_window=4096,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("VLLM_URL", "http://localhost:8000")
        self.api_key = kwargs.get("api_key")  # vLLM can be configured with API key
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"vLLM server not running at {self.base_url}")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """Send request to vLLM API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        
        if stream:
            return self._stream_chat(headers, payload)
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "model": data.get("model", self.model),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
        }
    
    def _stream_chat(self, headers, payload):
        """Stream from vLLM."""
        payload["stream"] = True
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        yield {"content": "", "done": True}
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        yield {"content": content}
                    except:
                        continue
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        """Generate embeddings using vLLM (if enabled)."""
        if isinstance(texts, str):
            texts = [texts]
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class TGIProvider(BaseChatProvider, LocalProviderMixin):
    """HuggingFace Text Generation Inference."""
    
    provider_name = "tgi"
    provider_type = ProviderType.LOCAL
    default_model = "meta-llama/Llama-2-7b-chat-hf"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("TGI_URL", "http://localhost:8080")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"TGI server not running at {self.base_url}")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, stream=False, **kwargs):
        """Send request to TGI."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens or 512,
                "return_full_text": False,
            }
        }
        
        if stream:
            payload["stream"] = True
            return self._stream_generate(payload)
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        generated_text = data.get("generated_text", "")
        
        return {
            "content": generated_text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def _stream_generate(self, payload):
        """Stream from TGI."""
        response = requests.post(
            f"{self.base_url}/generate_stream",
            json=payload,
            stream=True,
            timeout=120
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8').lstrip('data:'))
                    token = data.get("token", {})
                    text = token.get("text", "")
                    yield {"content": text}
                except:
                    continue
    
    def _messages_to_prompt(self, messages):
        """Convert messages to prompt string."""
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError("TGI doesn't support embeddings directly")
    
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class LlamaFileProvider(BaseChatProvider):
    """llamafile - single-file LLM execution."""
    
    provider_name = "llamafile"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llamafile_path = kwargs.get("llamafile_path")
        self.port = kwargs.get("port", 8080)
        self.base_url = f"http://localhost:{self.port}"
        
        if not self.llamafile_path:
            raise ValueError("llamafile_path required (path to .llamafile)")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def start_server(self) -> bool:
        """Start the llamafile server."""
        try:
            subprocess.Popen(
                [self.llamafile_path, "--server", "--port", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start llamafile: {e}")
            return False
    
    def chat(self, messages, **kwargs):
        """Send request to llamafile server."""
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        message = data["choices"][0].get("message", {})
        return {
            "content": message.get("content", ""),
            "model": self.model,
            "provider": self.provider_name,
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class KoboldCppProvider(BaseChatProvider, LocalProviderMixin):
    """KoboldCpp - llama.cpp with GUI and extras."""
    
    provider_name = "koboldcpp"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("KOBOLDCPP_URL", "http://localhost:5001")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"KoboldCpp not running at {self.base_url}")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send request to KoboldCpp."""
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_length": max_tokens or 512,
            "stop_sequence": ["User:", "System:"],
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        text = results[0].get("text", "") if results else ""
        
        return {
            "content": text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class TextGenerationWebuiProvider(BaseChatProvider, LocalProviderMixin):
    """oobabooga text-generation-webui."""
    
    provider_name = "textgenwebui"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("TEXTGENWEBUI_URL", "http://localhost:5000")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"text-generation-webui not running at {self.base_url}")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send request to text-generation-webui."""
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens or 512,
            "stop": ["User:", "System:"],
        }
        
        response = requests.post(
            f"{self.base_url}/api/v1/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        text = results[0].get("text", "") if results else ""
        
        return {
            "content": text,
            "model": model or self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class ExLlamaProvider(BaseChatProvider):
    """ExLlamaV2 - fast inference for GPTQ/AWQ models."""
    
    provider_name = "exllama"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b-gptq"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_directory = kwargs.get("model_directory")
        self.config = None
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        if not self.model_directory:
            raise ValueError("model_directory required (path to quantized model)")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=False,
        )
    
    def _load_model(self):
        """Lazy load ExLlama model."""
        if self.model is None:
            try:
                from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
                from exllamav2.generator import ExLlamaV2StreamingGenerator
                
                self.config = ExLlamaV2Config(self.model_directory)
                self.model = ExLlamaV2(self.config)
                self.tokenizer = ExLlamaV2Tokenizer(self.config)
                self.generator = ExLlamaV2StreamingGenerator(self.model, self.tokenizer)
                
                # Load model
                self.model.load()
                
            except ImportError:
                raise ImportError("exllamav2 required. Install: pip install exllamav2")
    
    def chat(self, messages, **kwargs):
        """Generate with ExLlama."""
        self._load_model()
        
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        
        # Set up generator
        self.generator.set_stop_conditions([])
        
        # Generate
        output = ""
        for chunk in self.generator.generate_simple(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
        ):
            output += chunk
        
        return {
            "content": output,
            "model": self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class AutoGPTQProvider(BaseChatProvider):
    """AutoGPTQ - GPTQ quantization support."""
    
    provider_name = "autogptq"
    provider_type = ProviderType.LOCAL
    default_model = "TheBloke/Llama-2-7B-GPTQ"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self.use_triton = kwargs.get("use_triton", False)
        self.inject_fused_attention = kwargs.get("inject_fused_attention", True)
        self.device = kwargs.get("device", "cuda:0")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,
            context_window=4096,
            async_support=False,
        )
    
    def _load_model(self):
        """Lazy load AutoGPTQ model."""
        if self._model is None:
            try:
                from auto_gptq import AutoGPTQForCausalLM
                from transformers import AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
                self._model = AutoGPTQForCausalLM.from_quantized(
                    self.model,
                    device=self.device,
                    use_triton=self.use_triton,
                    inject_fused_attention=self.inject_fused_attention,
                )
                
            except ImportError:
                raise ImportError("auto-gptq required. Install: pip install auto-gptq")
    
    def chat(self, messages, **kwargs):
        """Generate with AutoGPTQ."""
        self._load_model()
        
        prompt = self._messages_to_prompt(messages)
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True,
        )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        response = generated_text[len(prompt):]
        
        return {
            "content": response,
            "model": self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class AutoAWQProvider(BaseChatProvider):
    """AutoAWQ - AWQ quantization support."""
    
    provider_name = "autoawq"
    provider_type = ProviderType.LOCAL
    default_model = "TheBloke/Llama-2-7B-AWQ"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self.fuse_layers = kwargs.get("fuse_layers", True)
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,
            context_window=4096,
            async_support=False,
        )
    
    def _load_model(self):
        """Lazy load AutoAWQ model."""
        if self._model is None:
            try:
                from awq import AutoAWQForCausalLM
                from transformers import AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
                self._model = AutoAWQForCausalLM.from_quantized(
                    self.model,
                    fuse_layers=self.fuse_layers,
                )
                
            except ImportError:
                raise ImportError("autoawq required. Install: pip install autoawq")
    
    def chat(self, messages, **kwargs):
        """Generate with AutoAWQ."""
        self._load_model()
        
        prompt = self._messages_to_prompt(messages)
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True,
        )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]
        
        return {
            "content": response,
            "model": self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class TensorRTLLMProvider(BaseChatProvider):
    """NVIDIA TensorRT-LLM optimized inference."""
    
    provider_name = "tensorrtllm"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b-trt"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine_path = kwargs.get("engine_path")
        self.tokenizer_path = kwargs.get("tokenizer_path")
        
        if not self.engine_path:
            raise ValueError("engine_path required (path to TensorRT engine)")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, **kwargs):
        """Generate with TensorRT-LLM."""
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import ModelRunner
            
            # This is simplified - actual implementation requires more setup
            raise NotImplementedError("TensorRT-LLM provider requires full implementation with runner setup")
            
        except ImportError:
            raise ImportError("tensorrt-llm required. Install NVIDIA TensorRT-LLM")
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class DeepSpeedProvider(BaseChatProvider):
    """Microsoft DeepSpeed inference."""
    
    provider_name = "deepspeed"
    provider_type = ProviderType.LOCAL
    default_model = "microsoft/DialoGPT-large"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self.ds_config = kwargs.get("ds_config", {})
        self.mp_size = kwargs.get("mp_size", 1)
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,
            context_window=1024,
            async_support=False,
        )
    
    def _load_model(self):
        """Lazy load DeepSpeed model."""
        if self._model is None:
            try:
                import deepspeed
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
                model = AutoModelForCausalLM.from_pretrained(self.model)
                
                # Initialize DeepSpeed
                self._model = deepspeed.init_inference(
                    model,
                    mp_size=self.mp_size,
                    **self.ds_config
                )
                
            except ImportError:
                raise ImportError("deepspeed required. Install: pip install deepspeed")
    
    def chat(self, messages, **kwargs):
        """Generate with DeepSpeed."""
        self._load_model()
        
        prompt = self._messages_to_prompt(messages)
        
        inputs = self._tokenizer(prompt, return_tensors="pt")
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_tokens", 512),
        )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):]
        
        return {
            "content": response,
            "model": self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class RayServeProvider(BaseChatProvider, LocalProviderMixin):
    """Ray Serve for distributed inference."""
    
    provider_name = "rayserve"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("RAY_SERVE_URL", "http://localhost:8000")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"Ray Serve not running at {self.base_url}")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, **kwargs):
        """Send request to Ray Serve deployment."""
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("generated_text", ""),
            "model": self.model,
            "provider": self.provider_name,
        }
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class MLflowProvider(BaseChatProvider):
    """MLflow models serving."""
    
    provider_name = "mlflow"
    provider_type = ProviderType.LOCAL
    default_model = "llm-model"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracking_uri = kwargs.get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.model_uri = kwargs.get("model_uri")
        
        if not self.model_uri:
            raise ValueError("model_uri required (MLflow model URI)")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,
            context_window=4096,
            async_support=False,
        )
    
    def chat(self, messages, **kwargs):
        """Generate with MLflow served model."""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Load model as pyfunc
            model = mlflow.pyfunc.load_model(self.model_uri)
            
            # Prepare input
            prompt = self._messages_to_prompt(messages)
            
            # Predict
            result = model.predict([{"prompt": prompt}])
            
            return {
                "content": result[0] if isinstance(result, list) else result,
                "model": self.model,
                "provider": self.provider_name,
            }
            
        except ImportError:
            raise ImportError("mlflow required. Install: pip install mlflow")
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class BentoMLProvider(BaseChatProvider, LocalProviderMixin):
    """BentoML model serving."""
    
    provider_name = "bentoml"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("BENTOML_URL", "http://localhost:3000")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"BentoML not running at {self.base_url}")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, **kwargs):
        """Send request to BentoML service."""
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "content": data.get("generated_text", ""),
            "model": self.model,
            "provider": self.provider_name,
        }
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class TritonProvider(BaseChatProvider, LocalProviderMixin):
    """NVIDIA Triton Inference Server."""
    
    provider_name = "triton"
    provider_type = ProviderType.LOCAL
    default_model = "llama-2-7b"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("TRITON_URL", "http://localhost:8000")
        self.model_name = kwargs.get("model_name", "ensemble")
        
        if not self.check_server_running(self.base_url):
            logger.warning(f"Triton not running at {self.base_url}")
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            context_window=4096,
            async_support=True,
        )
    
    def chat(self, messages, **kwargs):
        """Send request to Triton server."""
        try:
            import tritonclient.http as httpclient
            from tritonclient.utils import InferenceServerException
            
            prompt = self._messages_to_prompt(messages)
            
            client = httpclient.InferenceServerClient(url=self.base_url.replace("http://", "").replace("https://", ""))
            
            # Prepare input
            input_text = httpclient.InferInput("text_input", [1], "BYTES")
            input_text.set_data_from_numpy(np.array([prompt.encode()], dtype=object))
            
            # Set parameters
            inputs = [input_text]
            outputs = [httpclient.InferRequestedOutput("text_output")]
            
            # Infer
            response = client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
            )
            
            generated_text = response.as_numpy("text_output")[0].decode()
            
            return {
                "content": generated_text,
                "model": self.model,
                "provider": self.provider_name,
            }
            
        except ImportError:
            raise ImportError("tritonclient required. Install: pip install tritonclient[all]")
    
    def _messages_to_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        raise NotImplementedError()
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError()


class MLXProvider(BaseChatProvider, LocalProviderMixin):
    """Apple MLX backend for Mac devices."""
    
    provider_name = "mlx"
    provider_type = ProviderType.LOCAL
    default_model = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=False,
            context_window=32768,
            async_support=True,
        )
        
    def _load_model(self):
        if self._model is None:
            try:
                import mlx.core as mx
                from mlx_lm import load, generate
                self._mlx_generate = generate
                self._model, self._tokenizer = load(self.model)
            except ImportError:
                raise ImportError("MLX required. Install with: pip install mlx mlx-lm")

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        self._load_model()
        
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"<|im_start|>{role}\\n{content}<|im_end|>\\n"
        prompt += "<|im_start|>assistant\\n"
        
        response = self._mlx_generate(
            self._model, 
            self._tokenizer, 
            prompt=prompt, 
            temp=temperature, 
            max_tokens=max_tokens or 4096,
            verbose=False
        )
        
        return {
            "content": response,
            "model": self.model,
            "provider": self.provider_name
        }
        
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
        
    def embed(self, texts, **kwargs):
        raise NotImplementedError("MLX embeddings not implemented")
        
    async def aembed(self, texts, **kwargs):
        raise NotImplementedError("MLX embeddings not implemented")


# Register only local providers with working implementations.
for provider_class in [
    OllamaProvider, LlamaCppProvider, TransformersProvider, VLLMProvider,
    TGIProvider, LlamaFileProvider, KoboldCppProvider, TextGenerationWebuiProvider,
    ExLlamaProvider, AutoGPTQProvider, AutoAWQProvider,
    DeepSpeedProvider, RayServeProvider, MLflowProvider, BentoMLProvider,
    TritonProvider, MLXProvider
]:
    ProviderRegistry.register(provider_class.provider_name, provider_class)


__all__ = [
    "OllamaProvider",
    "LlamaCppProvider",
    "TransformersProvider",
    "LlamaFileProvider",
    "KoboldCppProvider",
    "TextGenerationWebuiProvider",
    "ExLlamaProvider",
    "AutoGPTQProvider",
    "AutoAWQProvider",
    "VLLMProvider",
    "TGIProvider",
    "DeepSpeedProvider",
    "RayServeProvider",
    "MLflowProvider",
    "BentoMLProvider",
    "TritonProvider",
    "MLXProvider",
]
