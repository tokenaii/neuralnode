"""
Horus provider for NeuralNode.

This module standardizes all Horus-family models around:
- a unified 8K context window
- a single normalized chat template
- consistent response objects
- optional Replica TTS integration
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None

from .base import BaseLLMProvider, ProviderCapabilities, ProviderType
from ..core import LLMResponse, Message, StreamingChunk
from ..turboquant import TURBOQUANT_AVAILABLE, TurboQuant, TurboQuantConfig

logger = logging.getLogger(__name__)

HORUS_CONTEXT_WINDOW = 8192
UNIFIED_CHAT_TEMPLATE_NAME = "horus_unified"
UNIFIED_SYSTEM_PROMPT = (
    "You are Horus, an AI assistant developed by TokenAI. "
    "Be accurate, helpful, concise, and safe."
)


class HorusProvider(BaseLLMProvider):
    """Unified Horus model provider."""

    provider_name = "horus"
    provider_type = ProviderType.LOCAL
    default_model = "tokenaii/horus"

    HORUS_MODELS: Dict[str, Dict[str, Any]] = {
        "tokenaii/horus": {
            "name": "Horus 1.0 4B (Auto)",
            "size": "4B",
            "type": "safetensors",
            "repo_id": "tokenaii/horus",
            "subfolder": "Horus-1.0-4B",
        },
        "tokenaii/horus/Horus-1.0-4B": {
            "name": "Horus 1.0 4B (Full)",
            "size": "4B",
            "type": "safetensors",
            "repo_id": "tokenaii/horus",
            "subfolder": "Horus-1.0-4B",
        },
        "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf": {
            "name": "Horus 1.0 4B (GGUF 4-bit)",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q4_K_M",
            "file_size": "2.78 GB",
        },
        "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q5_K_M.gguf": {
            "name": "Horus 1.0 4B (GGUF 5-bit)",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q5_K_M",
            "file_size": "3.23 GB",
        },
        "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q6_K.gguf": {
            "name": "Horus 1.0 4B (GGUF 6-bit)",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q6_K",
            "file_size": "3.71 GB",
        },
        "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q8_0.gguf": {
            "name": "Horus 1.0 4B (GGUF 8-bit)",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q8_0",
            "file_size": "4.8 GB",
        },
        "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-F16.gguf": {
            "name": "Horus 1.0 4B (GGUF F16)",
            "size": "4B",
            "type": "gguf",
            "quantization": "F16",
            "file_size": "9.83 GB",
        },
    }

    CHAT_TEMPLATES = {
        "default": UNIFIED_CHAT_TEMPLATE_NAME,
        "instruct": UNIFIED_CHAT_TEMPLATE_NAME,
        "chat": UNIFIED_CHAT_TEMPLATE_NAME,
        UNIFIED_CHAT_TEMPLATE_NAME: UNIFIED_CHAT_TEMPLATE_NAME,
    }

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            streaming=True,
            embeddings=False,
            context_window=HORUS_CONTEXT_WINDOW,
            async_support=True,
        )

    def __init__(
        self,
        model_id: str = "tokenaii/horus",
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: Optional[Any] = None,
        bnb_4bit_use_double_quant: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        use_flash_attention: bool = True,
        attn_implementation: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        max_memory: Optional[Dict[str, Any]] = None,
        offload_folder: Optional[str] = None,
        offload_state_dict: bool = False,
        low_cpu_mem_usage: bool = True,
        use_safetensors: bool = True,
        variant: Optional[str] = None,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        proxies: Optional[Dict[str, str]] = None,
        force_download: bool = False,
        resume_download: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = False,
        tts_voice_id: str = "replic-aria-language{en-us}",
        enable_tts: bool = False,
        turboquant: Optional[Union[bool, TurboQuant, TurboQuantConfig, Dict[str, Any]]] = None,
        turboquant_bits: int = 4,
        turboquant_key_bits: Optional[int] = None,
        turboquant_value_bits: Optional[int] = None,
        turboquant_protected_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        if not self._is_gguf_model_id(model_id) and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Horus transformers models. "
                "Install with: pip install transformers torch"
            )

        # `create_provider("horus", model="...")` passes `model`; map it to Horus `model_id`.
        forwarded_model = kwargs.pop("model", None)
        if forwarded_model:
            model_id = str(forwarded_model)

        super().__init__(model=model_id, **kwargs)

        self.model_id = model_id
        self.device = device or self._resolve_device()
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.use_flash_attention = use_flash_attention
        self.attn_implementation = attn_implementation
        self.device_map = device_map
        self.max_memory = max_memory
        self.offload_folder = offload_folder
        self.offload_state_dict = offload_state_dict
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.use_safetensors = use_safetensors
        self.variant = variant
        self.subfolder = subfolder
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.proxies = proxies
        self.force_download = force_download
        self.resume_download = resume_download
        self.tts_voice_id = tts_voice_id
        self.enable_tts = enable_tts
        self.turboquant = turboquant
        self.turboquant_bits = turboquant_bits
        self.turboquant_key_bits = turboquant_key_bits
        self.turboquant_value_bits = turboquant_value_bits
        self.turboquant_protected_layers = turboquant_protected_layers

        self.generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
        }

        self.tokenizer = None
        self.model = None
        self._chat_template = UNIFIED_CHAT_TEMPLATE_NAME
        self._model_info = self._build_model_info()
        self._turboquant_runtime: Optional[TurboQuant] = None

    def _resolve_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_torch_dtype(self, torch_dtype: Optional[Any]) -> Any:
        if torch is None:
            return None
        if torch_dtype is None:
            return torch.float16 if self.device == "cuda" else torch.float32
        if isinstance(torch_dtype, str):
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "auto": "auto",
            }
            return mapping.get(torch_dtype.lower(), torch.float16)
        return torch_dtype

    @staticmethod
    def _is_gguf_model_id(model_id: str) -> bool:
        return model_id.lower().endswith(".gguf")

    def _build_model_info(self) -> Dict[str, Any]:
        base = dict(self.HORUS_MODELS.get(self.model_id, {}))
        base.setdefault("name", self.model_id.split("/")[-1])
        base.setdefault("size", "Unknown")
        base.setdefault("type", "gguf" if self._is_gguf_model_id(self.model_id) else "safetensors")
        base["context_window"] = HORUS_CONTEXT_WINDOW
        base["chat_template"] = UNIFIED_CHAT_TEMPLATE_NAME
        base["recommended_settings"] = {
            "max_new_tokens": self.generation_config["max_new_tokens"],
            "temperature": self.generation_config["temperature"],
            "top_p": self.generation_config["top_p"],
            "top_k": self.generation_config["top_k"],
            "repetition_penalty": self.generation_config["repetition_penalty"],
            "do_sample": self.generation_config["do_sample"],
            "num_beams": self.generation_config["num_beams"],
            "early_stopping": self.generation_config["early_stopping"],
        }
        return base

    def load(self) -> "HorusProvider":
        if self.model is not None:
            return self
        if self._is_gguf_model_id(self.model_id):
            return self._load_gguf()
        return self._load_transformers()

    def _load_gguf(self) -> "HorusProvider":
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF Horus models. "
                "Install with: pip install llama-cpp-python"
            )

        repo_id, filename = self._split_repo_and_filename(self.model_id)
        model_path = filename
        if repo_id and HF_HUB_AVAILABLE:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.cache_dir,
                local_files_only=self.local_files_only,
                token=self.token,
            )

        self.model = Llama(
            model_path=model_path,
            n_ctx=HORUS_CONTEXT_WINDOW,
            verbose=False,
        )
        return self

    def _load_transformers(self) -> "HorusProvider":
        repo_id, auto_subfolder = self._resolve_transformers_source(self.model_id)
        resolved_subfolder = self.subfolder or auto_subfolder
        tokenizer_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "proxies": self.proxies,
            "force_download": self.force_download,
            "resume_download": self.resume_download,
            "use_fast": True,
        }
        if self.token:
            tokenizer_kwargs["token"] = self.token
        if resolved_subfolder:
            tokenizer_kwargs["subfolder"] = resolved_subfolder
        if self.revision:
            tokenizer_kwargs["revision"] = self.revision

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Horus tokenizer from '{repo_id}'. "
                "If this repo is not Transformers-compatible, use a GGUF model id instead."
            ) from exc
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "use_safetensors": self.use_safetensors,
            "proxies": self.proxies,
            "force_download": self.force_download,
            "resume_download": self.resume_download,
        }
        if self.token:
            model_kwargs["token"] = self.token
        if resolved_subfolder:
            model_kwargs["subfolder"] = resolved_subfolder
        if self.revision:
            model_kwargs["revision"] = self.revision
        if self.variant:
            model_kwargs["variant"] = self.variant
        if self.offload_folder:
            model_kwargs["offload_folder"] = self.offload_folder
        if self.offload_state_dict:
            model_kwargs["offload_state_dict"] = self.offload_state_dict
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory

        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=(
                        self.bnb_4bit_compute_dtype or (torch.float16 if self.load_in_4bit else None)
                    ),
                    bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                )
                model_kwargs.setdefault("device_map", "auto")
            except ImportError:
                logger.warning("bitsandbytes is not available; loading Horus without 4/8-bit quantization.")

        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        elif self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Horus transformers model from '{repo_id}'. "
                "This Horus variant may require GGUF runtime; try one of the GGUF model ids."
            ) from exc
        if "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @staticmethod
    def _resolve_transformers_source(model_id: str) -> tuple[str, Optional[str]]:
        info = HorusProvider.HORUS_MODELS.get(model_id, {})
        if info:
            return info.get("repo_id", model_id), info.get("subfolder")
        parts = model_id.split("/")
        if len(parts) <= 2:
            return model_id, None
        # Generic fallback for alias-style model ids like namespace/repo/subfolder.
        return "/".join(parts[:2]), "/".join(parts[2:])

    @staticmethod
    def _split_repo_and_filename(model_id: str) -> tuple[Optional[str], str]:
        parts = model_id.split("/")
        if len(parts) >= 3 and parts[-1].lower().endswith(".gguf"):
            return "/".join(parts[:-1]), parts[-1]
        return None, model_id

    def _normalize_messages(self, messages: Sequence[Union[Dict[str, Any], Message]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, Message):
                normalized.append({"role": message.role.value, "content": message.content})
            elif isinstance(message, dict):
                normalized.append(
                    {
                        "role": str(message.get("role", "user")),
                        "content": str(message.get("content", "")),
                    }
                )
            else:
                raise TypeError(f"Unsupported message type: {type(message)!r}")
        return normalized

    def _render_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        system_messages = [m["content"].strip() for m in messages if m["role"] == "system" and m["content"].strip()]
        system_prompt = "\n".join(system_messages) if system_messages else UNIFIED_SYSTEM_PROMPT

        lines = [f"System: {system_prompt}"]
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            if not content:
                continue
            if role == "system":
                continue
            if role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Tool: {content}")
            else:
                lines.append(f"User: {content}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _resolve_generation_settings(self, **kwargs) -> Dict[str, Any]:
        settings = dict(self.generation_config)
        settings.update({k: v for k, v in kwargs.items() if v is not None})
        return settings

    def _build_turboquant_runtime(self) -> Optional[TurboQuant]:
        config = self.turboquant
        if not config:
            return None
        if not TURBOQUANT_AVAILABLE:
            logger.warning("TurboQuant requested for Horus, but turboquant is not installed.")
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

    def _get_generation_cache(self) -> Optional[Any]:
        if self._is_gguf_model_id(self.model_id):
            return None
        if self._turboquant_runtime is None:
            self._turboquant_runtime = self._build_turboquant_runtime()
        if self._turboquant_runtime is None:
            return None
        try:
            return self._turboquant_runtime.create_cache()
        except Exception as exc:
            logger.warning("TurboQuant cache creation failed for Horus; falling back to standard cache. %s", exc)
            return None

    def _generate_transformers_text(self, prompt: str, **kwargs) -> str:
        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Horus transformers model is not loaded.")

        settings = self._resolve_generation_settings(**kwargs)
        max_new_tokens = int(settings.get("max_new_tokens", 512))
        max_input_tokens = max(HORUS_CONTEXT_WINDOW - max_new_tokens, 1)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_cache = self._get_generation_cache()

        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": settings.get("temperature", 0.7),
                "top_p": settings.get("top_p", 0.9),
                "top_k": settings.get("top_k", 50),
                "repetition_penalty": settings.get("repetition_penalty", 1.1),
                "do_sample": settings.get("do_sample", True),
                "num_beams": settings.get("num_beams", 1),
                "early_stopping": settings.get("early_stopping", False),
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if generation_cache is not None:
                generation_kwargs["past_key_values"] = generation_cache
            outputs = self.model.generate(**generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _generate_gguf_text(self, prompt: str, **kwargs) -> str:
        self.load()
        if self.model is None:
            raise RuntimeError("Horus GGUF model is not loaded.")

        settings = self._resolve_generation_settings(**kwargs)
        output = self.model(
            prompt,
            max_tokens=int(settings.get("max_new_tokens", 512)),
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 0.9),
            top_k=settings.get("top_k", 50),
            repeat_penalty=settings.get("repetition_penalty", 1.1),
            stop=["\nUser:", "\nSystem:", "\nTool:"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        normalized = self._normalize_messages(messages)
        prompt = self._render_prompt(normalized)
        generation_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_tokens or self.generation_config["max_new_tokens"],
            "top_p": top_p if top_p is not None else self.generation_config["top_p"],
            **kwargs,
        }

        if stream:
            return self.stream_chat(normalized, **generation_kwargs)

        if self._is_gguf_model_id(self.model_id):
            content = self._generate_gguf_text(prompt, **generation_kwargs)
        else:
            content = self._generate_transformers_text(prompt, **generation_kwargs)

        return LLMResponse(
            content=content,
            model=model or self.model_id,
            provider=self.provider_name,
        )

    async def achat(self, messages: List[Dict[str, Any]], **kwargs) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def complete(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[StreamingChunk]:
        normalized = self._normalize_messages(messages)
        prompt = self._render_prompt(normalized)

        if self._is_gguf_model_id(self.model_id):
            yield StreamingChunk(content=self._generate_gguf_text(prompt, **kwargs), is_finished=True)
            return

        self.load()
        if self.model is None or self.tokenizer is None or TextIteratorStreamer is None:
            yield StreamingChunk(content=self._generate_transformers_text(prompt, **kwargs), is_finished=True)
            return

        settings = self._resolve_generation_settings(**kwargs)
        max_new_tokens = int(settings.get("max_new_tokens", 512))
        max_input_tokens = max(HORUS_CONTEXT_WINDOW - max_new_tokens, 1)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "temperature": settings.get("temperature", 0.7),
            "top_p": settings.get("top_p", 0.9),
            "top_k": settings.get("top_k", 50),
            "repetition_penalty": settings.get("repetition_penalty", 1.1),
            "do_sample": settings.get("do_sample", True),
            "num_beams": settings.get("num_beams", 1),
            "early_stopping": settings.get("early_stopping", False),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        generation_cache = self._get_generation_cache()
        if generation_cache is not None:
            generation_kwargs["past_key_values"] = generation_cache

        from threading import Thread

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for text in streamer:
            yield StreamingChunk(content=text)
        thread.join()
        yield StreamingChunk(content="", is_finished=True)

    def embed(self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("HorusProvider does not provide embeddings. Use a dedicated embedding model.")

    async def aembed(self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        raise NotImplementedError("HorusProvider does not provide embeddings. Use a dedicated embedding model.")

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            **self._model_info,
            "device": self.device,
            "device_map": self.device_map,
            "torch_dtype": str(self.torch_dtype),
            "loaded": self.model is not None,
            "quantization": {
                "4bit": self.load_in_4bit,
                "8bit": self.load_in_8bit,
                "gguf": self._model_info.get("quantization"),
            },
            "generation_config": dict(self.generation_config),
            "flash_attention": self.use_flash_attention,
            "attn_implementation": self.attn_implementation,
            "context_window": HORUS_CONTEXT_WINDOW,
            "chat_template": UNIFIED_CHAT_TEMPLATE_NAME,
            "tts_enabled": self.enable_tts,
            "tts_voice_id": self.tts_voice_id,
            "turboquant_enabled": bool(self.turboquant),
        }

    def set_chat_template(self, template_name: str):
        if template_name not in self.CHAT_TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(self.CHAT_TEMPLATES.keys())}"
            )
        self._chat_template = UNIFIED_CHAT_TEMPLATE_NAME

    def speak(self, text: str, output_file: Optional[str] = None, voice_id: Optional[str] = None) -> str:
        try:
            from ..replica import ReplicaTTS
        except ImportError as exc:
            raise ImportError("Replica TTS is unavailable. Install edge-tts to enable Horus speech.") from exc

        tts = ReplicaTTS(voice_id=voice_id or self.tts_voice_id)
        return tts.speak(text, output_file=output_file)

    def chat_and_speak(
        self,
        messages: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        voice_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        response = self.chat(messages, **kwargs)
        audio_path = self.speak(response.content, output_file=output_file, voice_id=voice_id)
        return {"response": response, "audio_path": audio_path}

    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        return {model_id: dict(info, context_window=HORUS_CONTEXT_WINDOW) for model_id, info in cls.HORUS_MODELS.items()}


def print_model_list():
    print("=" * 70)
    print("                    Available Horus Models")
    print("=" * 70)
    for model_id, info in HorusProvider.list_available_models().items():
        print(f"Model ID: {model_id}")
        print(f"Name:     {info.get('name', 'Unknown')}")
        print(f"Type:     {info.get('type', 'unknown')}")
        print(f"Context:  {info.get('context_window', HORUS_CONTEXT_WINDOW)} tokens")
        if info.get("quantization"):
            print(f"Quant:    {info['quantization']}")
        if info.get("file_size"):
            print(f"Size:     {info['file_size']}")
        print()
    print("Unified chat template:", UNIFIED_CHAT_TEMPLATE_NAME)
    print("Unified context window:", HORUS_CONTEXT_WINDOW)
    print("=" * 70)


HorusModel = HorusProvider


def load_horus(model_id: str = "tokenaii/horus", **kwargs) -> HorusProvider:
    return HorusProvider(model_id=model_id, **kwargs).load()


__all__ = ["HorusProvider", "HorusModel", "load_horus", "print_model_list"]
