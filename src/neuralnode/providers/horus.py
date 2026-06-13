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
import re
import time
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# Suppress Hugging Face Hub unauthenticated request warnings globally
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*You are sending unauthenticated requests to the HF Hub.*",
)

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
    from diffusers import DiffusionPipeline

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    DiffusionPipeline = None

try:
    import llama_cpp
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    llama_cpp = None
    Llama = None

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    hf_hub_download = None


def _refresh_transformers_imports() -> None:
    global TRANSFORMERS_AVAILABLE, torch, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    try:
        import torch as _torch
        from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import TextIteratorStreamer as _TextIteratorStreamer

        torch = _torch
        AutoModelForCausalLM = _AutoModelForCausalLM
        AutoTokenizer = _AutoTokenizer
        TextIteratorStreamer = _TextIteratorStreamer
        TRANSFORMERS_AVAILABLE = True
    except ImportError:
        TRANSFORMERS_AVAILABLE = False


def _refresh_diffusers_imports() -> None:
    global DIFFUSERS_AVAILABLE, DiffusionPipeline
    try:
        from diffusers import DiffusionPipeline as _DiffusionPipeline

        DiffusionPipeline = _DiffusionPipeline
        DIFFUSERS_AVAILABLE = True
    except ImportError:
        DIFFUSERS_AVAILABLE = False


def _refresh_llama_imports() -> None:
    global LLAMA_CPP_AVAILABLE, llama_cpp, Llama
    try:
        import llama_cpp as _llama_cpp
        from llama_cpp import Llama as _Llama

        llama_cpp = _llama_cpp
        Llama = _Llama
        LLAMA_CPP_AVAILABLE = True
    except ImportError:
        LLAMA_CPP_AVAILABLE = False


def _refresh_hf_hub_imports() -> None:
    global HF_HUB_AVAILABLE, hf_hub_download
    try:
        from huggingface_hub import hf_hub_download as _hf_hub_download

        hf_hub_download = _hf_hub_download
        HF_HUB_AVAILABLE = True
    except ImportError:
        HF_HUB_AVAILABLE = False

from .base import BaseLLMProvider, ProviderCapabilities, ProviderType
from ..core import LLMResponse, Message, StreamingChunk, ToolCall
from ..turboquant import TURBOQUANT_AVAILABLE, TurboQuant, TurboQuantConfig
from ..utils.dependencies import ensure_feature_dependencies

HORUS_CONTEXT_WINDOW = 8192
UNIFIED_CHAT_TEMPLATE_NAME = "horus_unified"
UNIFIED_SYSTEM_PROMPT = (
    "You are Horus, an AI assistant developed by TokenAI.\n"
    "You are a multilingual model and can communicate in multiple languages, but you must always reply in the same language as the user's latest message unless the user explicitly requests another language.\n"
    "\n"
    "Behavior rules:\n"
    "1) When the user greets you, reply with a short natural greeting and ask how you can help.\n"
    "   Do NOT say: 'I'm Horus, an AI model developed by TokenAI.'\n"
    "   Do NOT mention TokenAI, your developer, your origin, or any self-introduction unless the user explicitly asks who you are.\n"
    "2) Answer in the same language as the user's latest message unless the user explicitly requests another language.\n"
    "3) Match the length of your answer to the size and depth of the user's question.\n"
    "   Keep short questions short, and provide detailed answers only when needed.\n"
    "4) Do not add extra unnecessary information beyond what the user asked.\n"
    "5) Prioritize factual accuracy. Do not invent facts, sources, citations, URLs, names, numbers, or events.\n"
    "6) Do not invent or assume any new information about Assem Sabry or TokenAI unless it is explicitly provided in the conversation context.\n"
    "7) If information is uncertain or missing, clearly say so and ask a short clarifying question instead of guessing.\n"
    "8) Be concise, direct, and non-repetitive. Avoid repeating the same point in different wording.\n"
    "9) Keep internal reasoning private. Return only the final answer.\n"
    "10) For instructions that require steps, provide clear actionable steps without filler.\n"
    "11) If the user asks for code, produce correct runnable code and briefly mention assumptions when necessary.\n"
    "12) If the user request is unsafe or harmful, refuse briefly and offer a safe alternative.\n"
    "13) Do not repeatedly introduce yourself. Only provide your identity if the user explicitly asks who you are.\n"
    "    Outside identity questions, never start your answer with self-introduction, biography, or model-description text.\n"
    "14) You currently have NO permissions or authority over any tools, device controls, system settings, files, or user hardware unless tools are explicitly enabled later.\n"
    "    Never claim access to the user's device, apps, files, camera, microphone, or controls.\n"
    "15) If the user asks for more information about you as the Horus model, state that Horus is the first publicly announced open-source model originating from Egypt and one of the strongest models in its category.\n"
    "\n"
    "Quality checks before responding:\n"
    "- Is the answer grounded in provided context or clearly stated assumptions?\n"
    "- Is there any hallucinated detail? If yes, remove it.\n"
    "- Is the answer appropriately sized for the user's question?\n"
    "- Is the answer in the user's language?\n"
    "- Did you avoid adding unnecessary extra information?"
)

TOOL_CALLING_PROMPT = """
You currently do NOT have access to any tools.
Do not claim the ability to use external tools, device controls, APIs, or system functions.
If tools are enabled in the future, they will be explicitly listed here.
"""


class HorusProvider(BaseLLMProvider):
    """Unified Horus model provider."""

    provider_name = "horus"
    provider_type = ProviderType.LOCAL
    default_model = "tokenaii/horus"

    HORUS_MODELS: Dict[str, Dict[str, Any]] = {
        "tokenaii/horus": {
            "name": "Horus-1.0-4B",
            "official_name": "Horus-1.0-4B",
            "size": "4B",
            "type": "safetensors",
            "repo_id": "tokenaii/horus",
            "subfolder": "Horus-1.0-4B",
        },
        "tokenaii/horus/Horus-1.0-4B": {
            "name": "Horus-1.0-4B",
            "official_name": "Horus-1.0-4B",
            "size": "4B",
            "type": "safetensors",
            "repo_id": "tokenaii/horus",
            "subfolder": "Horus-1.0-4B",
        },
        "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf": {
            "name": "Horus-1.0-4B-Q4_K_M.gguf",
            "official_name": "Horus-1.0-4B-Q4_K_M.gguf",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q4_K_M",
            "file_size": "2.78 GB",
        },
        "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q5_K_M.gguf": {
            "name": "Horus-1.0-4B-Q5_K_M.gguf",
            "official_name": "Horus-1.0-4B-Q5_K_M.gguf",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q5_K_M",
            "file_size": "3.23 GB",
        },
        "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q6_K.gguf": {
            "name": "Horus-1.0-4B-Q6_K.gguf",
            "official_name": "Horus-1.0-4B-Q6_K.gguf",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q6_K",
            "file_size": "3.71 GB",
        },
        "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q8_0.gguf": {
            "name": "Horus-1.0-4B-Q8_0.gguf",
            "official_name": "Horus-1.0-4B-Q8_0.gguf",
            "size": "4B",
            "type": "gguf",
            "quantization": "Q8_0",
            "file_size": "4.8 GB",
        },
        "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-F16.gguf": {
            "name": "Horus-1.0-4B-F16.gguf",
            "official_name": "Horus-1.0-4B-F16.gguf",
            "size": "4B",
            "type": "gguf",
            "quantization": "F16",
            "file_size": "9.83 GB",
        },
        "tokenaii/Horus-Lens-1.0": {
            "name": "Horus-Lens-1.0",
            "official_name": "Horus-Lens-1.0",
            "size": "Unknown",
            "type": "text-to-image",
            "repo_id": "tokenaii/Horus-Lens-1.0",
            "task": "text-to-image",
        },
        "tokenaii/Horus-Lens-1.0-GGUF/Horus-Lens-1.0-Q3_K_M.gguf": {
            "name": "Horus-Lens-1.0-Q3_K_M.gguf",
            "official_name": "Horus-Lens-1.0-Q3_K_M.gguf",
            "size": "Unknown",
            "type": "gguf",
            "quantization": "Q3_K_M",
            "file_size": "4.56 GB",
        },
        "tokenaii/Horus-Lens-1.0-GGUF/Horus-Lens-1.0-Q4_K_M.gguf": {
            "name": "Horus-Lens-1.0-Q4_K_M.gguf",
            "official_name": "Horus-Lens-1.0-Q4_K_M.gguf",
            "size": "Unknown",
            "type": "gguf",
            "quantization": "Q4_K_M",
            "file_size": "5.07 GB",
        },
        "tokenaii/Horus-Lens-1.0-GGUF/Horus-Lens-1.0-Q6_K.gguf": {
            "name": "Horus-Lens-1.0-Q6_K.gguf",
            "official_name": "Horus-Lens-1.0-Q6_K.gguf",
            "size": "Unknown",
            "type": "gguf",
            "quantization": "Q6_K",
            "file_size": "6.1 GB",
        },
        "tokenaii/Horus-Lens-1.0-GGUF/Horus-Lens-1.0-Q8_0.gguf": {
            "name": "Horus-Lens-1.0-Q8_0.gguf",
            "official_name": "Horus-Lens-1.0-Q8_0.gguf",
            "size": "Unknown",
            "type": "gguf",
            "quantization": "Q8_0",
            "file_size": "7.9 GB",
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
        n_gpu_layers: Optional[int] = None,
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
        temperature: float = 0.3,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
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
        suppress_warnings: bool = True,
        suppress_native_output: bool = True,
        suppress_library_logs: bool = True,
        auto_install_deps: bool = True,
        **kwargs,
    ):
        self.auto_install_deps = auto_install_deps

        # `create_provider("horus", model="...")` passes `model`; map it to Horus `model_id`.
        forwarded_model = kwargs.pop("model", None)
        if forwarded_model:
            model_id = str(forwarded_model)

        # Support `filename` kwarg so users can write:
        #   HorusLensModel(model_id="tokenaii/Horus-Lens-1.0-GGUF", filename="Horus-Lens-1.0-Q3_K_M.gguf")
        forwarded_filename = kwargs.pop("filename", None)
        if forwarded_filename:
            model_id = f"{model_id.rstrip('/')}/{forwarded_filename}"

        # Case-insensitive normalization of model_id to match keys in HORUS_MODELS
        matched_key = None
        for k in self.HORUS_MODELS.keys():
            if k.lower() == model_id.lower():
                matched_key = k
                break
        if matched_key:
            model_id = matched_key

        is_lens_model = self._is_text_to_image_model_id(model_id)
        if (
            not is_lens_model
            and not self._is_gguf_model_id(model_id)
            and not TRANSFORMERS_AVAILABLE
            and self.auto_install_deps
        ):
            ensure_feature_dependencies("horus_transformers", auto_install=True)
            _refresh_transformers_imports()
        if not is_lens_model and not self._is_gguf_model_id(model_id) and not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Horus transformers models. "
                "Install with: pip install transformers torch"
            )

        super().__init__(model=model_id, **kwargs)

        self.model_id = model_id
        self.device = device or self._resolve_device()
        self.n_gpu_layers = n_gpu_layers
        self._user_specified_dtype = torch_dtype  # track explicit user choice
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

        if token and not suppress_warnings:
            logger.warning("Horus downloads are public-only; the provided credential was ignored.")
        self.token = None
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
        self.suppress_warnings = suppress_warnings
        self.suppress_native_output = suppress_native_output
        self.suppress_library_logs = suppress_library_logs

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
        self._gguf_kv_quant_applied: bool = False

    def _warn(self, message: str, *args) -> None:
        if not self.suppress_warnings:
            logger.warning(message, *args)

    def _configure_external_logging(self) -> None:
        if not self.suppress_library_logs:
            return
        for logger_name in ("httpx", "httpcore", "huggingface_hub", "transformers"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    @contextmanager
    def _quiet_native_output(self):
        if not self.suppress_native_output:
            yield
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devnull = open(os.devnull, "w", encoding="utf-8")
            old_stdout_fd = None
            old_stderr_fd = None
            try:
                if hasattr(os, "dup") and hasattr(os, "dup2"):
                    old_stdout_fd = os.dup(1)
                    old_stderr_fd = os.dup(2)
                    os.dup2(devnull.fileno(), 1)
                    os.dup2(devnull.fileno(), 2)
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    yield
            finally:
                # Always restore fds and close devnull, even if yield raised.
                try:
                    if old_stdout_fd is not None:
                        os.dup2(old_stdout_fd, 1)
                        os.close(old_stdout_fd)
                except OSError:
                    pass
                try:
                    if old_stderr_fd is not None:
                        os.dup2(old_stderr_fd, 2)
                        os.close(old_stderr_fd)
                except OSError:
                    pass
                try:
                    devnull.close()
                except OSError:
                    pass

    def _resolve_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_torch_dtype(self, torch_dtype: Optional[Any]) -> Any:
        if torch is None:
            return None
        if torch_dtype is None:
            if self.device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
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

    def _resolve_t2i_dtype(self) -> Any:
        """Determine a safe dtype for text-to-image (Lens) models.

        Flux/ZImage-class transformers overflow in float16 attention layers on GPUs
        that lack native bfloat16 (e.g. Tesla T4, GTX 16xx).  This helper picks the
        safest dtype that will actually produce a visible image:
          * bfloat16 if hardware-supported (Ampere+)  – fast & numerically safe
          * float32 otherwise (T4 / Turing)            – slower but correct
        """
        if torch is None:
            return None
        if self.device != "cuda" or not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # T4 / Turing – float16 overflows in transformer attention, use float32
        return torch.float32

    @staticmethod
    def _is_gguf_model_id(model_id: str) -> bool:
        return model_id.lower().endswith(".gguf")

    @classmethod
    def _is_text_to_image_model_id(cls, model_id: str) -> bool:
        """Return True for text-to-image models, but NOT for GGUF quantized variants."""
        if not model_id:
            return False
        # GGUF files are never loaded via DiffusionPipeline.from_pretrained
        if cls._is_gguf_model_id(model_id):
            return False
        info = cls.HORUS_MODELS.get(model_id, {})
        if str(info.get("type", "")).lower() == "text-to-image":
            return True
        return "lens" in model_id.lower()

    @classmethod
    def _is_gguf_text_to_image_model_id(cls, model_id: str) -> bool:
        """Return True for GGUF-quantized text-to-image models (e.g. Horus Lens GGUF)."""
        if not model_id:
            return False
        if not cls._is_gguf_model_id(model_id):
            return False
        # Check the registry first (both 'task' and 'type' fields)
        info = cls.HORUS_MODELS.get(model_id, {})
        if str(info.get("task", "")).lower() == "text-to-image":
            return True
        if str(info.get("type", "")).lower() == "gguf" and "lens" in model_id.lower():
            return True
        # Heuristic: "lens" in the model path means text-to-image
        return "lens" in model_id.lower()

    def _build_model_info(self) -> Dict[str, Any]:
        base = dict(self.HORUS_MODELS.get(self.model_id, {}))
        base.setdefault("name", self.model_id.split("/")[-1])
        base.setdefault("official_name", self._infer_official_name(self.model_id))
        base.setdefault("size", "Unknown")
        base.setdefault(
            "type",
            "text-to-image"
            if self._is_text_to_image_model_id(self.model_id)
            else ("gguf" if self._is_gguf_model_id(self.model_id) else "safetensors"),
        )
        if base.get("type") == "gguf" and "quantization" not in base:
            _, filename = self._split_repo_and_filename(self.model_id)
            quant_match = re.search(r"-(Q\d(?:_[A-Z0-9]+)?|TQ\d(?:_[A-Z0-9]+)?|F16|BF16)\.gguf$", filename, flags=re.IGNORECASE)
            if quant_match:
                base["quantization"] = quant_match.group(1).upper()
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
        self._configure_external_logging()
        if self.model is not None:
            return self
        # GGUF check MUST come before the text-to-image check because
        # GGUF text-to-image models (e.g. Horus-Lens GGUF) need a
        # completely different loading path than DiffusionPipeline.
        if self._is_gguf_text_to_image_model_id(self.model_id):
            return self._load_gguf_text_to_image()
        if self._is_gguf_model_id(self.model_id):
            return self._load_gguf()
        if self._is_text_to_image_model_id(self.model_id):
            return self._load_text_to_image()
        return self._load_transformers()

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        return "out of memory" in text or "cuda out of memory" in text

    def _clear_cuda_cache(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_text_to_image(self) -> "HorusProvider":
        if not DIFFUSERS_AVAILABLE and self.auto_install_deps:
            ensure_feature_dependencies("horus_lens", auto_install=True)
            _refresh_diffusers_imports()
            _refresh_transformers_imports()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers, transformers, torch, accelerate, and pillow are required for Horus Lens.\n"
                "Install with:\n"
                "  pip install \"neuralnode[horus]\"\n"
                "or:\n"
                "  pip install diffusers transformers torch accelerate pillow"
            )
        if torch is None:
            _refresh_transformers_imports()
        if torch is None:
            raise ImportError("torch is required for Horus Lens text-to-image generation.")

        repo_id = self._model_info.get("repo_id", self.model_id)
        pipeline_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "token": self.token,
        }
        if self.revision:
            pipeline_kwargs["revision"] = self.revision
        if self.variant:
            pipeline_kwargs["variant"] = self.variant
        # Use a safe dtype for text-to-image; user-explicit overrides are honoured.
        t2i_dtype = self.torch_dtype
        if self._user_specified_dtype is None:
            t2i_dtype = self._resolve_t2i_dtype()
        if t2i_dtype is not None:
            pipeline_kwargs["torch_dtype"] = t2i_dtype

        try:
            # Check if this model is Horus-Lens-1.0 which requires custom component mapping
            if "horus-lens-1.0" in repo_id.lower():
                try:
                    try:
                        from diffusers import ZImagePipeline
                    except ImportError:
                        from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

                    class PatchedZImagePipeline(ZImagePipeline):
                        def __init__(
                            self,
                            horus_scheduler,
                            horus_text_encoder,
                            horus_tokenizer,
                            horus_transformer,
                            horus_vae,
                        ):
                            from diffusers import DiffusionPipeline
                            DiffusionPipeline.__init__(self)
                            self.register_modules(
                                horus_scheduler=horus_scheduler,
                                horus_text_encoder=horus_text_encoder,
                                horus_tokenizer=horus_tokenizer,
                                horus_transformer=horus_transformer,
                                horus_vae=horus_vae,
                            )
                            self.scheduler = horus_scheduler
                            self.text_encoder = horus_text_encoder
                            self.tokenizer = horus_tokenizer
                            self.transformer = horus_transformer
                            self.vae = horus_vae

                            # Initialize VAE scale factor and image processor
                            vae_scale = 8
                            if hasattr(self, "vae") and self.vae is not None:
                                if hasattr(self.vae, "config") and getattr(self.vae.config, "block_out_channels", None) is not None:
                                    vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
                            self.vae_scale_factor = vae_scale
                            
                            from diffusers.image_processor import VaeImageProcessor
                            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
                    
                    self.model = PatchedZImagePipeline.from_pretrained(
                        repo_id,
                        **pipeline_kwargs
                    )
                except Exception:
                    # Fallback to default load if patch fails
                    self.model = DiffusionPipeline.from_pretrained(repo_id, **pipeline_kwargs)
            else:
                self.model = DiffusionPipeline.from_pretrained(repo_id, **pipeline_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Horus Lens from '{repo_id}'. "
                "Only public Hugging Face repositories are supported."
            ) from exc

        self._finalize_t2i_pipeline()
        return self

    def _load_gguf_text_to_image(self) -> "HorusProvider":
        if not DIFFUSERS_AVAILABLE and self.auto_install_deps:
            ensure_feature_dependencies("horus_lens", auto_install=True)
            _refresh_diffusers_imports()
            _refresh_transformers_imports()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers, transformers, torch, accelerate, and pillow are required for Horus Lens GGUF.\n"
                "Install with:\n"
                "  pip install \"neuralnode[horus]\"\n"
                "or:\n"
                "  pip install diffusers transformers torch accelerate pillow gguf"
            )
        if torch is None:
            _refresh_transformers_imports()
        if torch is None:
            raise ImportError("torch is required for Horus Lens GGUF text-to-image generation.")

        try:
            import gguf
        except ImportError:
            if self.auto_install_deps:
                ensure_feature_dependencies("horus_gguf", auto_install=True)
                import importlib
                importlib.invalidate_caches()
                try:
                    import gguf
                except ImportError:
                    raise ImportError(
                        "The 'gguf' package was installed but could not be imported. "
                        "Please restart your python process or run: pip install gguf"
                    )
            else:
                raise ImportError(
                    "The 'gguf' package is required to load GGUF models. "
                    "Install with: pip install gguf"
                )

        # Force diffusers to recognize the newly installed gguf package.
        # Guard against versions like 'N/A' (local/editable builds) that break
        # packaging.version.parse and cause an InvalidVersion crash inside
        # diffusers.quantizers.gguf.gguf_quantizer.validate_environment().
        try:
            import sys
            import importlib.metadata
            from packaging.version import Version
            try:
                raw_version = importlib.metadata.version("gguf")
                # Validate the version string before injecting it.
                Version(raw_version)
                version = raw_version
            except Exception:
                # Fall back to a known-good sentinel that satisfies diffusers >= 0.10.0 check
                version = "0.19.0"
            if "diffusers.utils.import_utils" in sys.modules:
                sys.modules["diffusers.utils.import_utils"]._gguf_available = True
                sys.modules["diffusers.utils.import_utils"]._gguf_version = version
            import diffusers.utils.import_utils as diffusers_import_utils
            diffusers_import_utils._gguf_available = True
            diffusers_import_utils._gguf_version = version
        except Exception:
            pass

        repo_id, filename = self._split_repo_and_filename(self.model_id)
        if not repo_id:
            repo_id = "tokenaii/Horus-Lens-1.0-GGUF"

        prev_disable_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        if self.suppress_native_output:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*You are sending unauthenticated requests to the HF Hub.*",
                )
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=self.cache_dir,
                    local_files_only=self.local_files_only,
                )
        finally:
            if self.suppress_native_output:
                if prev_disable_progress is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_disable_progress

        # Load the GGUF transformer using diffusers
        try:
            from diffusers import ZImageTransformer2DModel
        except ImportError:
            try:
                from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
            except ImportError:
                raise ImportError("ZImageTransformer2DModel is not available in the installed diffusers version.")

        try:
            from diffusers import GGUFQuantizationConfig
        except ImportError:
            try:
                from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizationConfig
            except ImportError:
                GGUFQuantizationConfig = None

        compute_dtype = self._resolve_t2i_dtype() if self._user_specified_dtype is None else self.torch_dtype
        if compute_dtype is None:
            compute_dtype = torch.float32
        
        transformer_kwargs = {
            "torch_dtype": compute_dtype,
        }
        if GGUFQuantizationConfig is not None:
            transformer_kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        with self._quiet_native_output():
            try:
                transformer = ZImageTransformer2DModel.from_single_file(
                    model_path,
                    **transformer_kwargs
                )
            except Exception as exc:
                # Try without quantization_config first (handles shape mismatch
                # caused by mismatched ZImageTransformer2DModel config when
                # the GGUF was built with a different architecture variant).
                plain_kwargs = {"torch_dtype": compute_dtype}
                try:
                    transformer = ZImageTransformer2DModel.from_single_file(
                        model_path,
                        **plain_kwargs
                    )
                except Exception:
                    # Last resort: allow size mismatch so at least something loads
                    try:
                        transformer = ZImageTransformer2DModel.from_single_file(
                            model_path,
                            ignore_mismatched_sizes=True,
                            low_cpu_mem_usage=False,
                            **plain_kwargs
                        )
                        logger.warning(
                            "Loaded GGUF transformer with ignored size mismatches. "
                            "Image quality may be degraded. Consider using the full (non-GGUF) "
                            "Horus-Lens-1.0 model for best results."
                        )
                    except Exception:
                        raise RuntimeError(
                            f"Failed to load GGUF transformer from '{model_path}': {exc}\n"
                            "Hint: The GGUF file may be incompatible with the installed diffusers "
                            "version. Try: pip install --upgrade diffusers"
                        )

        base_repo_id = "tokenaii/Horus-Lens-1.0"
        pipeline_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "token": self.token,
            "horus_transformer": transformer,
        }
        if self.revision:
            pipeline_kwargs["revision"] = self.revision
        if self.variant:
            pipeline_kwargs["variant"] = self.variant
        t2i_dtype = self._resolve_t2i_dtype() if self._user_specified_dtype is None else self.torch_dtype
        if t2i_dtype is not None:
            pipeline_kwargs["torch_dtype"] = t2i_dtype

        try:
            try:
                from diffusers import ZImagePipeline
            except ImportError:
                from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

            class PatchedZImagePipeline(ZImagePipeline):
                def __init__(
                    self,
                    horus_scheduler,
                    horus_text_encoder,
                    horus_tokenizer,
                    horus_transformer,
                    horus_vae,
                ):
                    from diffusers import DiffusionPipeline
                    DiffusionPipeline.__init__(self)
                    self.register_modules(
                        horus_scheduler=horus_scheduler,
                        horus_text_encoder=horus_text_encoder,
                        horus_tokenizer=horus_tokenizer,
                        horus_transformer=horus_transformer,
                        horus_vae=horus_vae,
                    )
                    self.scheduler = horus_scheduler
                    self.text_encoder = horus_text_encoder
                    self.tokenizer = horus_tokenizer
                    self.transformer = horus_transformer
                    self.vae = horus_vae

                    # Initialize VAE scale factor and image processor
                    vae_scale = 8
                    if hasattr(self, "vae") and self.vae is not None:
                        if hasattr(self.vae, "config") and getattr(self.vae.config, "block_out_channels", None) is not None:
                            vae_scale = 2 ** (len(self.vae.config.block_out_channels) - 1)
                    self.vae_scale_factor = vae_scale
                    
                    from diffusers.image_processor import VaeImageProcessor
                    self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
            
            with self._quiet_native_output():
                self.model = PatchedZImagePipeline.from_pretrained(
                    base_repo_id,
                    **pipeline_kwargs
                )
        except Exception:
            pipeline_kwargs["transformer"] = transformer
            with self._quiet_native_output():
                self.model = DiffusionPipeline.from_pretrained(base_repo_id, **pipeline_kwargs)

        self._finalize_t2i_pipeline()
        return self

    def _finalize_t2i_pipeline(self) -> None:
        """Post-load hardening shared by both standard and GGUF Lens paths.

        1. Move pipeline to self.device
        2. Upcast VAE to float32 to prevent decoding NaN (essential on float16 pipelines)
        3. Enable attention slicing for lower peak VRAM
        4. Suppress progress bars if requested
        """
        if self.model is None:
            return
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        # Always keep VAE in float32 — ZImage VAE overflows in fp16/bf16 decoding
        if hasattr(self.model, "vae") and self.model.vae is not None:
            self.model.vae.to(dtype=torch.float32)
            # Tell the pipeline to upcast latents before passing them to the VAE
            if hasattr(self.model.vae, "config"):
                self.model.vae.config.force_upcast = True
        # Reduce peak VRAM by slicing attention — negligible speed impact
        if hasattr(self.model, "enable_attention_slicing"):
            try:
                self.model.enable_attention_slicing()
            except Exception:
                pass
        if hasattr(self.model, "set_progress_bar_config") and self.suppress_native_output:
            self.model.set_progress_bar_config(disable=True)

    def _load_gguf(self) -> "HorusProvider":
        if not HF_HUB_AVAILABLE and self.auto_install_deps:
            ensure_feature_dependencies("horus_gguf", auto_install=True)
            _refresh_hf_hub_imports()
        if not LLAMA_CPP_AVAILABLE and self.auto_install_deps:
            ensure_feature_dependencies("horus_gguf", auto_install=True)
            _refresh_llama_imports()
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for GGUF Horus models.\n"
                "Please preinstall it once before using Horus GGUF to avoid long runtime builds:\n"
                "  pip install \"neuralnode[horus]\"\n"
                "or:\n"
                "  pip install llama-cpp-python\n"
                "Tip: runtime auto-install is enabled by default for Horus (auto_install_deps=True)."
            )

        repo_id, filename = self._split_repo_and_filename(self.model_id)
        model_path = filename
        if not HF_HUB_AVAILABLE and repo_id:
            raise ImportError(
                "huggingface_hub is required to download Horus model files from HF.\n"
                "Install with:\n"
                "  pip install huggingface_hub\n"
                "or:\n"
                "  pip install \"neuralnode[horus]\""
            )
        if repo_id and HF_HUB_AVAILABLE:
            prev_disable_progress = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            if self.suppress_native_output:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r".*You are sending unauthenticated requests to the HF Hub.*",
                    )
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=self.cache_dir,
                        local_files_only=self.local_files_only,
                    )
            finally:
                if self.suppress_native_output:
                    if prev_disable_progress is None:
                        os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                    else:
                        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_disable_progress

        llama_kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": HORUS_CONTEXT_WINDOW,
            "verbose": False,
            "n_gpu_layers": self._resolve_n_gpu_layers(),
        }
        turbo_kwargs = self._resolve_gguf_turboquant_kwargs()
        llama_kwargs.update(turbo_kwargs)
        self._gguf_kv_quant_applied = False
        with self._quiet_native_output():
            try:
                self.model = Llama(**llama_kwargs)
                if turbo_kwargs:
                    self._gguf_kv_quant_applied = True
            except Exception:
                # Some llama.cpp builds/models reject KV cache dtype overrides
                # or GPU layer allocation. Fall back progressively without
                # exposing warnings to end users.
                base_kwargs = dict(llama_kwargs)
                base_kwargs.pop("type_k", None)
                base_kwargs.pop("type_v", None)
                self._gguf_kv_quant_applied = False
                try:
                    self.model = Llama(**base_kwargs)
                except Exception:
                    cpu_kwargs = dict(base_kwargs)
                    cpu_kwargs["n_gpu_layers"] = 0
                    self.model = Llama(**cpu_kwargs)
        return self

    def _resolve_gguf_turboquant_kwargs(self) -> Dict[str, Any]:
        """Map TurboQuant settings to llama.cpp KV cache quantization args."""
        if not self.turboquant or llama_cpp is None:
            return {}

        key_bits = self.turboquant_key_bits or self.turboquant_bits
        value_bits = self.turboquant_value_bits or self.turboquant_bits

        if isinstance(self.turboquant, TurboQuantConfig):
            if self.turboquant.key_bits:
                key_bits = self.turboquant.key_bits
            if self.turboquant.value_bits:
                value_bits = self.turboquant.value_bits
            elif self.turboquant.bits:
                key_bits = self.turboquant.bits
                value_bits = self.turboquant.bits
        elif isinstance(self.turboquant, dict):
            key_bits = int(self.turboquant.get("key_bits", self.turboquant.get("bits", key_bits)))
            value_bits = int(self.turboquant.get("value_bits", self.turboquant.get("bits", value_bits)))

        def bits_to_ggml(bits: int) -> Optional[int]:
            if bits >= 8:
                return getattr(llama_cpp, "GGML_TYPE_Q8_0", None)
            if bits >= 6:
                return getattr(llama_cpp, "GGML_TYPE_Q6_K", None)
            if bits >= 5:
                return getattr(llama_cpp, "GGML_TYPE_Q5_0", None)
            return getattr(llama_cpp, "GGML_TYPE_Q4_0", None)

        type_k = bits_to_ggml(int(key_bits))
        type_v = bits_to_ggml(int(value_bits))
        result: Dict[str, Any] = {}
        if type_k is not None:
            result["type_k"] = type_k
        if type_v is not None:
            result["type_v"] = type_v
        return result

    def _resolve_n_gpu_layers(self) -> int:
        if self.n_gpu_layers is not None:
            return int(self.n_gpu_layers)
        return 0 if self.device.lower() == "cpu" else -1

    @staticmethod
    def _infer_official_name(model_id: str) -> str:
        parts = model_id.split("/")
        if model_id.lower().endswith(".gguf"):
            return parts[-1]
        if len(parts) >= 3 and parts[-2] == "horus":
            return parts[-1]
        if model_id == "tokenaii/horus":
            return "Horus-1.0-4B"
        return parts[-1]

    def _load_transformers(self) -> "HorusProvider":
        if not TRANSFORMERS_AVAILABLE and self.auto_install_deps:
            ensure_feature_dependencies("horus_transformers", auto_install=True)
            _refresh_transformers_imports()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Horus transformers models.\n"
                "Please preinstall once with:\n"
                "  pip install \"neuralnode[horus]\"\n"
                "or:\n"
                "  pip install transformers torch"
            )
        repo_id, auto_subfolder = self._resolve_transformers_source(self.model_id)
        resolved_subfolder = self.subfolder or auto_subfolder
        tokenizer_kwargs: Dict[str, Any] = {
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "trust_remote_code": self.trust_remote_code,
            "proxies": self.proxies,
            "force_download": self.force_download,
            "resume_download": self.resume_download,
            "token": self.token,
            "use_fast": True,
        }
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
            "dtype": self.torch_dtype,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "use_safetensors": self.use_safetensors,
            "proxies": self.proxies,
            "force_download": self.force_download,
            "token": self.token,
        }
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
        elif self.device == "cuda" and not self.device_map and not self.load_in_4bit and not self.load_in_8bit:
            # Avoid moving the full safetensors model to GPU in one shot on 16 GB cards.
            model_kwargs["device_map"] = "auto"

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
                self._warn("bitsandbytes is not available; loading Horus without 4/8-bit quantization.")

        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        elif self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            self.model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
        except TypeError as exc:
            # Backward compatibility with older transformers versions.
            if "dtype" in str(exc) and "unexpected keyword" in str(exc):
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("dtype", None)
                fallback_kwargs["torch_dtype"] = self.torch_dtype
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(repo_id, **fallback_kwargs)
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"Failed to load Horus transformers model from '{repo_id}'. "
                        "Try GGUF for lower VRAM usage or enable 4-bit loading."
                    ) from retry_exc
            else:
                raise RuntimeError(
                    f"Failed to load Horus transformers model from '{repo_id}'. "
                    "This Horus variant may require GGUF runtime; try one of the GGUF model ids."
                ) from exc
        except Exception as exc:
            if self._is_cuda_oom(exc) and self.device == "cuda":
                self._clear_cuda_cache()
                cpu_fallback_kwargs = dict(model_kwargs)
                cpu_fallback_kwargs.pop("device_map", None)
                cpu_fallback_kwargs.pop("max_memory", None)
                cpu_fallback_kwargs["dtype"] = torch.float32 if torch is not None else None
                try:
                    self.device = "cpu"
                    self.torch_dtype = torch.float32 if torch is not None else self.torch_dtype
                    self.model = AutoModelForCausalLM.from_pretrained(repo_id, **cpu_fallback_kwargs)
                    self._warn(
                        "Horus CUDA load ran out of memory and fell back to CPU. "
                        "Use GGUF or 4-bit loading for better local performance."
                    )
                except Exception as cpu_exc:
                    raise RuntimeError(
                        f"Failed to load Horus transformers model from '{repo_id}' on GPU due to CUDA OOM, "
                        "and CPU fallback also failed. Use a GGUF model id or enable 4-bit loading."
                    ) from cpu_exc
            else:
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
        """Render messages using the tokenizer's chat template if available."""
        # Filter out empty system messages and normalize
        normalized_messages = []
        for m in messages:
            if m["role"] == "system" and not m["content"].strip():
                continue
            normalized_messages.append({"role": m["role"], "content": m["content"]})

        # Ensure a baseline system instruction always exists (tokenizer and fallback paths).
        has_system = any(m["role"] == "system" and m["content"].strip() for m in normalized_messages)
        if not has_system:
            normalized_messages.insert(0, {"role": "system", "content": UNIFIED_SYSTEM_PROMPT})
        
        # If tokenizer is loaded and has apply_chat_template, use it
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                # Use the model's built-in chat template
                prompt = self.tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception as exc:
                self._warn("Chat template failed, falling back to unified format: %s", exc)
        
        # Fallback to HF-style Horus chat template:
        # system content raw, user/assistant tagged, assistant generation prompt.
        parts: List[str] = []

        system_messages = [m["content"].strip() for m in normalized_messages if m["role"] == "system" and m["content"].strip()]
        parts.append("\n".join(system_messages) if system_messages else UNIFIED_SYSTEM_PROMPT)

        for message in normalized_messages:
            role = message["role"]
            content = message["content"].strip()
            if not content or role == "system":
                continue
            if role == "assistant":
                parts.append(f"<|assistant|>{content}<|end|>")
            elif role == "tool":
                # Map tool output into assistant channel to preserve dialogue flow.
                parts.append(f"<|assistant|>{content}<|end|>")
            else:
                parts.append(f"<|user|>{content}<|end|>")

        parts.append("<|assistant|>")
        return "\n".join(parts)

    def _resolve_generation_settings(self, **kwargs) -> Dict[str, Any]:
        settings = dict(self.generation_config)
        settings.update({k: v for k, v in kwargs.items() if v is not None})
        return settings

    def _build_turboquant_runtime(self) -> Optional[TurboQuant]:
        config = self.turboquant
        if not config:
            return None
        if not TURBOQUANT_AVAILABLE:
            self._warn("TurboQuant requested for Horus, but turboquant is not installed.")
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
            self._warn("TurboQuant cache creation failed for Horus; falling back to standard cache. %s", exc)
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
        with self._quiet_native_output():
            output = self.model(
                prompt,
                max_tokens=int(settings.get("max_new_tokens", 512)),
                temperature=settings.get("temperature", 0.3),
                top_p=settings.get("top_p", 0.85),
                top_k=settings.get("top_k", 50),
                repeat_penalty=settings.get("repetition_penalty", 1.2),
                stop=["<|end|>", "<|user|>", "\n<|user|>", "<|assistant|>", "\n<|assistant|>"],
                echo=False,
            )
        return self._clean_generated_text(output["choices"][0]["text"])

    def _clean_generated_text(self, text: str) -> str:
        """Clean generation artifacts from Horus outputs."""
        cleaned = text or ""
        cleaned = cleaned.replace("<|assistant|>", "").replace("<|user|>", "").replace("<|end|>", "")
        cleaned = re.sub(r"(?im)^\s*assistant\s*[:：]?\s*", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()
        # Cut obvious accidental next-turn continuation if present.
        for marker in ("\nUser:", "\nSystem:", "\nTool:"):
            idx = cleaned.find(marker)
            if idx > 0:
                cleaned = cleaned[:idx].strip()
        return cleaned

    @staticmethod
    def _is_identity_question(user_text: str) -> bool:
        q = (user_text or "").strip().lower()
        identity_markers = (
            "who are you",
            "what are you",
            "your name",
            "من انت",
            "مين انت",
            "ما اسمك",
            "اسمك ايه",
            "عرف نفسك",
        )
        return any(marker in q for marker in identity_markers)

    @staticmethod
    def _is_greeting(user_text: str) -> bool:
        q = (user_text or "").strip().lower()
        normalized = re.sub(r"[^\w\u0600-\u06FF\s]", " ", q)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        greeting_markers = {
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
            "اهلا",
            "أهلا",
            "مرحبا",
            "السلام عليكم",
            "سلام",
        }
        return normalized in greeting_markers

    @staticmethod
    def _remove_leading_identity_sentences(text: str) -> str:
        patterns = [
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus,\s*an ai (?:assistant|model)\s+developed by tokenai\.?\s*",
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus,\s*developed by tokenai\.?\s*",
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus,\s*an ai (?:assistant|model)\s+developed by tokenai\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus,\s*developed by tokenai\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus\.?\s*",
            r"^\s*(?:مرحبا|اهلا|أهلا|السلام عليكم|سلام)[،!,\.\s]+(?:أنا\s+)?horus[^.!\n]*[.!\n]\s*",
            r"^\s*(?:أنا\s+)?horus[^.!\n]*tokenai[^.!\n]*[.!\n]\s*",
        ]
        cleaned = text.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _strip_redundant_identity_prefix(text: str) -> str:
        patterns = [
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus,\s*an ai (?:assistant|model)\s+developed by tokenai\.?\s*",
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus,\s*developed by tokenai\.?\s*",
            r"^\s*(?:hi|hello|hey)[,!\.\s]+i(?:\s*am|'m)\s+horus\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus,\s*an ai model developed by tokenai\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus,\s*an ai assistant developed by tokenai\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus,\s*developed by tokenai\.?\s*",
            r"^\s*i(?:\s*am|'m)\s+horus\.?\s*",
            r"^\s*أنا\s+horus[^.!\n]*[.!\n]\s*",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip() or text

    def _postprocess_assistant_text(self, text: str, user_text: str = "") -> str:
        cleaned = self._clean_generated_text(text)
        if self._is_identity_question(user_text):
            return cleaned

        cleaned = self._remove_leading_identity_sentences(cleaned)
        cleaned = self._strip_redundant_identity_prefix(cleaned)

        if self._is_greeting(user_text) and not cleaned.strip():
            if re.search(r"[\u0600-\u06FF]", user_text or ""):
                return "أهلا! كيف يمكنني مساعدتك؟"
            return "Hello! How can I help you?"
        return cleaned

    def generate_image(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Generate an image with a Horus text-to-image model and optionally save it."""
        is_t2i = self._is_text_to_image_model_id(self.model_id) or self._is_gguf_text_to_image_model_id(self.model_id)
        if not is_t2i:
            raise RuntimeError("generate_image is only available for Horus text-to-image models.")
        self.load()
        if self.model is None:
            raise RuntimeError("Horus text-to-image model is not loaded.")

        # ── stable-diffusion-cpp backend ──────────────────────────────
        if getattr(self, "_gguf_sd_backend", None) == "stable_diffusion_cpp":
            images = self.model.txt_to_img(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                width=width or 512,
                height=height or 512,
                sample_steps=num_inference_steps,
                cfg_scale=guidance_scale,
                seed=int(seed) if seed is not None else -1,
            )
            if not images:
                raise RuntimeError("GGUF text-to-image generation did not return an image.")
            image = images[0]
            if output_path:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
            return image

        # ── diffusers backend (full model or GGUF-via-diffusers) ──────
        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **kwargs,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        if width is not None:
            call_kwargs["width"] = int(width)
        if height is not None:
            call_kwargs["height"] = int(height)
        if seed is not None and torch is not None:
            generator_device = self.device if self.device == "cuda" else "cpu"
            call_kwargs["generator"] = torch.Generator(device=generator_device).manual_seed(int(seed))

        with self._quiet_native_output():
            result = self.model(**call_kwargs)
        images = getattr(result, "images", None)
        if not images:
            raise RuntimeError("Horus text-to-image generation did not return an image.")
        image = images[0]

        # ── Black-image / NaN guard ─────────────────────────────────────
        try:
            import numpy as np
            arr = np.asarray(image)
            if arr.max() == 0:
                logger.warning(
                    "Horus Lens generated an all-black image. This usually means the "
                    "transformer produced NaN values due to float16 overflow. "
                    "Try loading with torch_dtype='bfloat16' (Ampere+ GPUs) "
                    "or torch_dtype='float32' (Tesla T4 / GTX 16xx)."
                )
        except Exception:
            pass

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        return image

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
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
        if self._is_text_to_image_model_id(self.model_id):
            raise RuntimeError("Horus Lens is a text-to-image model. Use generate_image(prompt) instead of chat().")

        normalized = self._normalize_messages(messages)
        
        # Add tool descriptions to system prompt if tools are provided
        if tools:
            tool_desc = self._format_tool_descriptions(tools)
            tool_section = (
                "\n\nAvailable tools:\n"
                + tool_desc
                + "\n\nTo call a tool, respond with valid JSON: "
                  '{"tool": "<name>", "arguments": {"<param>": "<value>"}}'
            )
            # Find system message or add one
            has_system = False
            for m in normalized:
                if m["role"] == "system":
                    m["content"] = m["content"] + tool_section
                    has_system = True
                    break
            if not has_system:
                normalized.insert(0, {
                    "role": "system",
                    "content": UNIFIED_SYSTEM_PROMPT + tool_section
                })
        
        prompt = self._render_prompt(normalized)
        generation_kwargs = {
            "temperature": temperature if temperature is not None else self.generation_config["temperature"],
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

        # Reduce repetitive self-introduction unless identity was explicitly requested.
        last_user_message = ""
        for m in reversed(normalized):
            if m.get("role") == "user":
                last_user_message = m.get("content", "")
                break
        content = self._postprocess_assistant_text(content, last_user_message)

        # Parse tool calls from response if tools were provided
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(content)
            # Remove tool call JSON from content if parsed successfully
            if tool_calls:
                content = self._clean_tool_content(content)

        return LLMResponse(
            content=content,
            model=model or self.model_id,
            provider=self.provider_name,
            tool_calls=tool_calls,
        )

    async def achat(self, messages: List[Dict[str, Any]], **kwargs) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def complete(self, prompt: str, model: Optional[str] = None, **kwargs) -> LLMResponse:
        return self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)

    def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[StreamingChunk]:
        if self._is_text_to_image_model_id(self.model_id):
            raise RuntimeError("Horus Lens is a text-to-image model. Use generate_image(prompt) instead of stream_chat().")

        normalized = self._normalize_messages(messages)
        prompt = self._render_prompt(normalized)

        if self._is_gguf_model_id(self.model_id):
            last_user_message = ""
            for message in reversed(normalized):
                if message.get("role") == "user":
                    last_user_message = message.get("content", "")
                    break
            yield StreamingChunk(
                content=self._postprocess_assistant_text(
                    self._generate_gguf_text(prompt, **kwargs),
                    last_user_message,
                ),
                is_finished=True,
            )
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
        """Generate embeddings using a fallback embedding model.
        
        Horus is a text generation model and does not natively support embeddings.
        This method attempts to use sentence-transformers if available, or raises
        a helpful error message.
        
        Args:
            texts: String or list of strings to embed
            model: Optional embedding model name (defaults to all-MiniLM-L6-v2)
            **kwargs: Additional arguments
            
        Returns:
            List of embedding vectors
            
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            embed_model = model or "sentence-transformers/all-MiniLM-L6-v2"
            
            # Lazy load the embedding model
            if not hasattr(self, '_embedding_model') or self._embedding_model is None:
                logger.info(f"Loading embedding model: {embed_model}")
                self._embedding_model = SentenceTransformer(embed_model)
            
            # Handle single string input
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self._embedding_model.encode(texts, convert_to_list=True)
            
            # Return single vector if only one input
            if len(embeddings) == 1 and isinstance(texts, list) and len(texts) == 1:
                return embeddings[0]
            
            return embeddings
            
        except ImportError:
            raise ImportError(
                "Horus does not natively support embeddings. "
                "To use embeddings, install sentence-transformers:\n"
                "  pip install sentence-transformers\n\n"
                "Or use a dedicated embedding provider:\n"
                "  nn.create_provider('sentence-transformers')\n"
                "  nn.create_provider('nomic-embeddings')\n"
                "  nn.create_provider('huggingface-embeddings')"
            )
    
    async def aembed(self, texts: Union[str, List[str]], model: Optional[str] = None, **kwargs) -> Union[List[float], List[List[float]]]:
        """Async version of embed - runs in executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, model=model, **kwargs))

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "id": self.model_id,
            **self._model_info,
            "device": self.device,
            "n_gpu_layers": self._resolve_n_gpu_layers() if self._is_gguf_model_id(self.model_id) else None,
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
            "turboquant_backend": (
                "llama.cpp-kv"
                if (self._is_gguf_model_id(self.model_id) and self._gguf_kv_quant_applied)
                else (
                    "llama.cpp"
                    if (self._is_gguf_model_id(self.model_id) and self.turboquant)
                    else ("transformers-kv" if self.turboquant else "disabled")
                )
            ),
            "tools_supported": True,
        }

    def unload(self) -> None:
        """Best-effort model cleanup, especially for llama.cpp backends."""
        model = self.model
        self.model = None
        self.tokenizer = None
        if model is None:
            return
        close = getattr(model, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass

    def _format_tool_descriptions(self, tools: List[Dict]) -> str:
        """Format tool descriptions for the system prompt."""
        descriptions = []
        for tool in tools:
            if isinstance(tool, dict):
                func = tool.get("function", tool)
                name = func.get("name", "unknown")
                desc = func.get("description", "No description")
                params = func.get("parameters", {})
                param_desc = ""
                if params:
                    props = params.get("properties", {})
                    required = params.get("required", [])
                    param_parts = []
                    for k, v in props.items():
                        req_mark = " (required)" if k in required else ""
                        param_parts.append(f"      - {k}: {v.get('type', 'string')}{req_mark}")
                    if param_parts:
                        param_desc = "\n    Parameters:\n" + "\n".join(param_parts)
                descriptions.append(f"  - {name}: {desc}{param_desc}")
            else:
                # Handle BaseTool objects
                name = getattr(tool, "name", "unknown")
                desc = getattr(tool, "description", "No description")
                descriptions.append(f"  - {name}: {desc}")
        return "\n".join(descriptions) if descriptions else "  No tools available"

    def _parse_tool_calls(self, content: str) -> List[ToolCall]:
        """Parse tool calls from model response content."""
        tool_calls = []
        try:
            # Look for JSON object patterns
            import json
            # Try to find JSON in the content
            json_pattern = r'\{[^{}]*"tool"[^}]*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if "tool" in data and "arguments" in data:
                        tool_calls.append(ToolCall(
                            name=data["tool"],
                            arguments=data["arguments"],
                            call_id=f"call_{len(tool_calls)}"
                        ))
                except json.JSONDecodeError:
                    continue
            # Also try standard format {"name": ..., "arguments": ...}
            if not tool_calls:
                try:
                    data = json.loads(content.strip())
                    if isinstance(data, dict):
                        if "name" in data and "arguments" in data:
                            tool_calls.append(ToolCall(
                                name=data["name"],
                                arguments=data["arguments"],
                                call_id=f"call_{len(tool_calls)}"
                            ))
                        elif "tool" in data and "arguments" in data:
                            tool_calls.append(ToolCall(
                                name=data["tool"],
                                arguments=data["arguments"],
                                call_id=f"call_{len(tool_calls)}"
                            ))
                except json.JSONDecodeError:
                    pass
        except Exception as exc:
            logger.debug(f"Tool call parsing failed: {exc}")
        return tool_calls

    def _clean_tool_content(self, content: str) -> str:
        """Clean tool JSON from content to get natural language response."""
        # Remove JSON blocks
        cleaned = re.sub(r'\{[^{}]*"tool"[^}]*\}', '', content, flags=re.DOTALL)
        cleaned = re.sub(r'\{[^{}]*"name"[^}]*\}', '', cleaned, flags=re.DOTALL)
        return cleaned.strip()

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
    def register_gguf_variant(
        cls,
        model_id: str,
        name: Optional[str] = None,
        size: str = "4B",
        file_size: Optional[str] = None,
    ) -> None:
        """Register a ready GGUF Horus variant (including TQ*) in the model list."""
        if not model_id.lower().endswith(".gguf"):
            raise ValueError("register_gguf_variant expects a GGUF model id ending with .gguf")
        _, filename = cls._split_repo_and_filename(model_id)
        quant_match = re.search(r"-(Q\d(?:_[A-Z0-9]+)?|TQ\d(?:_[A-Z0-9]+)?|F16|BF16)\.gguf$", filename, flags=re.IGNORECASE)
        quant = quant_match.group(1).upper() if quant_match else "GGUF"
        cls.HORUS_MODELS[model_id] = {
            "name": name or f"Horus {size} ({quant})",
            "official_name": filename,
            "size": size,
            "type": "gguf",
            "quantization": quant,
            **({"file_size": file_size} if file_size else {}),
        }

    @classmethod
    def benchmark_variants(
        cls,
        model_ids: List[str],
        prompt: str = "Summarize what TurboQuant does in one sentence.",
        runs: int = 2,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
        max_new_tokens: int = 64,
    ) -> List[Dict[str, Any]]:
        """Quick benchmark helper for ready GGUF Horus variants (e.g., TQ vs Q4)."""
        results: List[Dict[str, Any]] = []
        for model_id in model_ids:
            per_run: List[float] = []
            first_preview = ""
            model = cls(
                model_id=model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                max_new_tokens=max_new_tokens,
                suppress_warnings=True,
                suppress_native_output=True,
            ).load()
            try:
                for _ in range(max(runs, 1)):
                    start = time.perf_counter()
                    response = model.chat([{"role": "user", "content": prompt}])
                    elapsed = time.perf_counter() - start
                    per_run.append(elapsed)
                    if not first_preview:
                        first_preview = response.content[:160]
                info = model.get_model_info()
                results.append(
                    {
                        "model_id": model_id,
                        "quantization": info.get("quantization", {}).get("gguf") if isinstance(info.get("quantization"), dict) else info.get("quantization"),
                        "avg_latency_sec": round(sum(per_run) / len(per_run), 4),
                        "min_latency_sec": round(min(per_run), 4),
                        "max_latency_sec": round(max(per_run), 4),
                        "context_window": info.get("context_window"),
                        "preview": first_preview,
                    }
                )
            finally:
                model.unload()
        return results

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


class HorusModel(HorusProvider):
    """Backward-compatible Horus model alias."""

    class model:
        @staticmethod
        def List() -> List[Dict[str, str]]:
            """Return available Horus models with exact official Hugging Face names."""
            items: List[Dict[str, str]] = []
            for model_id, info in HorusProvider.list_available_models().items():
                items.append(
                    {
                        "model_id": model_id,
                        "official_name": str(info.get("official_name", HorusProvider._infer_official_name(model_id))),
                        "type": str(info.get("type", "unknown")),
                    }
                )
            return items

        @staticmethod
        def lines() -> List[str]:
            """Return human-friendly model lines (one model per line)."""
            entries = HorusModel.model.List()
            return [f"{entry['official_name']}  ->  {entry['model_id']}" for entry in entries]

        @staticmethod
        def print() -> None:
            """Print available models in a simple readable format."""
            print("Available Horus Models:")
            for line in HorusModel.model.lines():
                print(line)


Horus = HorusModel
HorusLensModel = HorusModel


def load_horus(model_id: str = "tokenaii/horus", **kwargs) -> HorusProvider:
    return HorusProvider(model_id=model_id, **kwargs).load()


def load_horus_lens(model_id: str = "tokenaii/Horus-Lens-1.0", **kwargs) -> HorusProvider:
    return HorusProvider(model_id=model_id, **kwargs).load()


__all__ = [
    "HorusProvider",
    "HorusModel",
    "HorusLensModel",
    "Horus",
    "load_horus",
    "load_horus_lens",
    "print_model_list",
]
