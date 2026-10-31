"""Universal Local Model Provider - Supports ALL model formats and architectures.

Supports:
- SafeTensors (HuggingFace)
- GGUF/GGML (Llama.cpp)
- ONNX (ONNX Runtime)
- PyTorch (.bin, .pt, .pth)
- AWQ/GPTQ (Quantized models)
"""

import os
import re
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Union, Callable
from dataclasses import dataclass
from enum import Enum

from ..core import BaseProvider, Message, MessageRole, LLMResponse, StreamingChunk, BaseTool


class ModelFormat(Enum):
    """Supported model formats."""

    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    GGML = "ggml"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    AWQ = "awq"
    GPTQ = "gptq"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    name: str
    format: ModelFormat
    architecture: str
    parameters: int
    quantization: Optional[str]
    device: str
    memory_used_mb: float


def _generation_options(temperature, max_tokens, pad_token_id, eos_token_id):
    options = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }
    if temperature > 0:
        options["temperature"] = temperature
    return options


class UniversalLocalProvider(BaseProvider):
    """Universal provider that auto-detects and loads ANY local model format.

    Automatically detects the model format and uses the appropriate backend:
    - SafeTensors → Transformers
    - GGUF/GGML → Llama.cpp (via llama-cpp-python)
    - ONNX → ONNX Runtime
    - PyTorch → Transformers
    - AWQ/GPTQ → AutoAWQ / Transformers
    """

    @property
    def provider_name(self) -> str:
        """Return the provider identifier used by NeuralNode."""
        return "universal-local"

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        context_length: int = 4096,
        quantization: Optional[str] = None,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU if available
        n_threads: Optional[int] = None,
        **kwargs,
    ):
        """Initialize universal local model provider.

        Args:
            model_path: Path to model file or directory
            device: "auto", "cpu", "cuda", "cuda:0", "mps"
            context_length: Maximum context length
            quantization: "q4_0", "q5_k_m", "q8_0", etc. for GGUF
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            n_threads: Number of CPU threads (auto if None)
        """
        super().__init__(model="universal-local", **kwargs)
        self.model_path = Path(model_path)
        self.device = device
        self.context_length = context_length
        self.quantization = quantization
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count() or 4

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_vision = False
        self._format: ModelFormat = ModelFormat.UNKNOWN
        self._info: Optional[ModelInfo] = None
        self._is_loaded = False

        # Detect format before loading
        self._format = self._detect_format()

    def _detect_format(self) -> ModelFormat:
        """Auto-detect model format from file extension and contents."""
        if self.model_path.is_dir():
            # Check directory contents
            files = list(self.model_path.iterdir())

            for f in files:
                if f.suffix == ".gguf":
                    return ModelFormat.GGUF
                elif f.suffix == ".ggml":
                    return ModelFormat.GGML
                elif f.suffix == ".onnx":
                    return ModelFormat.ONNX
                elif f.suffix == ".safetensors":
                    return ModelFormat.SAFETENSORS
                elif f.suffix in [".bin", ".pt", ".pth"]:
                    return ModelFormat.PYTORCH
                elif "awq" in f.name.lower():
                    return ModelFormat.AWQ
                elif "gptq" in f.name.lower():
                    return ModelFormat.GPTQ

            # Check config.json for clues
            config_file = self.model_path / "config.json"
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    if "quantization_config" in config:
                        q_config = config["quantization_config"]
                        if "awq" in str(q_config).lower():
                            return ModelFormat.AWQ
                        if "gptq" in str(q_config).lower():
                            return ModelFormat.GPTQ
                except:
                    pass

            return ModelFormat.SAFETENSORS  # Default for directories

        else:
            # Single file
            suffix = self.model_path.suffix.lower()
            name = self.model_path.name.lower()

            if suffix == ".gguf":
                return ModelFormat.GGUF
            elif suffix == ".ggml":
                return ModelFormat.GGML
            elif suffix == ".onnx":
                return ModelFormat.ONNX
            elif suffix == ".safetensors":
                return ModelFormat.SAFETENSORS
            elif suffix in [".bin", ".pt", ".pth"]:
                return ModelFormat.PYTORCH
            elif "awq" in name:
                return ModelFormat.AWQ
            elif "gptq" in name:
                return ModelFormat.GPTQ

            return ModelFormat.UNKNOWN

    def _load_model(self):
        """Load model using appropriate backend."""
        if self._is_loaded:
            return

        print(f"Loading {self._format.value} model from {self.model_path}...")

        if self._format == ModelFormat.GGUF or self._format == ModelFormat.GGML:
            self._load_gguf()
        elif self._format == ModelFormat.SAFETENSORS:
            self._load_safetensors()
        elif self._format == ModelFormat.PYTORCH:
            self._load_pytorch()
        elif self._format == ModelFormat.ONNX:
            self._load_onnx()
        elif self._format == ModelFormat.AWQ:
            self._load_awq()
        elif self._format == ModelFormat.GPTQ:
            self._load_gptq()
        else:
            raise ValueError(f"Unsupported model format: {self._format}")

        self._is_loaded = True
        print(f"✓ Model loaded successfully ({self._format.value})")

    def _load_gguf(self):
        """Load GGUF/GGML model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python required. Install: pip install llama-cpp-python")

        # Find model file
        if self.model_path.is_dir():
            model_files = list(self.model_path.glob("*.gguf")) + list(
                self.model_path.glob("*.ggml")
            )
            if not model_files:
                raise FileNotFoundError(f"No .gguf/.ggml files found in {self.model_path}")
            model_file = str(model_files[0])
        else:
            model_file = str(self.model_path)

        # Auto-detect GPU layers if -1
        n_gpu_layers = self.n_gpu_layers
        if n_gpu_layers == -1:
            try:
                import subprocess

                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode == 0:
                    n_gpu_layers = 1000  # Offload all to GPU
                else:
                    n_gpu_layers = 0
            except:
                n_gpu_layers = 0

        self._model = Llama(
            model_path=model_file,
            n_ctx=self.context_length,
            n_threads=self.n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        self._info = ModelInfo(
            name=Path(model_file).stem,
            format=self._format,
            architecture="llama",
            parameters=0,  # Will be updated if available in metadata
            quantization=self.quantization,
            device="cuda" if n_gpu_layers > 0 else "cpu",
            memory_used_mb=0,
        )

    def _load_safetensors(self):
        """Load SafeTensors model using Transformers."""
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required. Install: pip install transformers torch"
            )

        # Determine device
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        load_kwargs = {
            "torch_dtype": torch.float16 if device.startswith("cuda") else torch.float32,
            "trust_remote_code": True,
            "local_files_only": True,
        }

        config = AutoConfig.from_pretrained(
            str(self.model_path), trust_remote_code=True, local_files_only=True
        )
        architectures = getattr(config, "architectures", None) or []
        self._is_vision = getattr(config, "vision_config", None) is not None or any(
            marker in architecture.lower()
            for architecture in architectures
            for marker in ("vision", "llava", "idefics", "paligemma", "florence")
        )

        if self._is_vision:
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as VisionModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as VisionModel

            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path), trust_remote_code=True, local_files_only=True
            )
            self._tokenizer = getattr(self._processor, "tokenizer", None)
            self._model = VisionModel.from_pretrained(str(self.model_path), **load_kwargs)
            self._model = self._model.to(device)
        else:
            if device.startswith("cuda"):
                load_kwargs["device_map"] = "auto"
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), trust_remote_code=True, local_files_only=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)

        if device == "cpu":
            self._model = self._model.to("cpu")

        # Estimate parameters
        params = sum(p.numel() for p in self._model.parameters())

        self._info = ModelInfo(
            name=self.model_path.name,
            format=ModelFormat.SAFETENSORS,
            architecture=getattr(self._model.config, "model_type", "unknown"),
            parameters=params,
            quantization=None,
            device=device,
            memory_used_mb=params * 2 / (1024**2) if device == "cuda" else params * 4 / (1024**2),
        )

    def _load_pytorch(self):
        """Load PyTorch checkpoint."""
        # Similar to safetensors but with checkpoint loading
        self._load_safetensors()  # Transformers handles both

    def _load_onnx(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime required. Install: pip install onnxruntime")

        # Determine execution providers
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        model_file = (
            str(self.model_path)
            if self.model_path.is_file()
            else str(self.model_path / "model.onnx")
        )

        self._model = ort.InferenceSession(model_file, providers=providers)
        self._info = ModelInfo(
            name=self.model_path.name,
            format=ModelFormat.ONNX,
            architecture="onnx",
            parameters=0,
            quantization=None,
            device="cuda" if "CUDA" in str(self._model.get_providers()) else "cpu",
            memory_used_mb=0,
        )

    def _load_awq(self):
        """Load AWQ quantized model."""
        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("autoawq required. Install: pip install autoawq")

        self._model = AutoAWQForCausalLM.from_quantized(
            str(self.model_path), fuse_layers=True, trust_remote_code=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        self._info = ModelInfo(
            name=self.model_path.name,
            format=ModelFormat.AWQ,
            architecture="awq",
            parameters=0,
            quantization="AWQ",
            device="cuda",
            memory_used_mb=0,
        )

    def _load_gptq(self):
        """Load GPTQ quantized model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
            import torch
        except ImportError:
            raise ImportError("transformers with GPTQ support required")

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), device_map="auto", torch_dtype=torch.float16
        )

        self._info = ModelInfo(
            name=self.model_path.name,
            format=ModelFormat.GPTQ,
            architecture="gptq",
            parameters=0,
            quantization="GPTQ",
            device="cuda",
            memory_used_mb=0,
        )

    def _build_prompt(self, messages: List[Message]) -> str:
        """Build prompt from messages."""
        if self._format in [ModelFormat.GGUF, ModelFormat.GGML] and self._model:
            # Use llama.cpp chat format
            return self._model.create_chat_completion(
                messages=[
                    {"role": m.role.value, "content": m.content}
                    for m in messages
                    if m.role != MessageRole.TOOL
                ],
                stream=False,
                stop=["</s>"],
            )["choices"][0]["message"]["content"]

        # Default format
        prompt_parts = []
        system_msg = ""

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_msg = msg.content
            elif msg.role == MessageRole.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = ""
        if system_msg:
            prompt += f"System: {system_msg}\n\n"
        prompt += "\n".join(prompt_parts)
        prompt += "\nAssistant:"

        return prompt

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        stream: bool = False,
        **options,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Generate response."""
        self._load_model()

        if self._format in [ModelFormat.GGUF, ModelFormat.GGML]:
            return self._chat_gguf(messages, stream, **options)
        elif self._format == ModelFormat.ONNX:
            return self._chat_onnx(messages, **options)
        else:
            return self._chat_transformers(messages, stream, **options)

    def _chat_gguf(self, messages: List[Message], stream: bool, **options) -> LLMResponse:
        """Chat using GGUF model."""
        prompt = self._build_prompt(messages)

        output = self._model(
            prompt,
            max_tokens=options.get("max_tokens", 512),
            temperature=options.get("temperature", 0.7),
            stop=["</s>", "User:", "System:"],
            echo=False,
        )

        return LLMResponse(
            content=output["choices"][0]["text"].strip(),
            model="gguf",
            provider="local-gguf",
            tool_calls=[],
            usage={
                "prompt_tokens": output["usage"]["prompt_tokens"],
                "completion_tokens": output["usage"]["completion_tokens"],
            },
        )

    def _chat_transformers(self, messages: List[Message], stream: bool, **options) -> LLMResponse:
        """Chat using Transformers model."""
        import torch

        if self._processor is not None:
            return self._chat_vision_transformers(messages, **options)

        prompt = self._build_prompt(messages)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length,
        )

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        temperature = options.get("temperature", 0.7)
        # Transformers rejects temperature=0 even when sampling is disabled.
        generation_options = _generation_options(
            temperature,
            options.get("max_tokens", 512),
            self._tokenizer.pad_token_id,
            self._tokenizer.eos_token_id,
        )

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_options)

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return LLMResponse(
            content=response_text.strip(),
            model=self._info.name if self._info else "transformers",
            provider="local-transformers",
            tool_calls=[],
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(generated),
            },
        )

    @staticmethod
    def _open_message_images(message: Message):
        """Decode local image metadata into RGB PIL images."""
        metadata = message.metadata or {}
        if metadata.get("image_url"):
            raise ValueError(
                "Local Hugging Face vision models require a local image path; "
                "download URL images first to avoid fetching untrusted network targets."
            )

        encoded_images = metadata.get("images") or []
        image_path = metadata.get("image_path")
        if image_path and not encoded_images:
            encoded_images = [image_path]

        if not encoded_images:
            return []

        from PIL import Image

        images = []
        for value in encoded_images:
            image_path = None
            if isinstance(value, str) and len(value) < 4096:
                try:
                    candidate = Path(value)
                    if candidate.is_file():
                        image_path = candidate
                except OSError:
                    pass
            if image_path is not None:
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))
                continue

            if isinstance(value, str) and value.startswith("data:") and "," in value:
                value = value.split(",", 1)[1]
            raw = base64.b64decode(value, validate=True)
            with Image.open(BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
        return images

    def _chat_vision_transformers(self, messages: List[Message], **options) -> LLMResponse:
        """Run a Transformers vision-language model on text and local images."""
        import torch

        model_messages = []
        images = []
        image_counts = {}
        for message in messages:
            if message.role == MessageRole.TOOL:
                continue
            content = []
            message_images = self._open_message_images(message)
            image_counts[id(message)] = len(message_images)
            for image in message_images:
                content.append({"type": "image"})
                images.append(image)
            if message.content:
                content.append({"type": "text", "text": message.content})
            model_messages.append({"role": message.role.value, "content": content})

        processor = self._processor
        if hasattr(processor, "apply_chat_template"):
            try:
                prompt = processor.apply_chat_template(
                    model_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except (AttributeError, ValueError):
                prompt = None
        else:
            prompt = None

        image_token = getattr(processor, "image_token", None)
        image_only = bool(images and not prompt and not isinstance(image_token, str))
        if not image_only and (
            not prompt or (images and isinstance(image_token, str) and image_token not in prompt)
        ):
            image_marker = image_token if isinstance(image_token, str) else "<image>"
            prompt_parts = []
            for message in messages:
                if message.role == MessageRole.SYSTEM:
                    prompt_parts.append(f"System: {message.content}")
                elif message.role == MessageRole.USER:
                    prompt_parts.append(
                        f"User: {(image_marker + ' ') * image_counts.get(id(message), 0)}{message.content}"
                    )
                elif message.role == MessageRole.ASSISTANT:
                    prompt_parts.append(f"Assistant: {message.content}")
            prompt = "\n".join(prompt_parts) + "\nAssistant:"

        if image_only:
            # Captioning processors such as BLIP expect image-only inputs;
            # arbitrary chat-style role text becomes decoder tokens and can
            # consume the entire generation budget without producing a caption.
            inputs = processor(images=images, return_tensors="pt")
        else:
            inputs = processor(
                text=prompt,
                images=images or None,
                return_tensors="pt",
            )
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        temperature = options.get("temperature", 0.7)
        generation_options = {
            "max_new_tokens": options.get("max_tokens", 512),
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_options)
        input_ids = inputs.get("input_ids")
        if getattr(self._model.config, "is_encoder_decoder", False) or input_ids is None:
            generated = outputs[0]
        else:
            generated = outputs[0][input_ids.shape[1] :]
        response_text = processor.decode(generated, skip_special_tokens=True)
        return LLMResponse(
            content=response_text.strip(),
            model=self._info.name if self._info else "vision-transformers",
            provider="local-transformers",
            tool_calls=[],
            usage={
                "prompt_tokens": int(input_ids.shape[1]) if input_ids is not None else 0,
                "completion_tokens": len(generated),
            },
        )

    def _chat_onnx(self, messages: List[Message], **options) -> LLMResponse:
        """Chat using ONNX model."""
        # ONNX implementation would go here
        # This is a placeholder - full implementation requires ONNX-specific code
        return LLMResponse(
            content="ONNX chat not fully implemented yet.",
            model="onnx",
            provider="local-onnx",
            tool_calls=[],
            usage={},
        )

    async def achat(self, messages, tools=None, stream=False, **options):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.chat(messages, tools, stream, **options)
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings if supported."""
        if self._format in [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH]:
            import torch

            self._load_model()

            embeddings = []
            for text in texts:
                inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                device = next(self._model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model(**inputs, output_hidden_states=True)
                    hidden = outputs.hidden_states[-1]
                    embedding = hidden.mean(dim=1).squeeze().cpu().numpy().tolist()
                    embeddings.append(embedding)

            return embeddings

        raise NotImplementedError(f"Embeddings not supported for {self._format.value}")

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts))

    @property
    def capabilities(self) -> Dict[str, bool]:
        return {
            "tools": False,
            "streaming": False,
            "embeddings": self._format in [ModelFormat.SAFETENSORS, ModelFormat.PYTORCH],
            "vision": self._is_vision,
        }

    def get_info(self) -> ModelInfo:
        """Get model information."""
        if not self._is_loaded:
            self._load_model()
        return self._info

    def unload(self):
        """Unload model from memory."""
        import gc

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_vision = False
        self._is_loaded = False

        try:
            import torch

            torch.cuda.empty_cache()
        except:
            pass

        gc.collect()
        print("Model unloaded")


# Convenience function
def load_local_model(model_path: str, device: str = "auto", **kwargs) -> UniversalLocalProvider:
    """Quick function to load any local model."""
    return UniversalLocalProvider(model_path, device, **kwargs)
