"""Local module for hardware detection and local model management."""

from .hardware import HardwareDetector
from .llm_hub import LocalLLMHub, ModelInfo
from typing import Any, List, Dict, Optional, Union, Iterator
from ..core.types import BaseProvider, LLMResponse, Message, StreamingChunk, TokenUsage, ToolDefinition
import threading
import time


class LocalLLM(BaseProvider):
    """Local LLM provider for running models locally with lazy loading."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = -1,
        lazy_load: bool = True,
        **kwargs: Any
    ):
        """
        Initialize local LLM.
        
        Args:
            model_path: Path to local model file (e.g., .gguf, .safetensors)
            model_name: Name of model to load from hub
            n_ctx: Context window size
            n_threads: Number of threads to use (-1 for auto)
            lazy_load: If True, don't load model until first use (recommended)
            **kwargs: Additional parameters
        """
        super().__init__(model=model_name or model_path, **kwargs)
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads if n_threads > 0 else None
        self._model = None
        self._tokenizer = None
        self._loading = False
        self._loaded = False
        
        # Load immediately only if lazy_load=False and model_path provided
        if model_path and not lazy_load:
            self.load_model()
    
    def load_model(self, timeout: Optional[int] = None) -> None:
        """
        Load the model explicitly.
        
        Args:
            timeout: Maximum seconds to wait for loading (None = no timeout)
        """
        if self._loaded or self._loading:
            return
            
        if not self.model_path:
            raise ValueError("No model_path specified")
        
        self._loading = True
        
        # Load in thread with progress
        load_thread = threading.Thread(target=self._load_model_thread)
        load_thread.daemon = True
        load_thread.start()
        
        # Wait with progress
        start_time = time.time()
        dots = 0
        while load_thread.is_alive():
            if timeout and (time.time() - start_time) > timeout:
                print(f"\n⚠️  Model loading timeout after {timeout}s")
                self._loading = False
                raise TimeoutError(f"Model loading exceeded {timeout} seconds")
            
            dots = (dots + 1) % 4
            print(f"\r⏳ Loading model{'.' * dots}{' ' * (3-dots)}", end='', flush=True)
            time.sleep(0.5)
        
        print(f"\r✅ Model loaded successfully!{' ' * 10}")
        self._loading = False
        self._loaded = True
    
    def _load_model_thread(self) -> None:
        """Load model in background thread."""
        try:
            self._load_model_impl(self.model_path)
        except Exception as e:
            print(f"\n❌ Error loading model: {e}")
            self._loading = False
            raise
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before use."""
        if not self._loaded and self.model_path:
            print("ℹ️  Auto-loading model (use lazy_load=False to load on init)")
            self.load_model()
    
    def _load_model_impl(self, model_path: str) -> None:
        """Load model from path (implementation)."""
        import os
        from pathlib import Path
        
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Detect model format
        suffix = path.suffix.lower()
        
        if suffix == ".gguf":
            self._load_gguf_model(str(path))
        elif suffix in [".safetensors", ".bin", ".pt", ".pth"]:
            self._load_hf_model(str(path))
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
    
    def _load_gguf_model(self, model_path: str) -> None:
        """Load GGUF format model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
            
            self._model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
    
    def _load_hf_model(self, model_path: str) -> None:
        """Load HuggingFace format model with detailed progress."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            import torch
            
            print(f"\n📦 Loading HuggingFace model from: {model_path}")
            print("🔄 Step 1/3: Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print("✅ Tokenizer loaded")
            
            print("🔄 Step 2/3: Loading model config...")
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            print(f"✅ Config loaded: {config.model_type if hasattr(config, 'model_type') else 'unknown'}")
            
            print("🔄 Step 3/3: Loading model weights (this may take several minutes)...")
            print("   Using: float16, auto device mapping, low CPU memory")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            device = next(self._model.parameters()).device
            print(f"✅ Model loaded on device: {device}")
            
        except ImportError:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            print(f"❌ Error loading HF model: {e}")
            raise
    
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send chat completion request."""
        self._ensure_loaded()
        
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first or provide model_path on init.")
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Generate
        if hasattr(self._model, 'create_chat_completion'):
            # llama-cpp style
            response = self._model.create_chat_completion(
                messages=[{"role": m.role.value, "content": m.content} for m in messages],
                stream=stream,
                **kwargs
            )
            
            if stream:
                return self._stream_response(response)
            else:
                content = response["choices"][0]["message"]["content"]
                return LLMResponse(
                    text=content,
                    model=self.model or "local",
                    provider="local",
                    usage=TokenUsage()
                )
        else:
            # HF transformers style
            inputs = self._tokenizer(prompt, return_tensors="pt")
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True
            )
            content = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return LLMResponse(
                text=content,
                model=self.model or "local",
                provider="local",
                usage=TokenUsage()
            )
    
    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to prompt string."""
        prompt_parts = []
        for msg in messages:
            if msg.role.value == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role.value == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role.value == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _stream_response(self, response) -> Iterator[StreamingChunk]:
        """Stream response chunks."""
        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield StreamingChunk(
                    text=delta["content"],
                    is_finished=False
                )
    
    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Async version of chat."""
        return self.chat(messages, tools, stream, **kwargs)
    
    def complete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send text completion request."""
        return self.chat(
            [Message(role="user", content=prompt)],
            stream=stream,
            **kwargs
        )
    
    async def acomplete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Async version of complete."""
        return self.complete(prompt, stream, **kwargs)
    
    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Generate embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Simple fallback embedding (not ideal but functional)
        # In production, use proper embedding models
        import hashlib
        embeddings = []
        for text in texts:
            # Create deterministic pseudo-embedding based on text hash
            hash_val = hashlib.md5(text.encode()).hexdigest()
            embedding = [float(int(hash_val[i:i+2], 16)) / 255.0 for i in range(0, 64, 2)]
            embeddings.append(embedding)
        
        return embeddings
    
    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any
    ) -> List[List[float]]:
        """Async version of embed."""
        return self.embed(texts, **kwargs)
    
    @property
    def supports_tools(self) -> bool:
        """Whether this provider supports tool calling."""
        return False  # Local models may not support function calling
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming."""
        return hasattr(self._model, 'create_chat_completion') if self._model else False
    
    @property
    def supports_structured_output(self) -> bool:
        """Whether this provider supports structured output."""
        return False


    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._loaded


__all__ = [
    "HardwareDetector",
    "LocalLLMHub",
    "ModelInfo",
    "LocalLLM",
]
