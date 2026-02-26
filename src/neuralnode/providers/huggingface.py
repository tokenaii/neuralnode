"""HuggingFace provider implementation."""

from typing import Any, AsyncIterator, Iterator, List, Optional, Union

from ..core.types import (
    BaseProvider,
    LLMResponse,
    Message,
    MessageRole,
    ProviderType,
    StreamingChunk,
    TokenUsage,
    ToolDefinition,
)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Transformers local provider."""

    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.provider_type = ProviderType.HUGGINGFACE
        self._pipeline = None
        self._tokenizer = None
        self.device = device or ("cuda" if self._has_cuda() else "cpu")
        self._model_loaded = False

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_model(self):
        """Lazy load the model."""
        if self._model_loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModelForCausalLM.from_pretrained(
                self.model,
                torch_dtype="auto",
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if self.device == "cpu":
                model = model.to("cpu")
            
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self._tokenizer,
            )
            self._model_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")

    def _convert_messages_to_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a single prompt string."""
        parts = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                parts.append(f"System: {msg.content}")
            elif msg.role == MessageRole.USER:
                parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                parts.append(f"Assistant: {msg.content}")
        
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send chat completion request."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        self._load_model()
        
        prompt = self._convert_messages_to_prompt(messages)
        
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "do_sample": True,
            "return_full_text": False,
        }
        
        if stream:
            return self._handle_streaming(prompt, generation_kwargs)
        
        outputs = self._pipeline(prompt, **generation_kwargs)
        generated_text = outputs[0]["generated_text"]
        
        # Estimate tokens
        prompt_tokens = len(self._tokenizer.encode(prompt))
        completion_tokens = len(self._tokenizer.encode(generated_text))
        
        result = LLMResponse(
            text=generated_text.strip(),
            model=self.model,
            provider=self.provider_type,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="stop",
        )
        self._emit("chat_complete", {"response": result})
        return result

    def _handle_streaming(
        self, prompt: str, generation_kwargs: dict
    ) -> Iterator[StreamingChunk]:
        """Handle streaming response."""
        # HuggingFace pipelines don't support true streaming
        # Generate and yield chunks
        outputs = self._pipeline(prompt, **generation_kwargs)
        generated_text = outputs[0]["generated_text"]
        
        # Simulate streaming by yielding word by word
        words = generated_text.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            yield StreamingChunk(
                text=word + (" " if not is_last else ""),
                is_finished=is_last,
            )

    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async chat completion."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.chat(messages, tools, stream, **kwargs)
        )

    def complete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Text completion."""
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return self.chat(messages, stream=stream, **kwargs)

    async def acomplete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async text completion."""
        messages = [Message(role=MessageRole.USER, content=prompt)]
        return await self.achat(messages, stream=stream, **kwargs)

    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
            
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = model.encode(texts)
            return embeddings.tolist()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )

    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async embeddings."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def supports_streaming(self) -> bool:
        return True  # Simulated streaming

    @property
    def supports_structured_output(self) -> bool:
        return False
