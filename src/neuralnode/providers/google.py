"""Google Gemini provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

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


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    def __init__(
        self,
        model: str = "gemini-pro",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        # Lazy import
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError("google-generativeai not installed. Install with: pip install google-generativeai")
        self.provider_type = ProviderType.GOOGLE

    def _convert_messages(self, messages: List[Message]) -> tuple[str, str]:
        """
        Convert messages to Gemini format.
        Returns (system_prompt, user_prompt)
        """
        system_parts = []
        user_parts = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role == MessageRole.USER:
                user_parts.append(msg.content)
            elif msg.role == MessageRole.ASSISTANT:
                # Gemini uses a different format - we'll include this in context
                user_parts.append(f"Assistant: {msg.content}")
        
        system_prompt = "\n".join(system_parts)
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt

    def _estimate_usage(self, messages: List[Message], response_text: str) -> TokenUsage:
        """Estimate token usage."""
        prompt_chars = sum(len(m.content) for m in messages)
        completion_chars = len(response_text)
        
        # Gemini uses ~4 chars per token
        return TokenUsage(
            prompt_tokens=prompt_chars // 4,
            completion_tokens=completion_chars // 4,
            total_tokens=(prompt_chars + completion_chars) // 4,
        )

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send chat completion request."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        
        system_prompt, user_prompt = self._convert_messages(messages)
        
        # Combine system and user prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt
        
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_output_tokens": kwargs.get("max_tokens", 2048),
            "top_p": kwargs.get("top_p", 0.95),
        }
        
        if stream:
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config,
                stream=True,
            )
            return self._handle_streaming(response)
        
        response = self.client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        
        text = response.text if hasattr(response, "text") else ""
        usage = self._estimate_usage(messages, text)
        
        result = LLMResponse(
            text=text,
            model=self.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason="stop" if response.candidates else "error",
            raw_response={"candidates": len(response.candidates) if response.candidates else 0},
        )
        self._emit("chat_complete", {"response": result})
        return result

    def _handle_streaming(self, response) -> Iterator[StreamingChunk]:
        """Handle streaming response."""
        for chunk in response:
            text = chunk.text if hasattr(chunk, "text") else ""
            yield StreamingChunk(text=text, is_finished=False)
        
        yield StreamingChunk(text="", is_finished=True)

    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async chat completion."""
        # Gemini doesn't have native async support, run in thread
        import asyncio
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.chat(messages, tools, stream, **kwargs)
        )
        return result

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
        """Generate embeddings using Gemini embedding model."""
        if isinstance(texts, str):
            texts = [texts]
        
        model = kwargs.get("model", "models/embedding-001")
        
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document",
            )
            embeddings.append(result["embedding"])
        
        return embeddings

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
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True
