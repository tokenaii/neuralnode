"""OpenAI provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from openai import AsyncOpenAI, OpenAI

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


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.provider_type = ProviderType.OPENAI

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal messages to OpenAI format."""
        return [
            {
                "role": m.role.value,
                "content": m.content,
                **({"name": m.name} if m.name else {}),
                **({"tool_calls": m.tool_calls} if m.tool_calls else {}),
                **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
            }
            for m in messages
        ]

    def _convert_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to OpenAI format."""
        if not tools:
            return None
        return [tool.to_openai_format() for tool in tools]

    def _make_usage(self, usage: Optional[Any]) -> TokenUsage:
        """Convert OpenAI usage to internal format."""
        if not usage:
            return TokenUsage()
        return TokenUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
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
        
        openai_tools = self._convert_tools(tools)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            tools=openai_tools,
            stream=stream,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        
        if stream:
            return self._handle_streaming(response)
        
        result = LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            provider=self.provider_type,
            usage=self._make_usage(response.usage),
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump(),
        )
        self._emit("chat_complete", {"response": result})
        return result

    def _handle_streaming(self, response) -> Iterator[StreamingChunk]:
        """Handle streaming response."""
        for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            
            is_finished = chunk.choices[0].finish_reason is not None
            
            yield StreamingChunk(
                text=content,
                is_finished=is_finished,
                usage=self._make_usage(chunk.usage) if chunk.usage else None,
            )
            
            if is_finished:
                break

    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async chat completion."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        
        openai_tools = self._convert_tools(tools)
        
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            tools=openai_tools,
            stream=stream,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
        
        if stream:
            return self._handle_async_streaming(response)
        
        result = LLMResponse(
            text=response.choices[0].message.content or "",
            model=self.model,
            provider=self.provider_type,
            usage=self._make_usage(response.usage),
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump(),
        )
        self._emit("chat_complete", {"response": result})
        return result

    async def _handle_async_streaming(self, response) -> AsyncIterator[StreamingChunk]:
        """Handle async streaming response."""
        async for chunk in response:
            delta = chunk.choices[0].delta
            content = delta.content or ""
            
            is_finished = chunk.choices[0].finish_reason is not None
            
            yield StreamingChunk(
                text=content,
                is_finished=is_finished,
                usage=self._make_usage(chunk.usage) if chunk.usage else None,
            )
            
            if is_finished:
                break

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
        """Generate embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        model = kwargs.get("model", "text-embedding-3-small")
        
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]

    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        model = kwargs.get("model", "text-embedding-3-small")
        
        response = await self.async_client.embeddings.create(
            model=model,
            input=texts,
        )
        
        return [item.embedding for item in response.data]

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True
