"""Anthropic provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from anthropic import Anthropic, AsyncAnthropic

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


class AnthropicProvider(BaseProvider):
    """Anthropic Claude API provider."""

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        self.provider_type = ProviderType.ANTHROPIC

    def _convert_messages(self, messages: List[Message]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Convert messages to Anthropic format.
        Returns (system_prompt, anthropic_messages)
        """
        system_prompt = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt += msg.content + "\n"
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.content}],
                })
            elif msg.role == MessageRole.ASSISTANT:
                anthropic_messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg.content}],
                })
        
        return system_prompt.strip(), anthropic_messages

    def _convert_tools(
        self, tools: Optional[List[ToolDefinition]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert tools to Anthropic format."""
        if not tools:
            return None
        
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required,
                },
            })
        return anthropic_tools

    def _estimate_usage(self, messages: List[Message], response_text: str) -> TokenUsage:
        """Estimate token usage (Anthropic doesn't always return this)."""
        # Rough estimate: 1 token ≈ 4 characters
        prompt_chars = sum(len(m.content) for m in messages)
        completion_chars = len(response_text)
        
        prompt_tokens = prompt_chars // 4
        completion_tokens = completion_chars // 4
        
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
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
        
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages,
            "system": system_prompt if system_prompt else None,
            **{k: v for k, v in kwargs.items() if v is not None and k != "max_tokens"},
        }
        
        if anthropic_tools:
            request_params["tools"] = anthropic_tools
        
        response = self.client.messages.create(**request_params)
        
        if stream:
            return self._handle_streaming(response)
        
        # Extract text content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        # Get usage if available
        usage = TokenUsage()
        if hasattr(response, "usage"):
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        else:
            usage = self._estimate_usage(messages, text_content)
        
        result = LLMResponse(
            text=text_content,
            model=self.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason=response.stop_reason,
            raw_response={"id": response.id, "stop_reason": response.stop_reason},
        )
        self._emit("chat_complete", {"response": result})
        return result

    def _handle_streaming(self, response) -> Iterator[StreamingChunk]:
        """Handle streaming response."""
        # Anthropic streaming not fully implemented in this version
        # Fallback to non-streaming
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        yield StreamingChunk(text=text_content, is_finished=True)

    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async chat completion."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        
        system_prompt, anthropic_messages = self._convert_messages(messages)
        anthropic_tools = self._convert_tools(tools)
        
        request_params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": anthropic_messages,
            "system": system_prompt if system_prompt else None,
            **{k: v for k, v in kwargs.items() if v is not None and k != "max_tokens"},
        }
        
        if anthropic_tools:
            request_params["tools"] = anthropic_tools
        
        response = await self.async_client.messages.create(**request_params)
        
        if stream:
            return self._handle_async_streaming(response)
        
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        usage = TokenUsage()
        if hasattr(response, "usage"):
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        else:
            usage = self._estimate_usage(messages, text_content)
        
        result = LLMResponse(
            text=text_content,
            model=self.model,
            provider=self.provider_type,
            usage=usage,
            finish_reason=response.stop_reason,
            raw_response={"id": response.id, "stop_reason": response.stop_reason},
        )
        self._emit("chat_complete", {"response": result})
        return result

    async def _handle_async_streaming(self, response) -> AsyncIterator[StreamingChunk]:
        """Handle async streaming response."""
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content += block.text
        
        yield StreamingChunk(text=text_content, is_finished=True)

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
        """Anthropic doesn't provide embeddings API yet."""
        raise NotImplementedError("Anthropic doesn't provide embeddings API")

    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Anthropic doesn't provide embeddings API yet."""
        raise NotImplementedError("Anthropic doesn't provide embeddings API")

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return True
