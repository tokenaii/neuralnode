"""Main NeuralNode interface."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Type, TypeVar, Union

from ..core.types import (
    BaseProvider,
    ChatOptions,
    LLMResponse,
    Message,
    MessageRole,
    ProviderType,
    StreamingChunk,
    ToolDefinition,
    TokenUsage,
)
from ..memory.conversation import ConversationMemory
from ..utils.auto_provider import AutoProvider
from ..utils.time_context import inject_time_context

T = TypeVar("T")


class NeuralNode:
    """
    Main interface for NeuralNode AI framework.
    
    Provides a unified API for interacting with LLMs across
    different providers (OpenAI, Anthropic, Ollama, etc.)
    """

    def __init__(
        self,
        provider: Union[str, ProviderType] = ProviderType.AUTO,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        memory: Optional[ConversationMemory] = None,
        auto_time: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize NeuralNode.
        
        Args:
            provider: The LLM provider to use ("auto", "openai", "anthropic", etc.)
            model: Model name to use
            api_key: API key for the provider
            base_url: Custom base URL for the provider
            memory: Conversation memory instance
            auto_time: Automatically inject current time into prompts
            **kwargs: Additional provider-specific options
        """
        self._auto_time = auto_time
        self._memory = memory or ConversationMemory()
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # Initialize provider
        provider_enum = provider if isinstance(provider, ProviderType) else ProviderType(provider)
        
        if provider_enum == ProviderType.AUTO:
            self._provider = AutoProvider.select(model, api_key)
        else:
            self._provider = self._create_provider(
                provider_enum, model, api_key, base_url, **kwargs
            )

    def _create_provider(
        self,
        provider: ProviderType,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        **kwargs: Any,
    ) -> BaseProvider:
        """Create a provider instance."""
        if provider == ProviderType.OPENAI:
            from ..providers.openai import OpenAIProvider
            return OpenAIProvider(
                model=model or "gpt-4",
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                **kwargs,
            )
        elif provider == ProviderType.ANTHROPIC:
            from ..providers.anthropic import AnthropicProvider
            return AnthropicProvider(
                model=model or "claude-3-sonnet-20240229",
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                **kwargs,
            )
        elif provider == ProviderType.GOOGLE:
            from ..providers.google import GoogleProvider
            return GoogleProvider(
                model=model or "gemini-pro",
                api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                **kwargs,
            )
        elif provider == ProviderType.OLLAMA:
            from ..providers.ollama import OllamaProvider
            return OllamaProvider(
                model=model or "llama3",
                base_url=base_url or "http://localhost:11434",
                **kwargs,
            )
        elif provider == ProviderType.HUGGINGFACE:
            from ..providers.huggingface import HuggingFaceProvider
            return HuggingFaceProvider(
                model=model or "HuggingFaceH4/zephyr-7b-beta",
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def provider(self) -> BaseProvider:
        """Get the underlying provider."""
        return self._provider

    @property
    def model(self) -> str:
        """Get the current model name."""
        return self._provider.model or "unknown"

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        remember: bool = True,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """
        Send a chat message to the LLM.
        
        Args:
            message: The user message
            system: System prompt/instruction
            history: Previous conversation messages
            tools: Tools available to the LLM
            stream: Whether to stream the response
            remember: Whether to store this in memory
            **kwargs: Additional options (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse or iterator of StreamingChunk if streaming
        """
        # Build messages
        messages: List[Message] = []
        
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        elif self._auto_time:
            # Inject time context automatically
            time_context = inject_time_context()
            messages.append(Message(role=MessageRole.SYSTEM, content=time_context))
        
        # Add history or memory
        if history:
            messages.extend(history)
        elif remember and self._memory:
            messages.extend(self._memory.get_messages())
        
        messages.append(Message(role=MessageRole.USER, content=message))
        
        # Make request
        result = self._provider.chat(messages, tools=tools, stream=stream, **kwargs)
        
        # Handle streaming
        if stream:
            return self._wrap_streaming(result, remember)
        
        # Store in memory
        if remember and self._memory:
            self._memory.add(Message(role=MessageRole.USER, content=message))
            self._memory.add(Message(role=MessageRole.ASSISTANT, content=result.text))
        
        self._emit("chat_complete", {"response": result})
        return result

    def _wrap_streaming(
        self,
        stream_result: Iterator[StreamingChunk],
        remember: bool,
    ) -> Iterator[StreamingChunk]:
        """Wrap streaming result to capture full text."""
        full_text = ""
        for chunk in stream_result:
            full_text += chunk.text
            yield chunk
        
        if remember and self._memory:
            # Note: user message already added before streaming
            self._memory.add(Message(role=MessageRole.ASSISTANT, content=full_text))

    async def achat(
        self,
        message: str,
        system: Optional[str] = None,
        history: Optional[List[Message]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        remember: bool = True,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async version of chat."""
        messages: List[Message] = []
        
        if system:
            messages.append(Message(role=MessageRole.SYSTEM, content=system))
        elif self._auto_time:
            time_context = inject_time_context()
            messages.append(Message(role=MessageRole.SYSTEM, content=time_context))
        
        if history:
            messages.extend(history)
        elif remember and self._memory:
            messages.extend(self._memory.get_messages())
        
        messages.append(Message(role=MessageRole.USER, content=message))
        
        result = await self._provider.achat(messages, tools=tools, stream=stream, **kwargs)
        
        if stream:
            return self._wrap_async_streaming(result, remember)
        
        if remember and self._memory:
            self._memory.add(Message(role=MessageRole.USER, content=message))
            self._memory.add(Message(role=MessageRole.ASSISTANT, content=result.text))
        
        self._emit("chat_complete", {"response": result})
        return result

    async def _wrap_async_streaming(
        self,
        stream_result: AsyncIterator[StreamingChunk],
        remember: bool,
    ) -> AsyncIterator[StreamingChunk]:
        """Wrap async streaming result."""
        full_text = ""
        async for chunk in stream_result:
            full_text += chunk.text
            yield chunk
        
        if remember and self._memory:
            self._memory.add(Message(role=MessageRole.ASSISTANT, content=full_text))

    def complete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Text completion (non-chat)."""
        return self._provider.complete(prompt, stream=stream, **kwargs)

    async def acomplete(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async text completion."""
        return await self._provider.acomplete(prompt, stream=stream, **kwargs)

    def structured_output(
        self,
        message: str,
        output_schema: Type[T],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """
        Get structured output from the LLM.
        
        Args:
            message: The user message
            output_schema: Pydantic model defining the output structure
            system: System prompt
            **kwargs: Additional options
            
        Returns:
            Instance of output_schema with LLM response
        """
        import json
        
        schema_desc = output_schema.model_json_schema()
        prompt = f"""{message}

Respond with a JSON object matching this schema:
{json.dumps(schema_desc, indent=2)}

Respond ONLY with valid JSON."""
        
        response = self.chat(prompt, system=system, remember=False, **kwargs)
        
        if isinstance(response, Iterator):
            # Handle streaming - collect all chunks
            text = "".join(chunk.text for chunk in response)
        else:
            text = response.text
        
        # Parse JSON response
        try:
            data = json.loads(text.strip())
            return output_schema(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured output: {e}\nResponse: {text}")

    async def astructured_output(
        self,
        message: str,
        output_schema: Type[T],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Async version of structured_output."""
        import json
        
        schema_desc = output_schema.model_json_schema()
        prompt = f"""{message}

Respond with a JSON object matching this schema:
{json.dumps(schema_desc, indent=2)}

Respond ONLY with valid JSON."""
        
        response = await self.achat(prompt, system=system, remember=False, **kwargs)
        
        if isinstance(response, AsyncIterator):
            text = ""
            async for chunk in response:
                text += chunk.text
        else:
            text = response.text
        
        try:
            data = json.loads(text.strip())
            return output_schema(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured output: {e}\nResponse: {text}")

    def embed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Generate embeddings."""
        return self._provider.embed(texts, **kwargs)

    async def aembed(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Async embeddings."""
        return await self._provider.aembed(texts, **kwargs)

    def agent(
        self,
        tools: Optional[List[ToolDefinition]] = None,
        system: Optional[str] = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> Agent:
        """Create an AI agent."""
        from ..agents.agent import Agent
        return Agent(
            llm=self,
            tools=tools or [],
            system=system,
            max_iterations=max_iterations,
            **kwargs,
        )

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._memory.clear()

    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add a callback for events."""
        self._callbacks.append(callback)
        self._provider.add_callback(callback)

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        """Emit an event."""
        for callback in self._callbacks:
            try:
                callback(event, data)
            except Exception:
                pass

    @property
    def supports_tools(self) -> bool:
        """Whether current provider supports tool calling."""
        return self._provider.supports_tools

    @property
    def supports_streaming(self) -> bool:
        """Whether current provider supports streaming."""
        return self._provider.supports_streaming

    @property
    def supports_structured_output(self) -> bool:
        """Whether current provider supports structured output."""
        return self._provider.supports_structured_output
