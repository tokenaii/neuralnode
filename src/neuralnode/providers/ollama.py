"""Ollama local provider implementation."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import requests
import aiohttp

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


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.provider_type = ProviderType.OLLAMA

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], str]:
        """
        Convert messages to Ollama format.
        Returns (system_prompt, full_prompt)
        """
        system_parts = []
        conversation_parts = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
            elif msg.role == MessageRole.USER:
                conversation_parts.append(f"User: {msg.content}")
            elif msg.role == MessageRole.ASSISTANT:
                conversation_parts.append(f"Assistant: {msg.content}")
        
        system_prompt = "\n".join(system_parts) if system_parts else None
        full_prompt = "\n".join(conversation_parts)
        
        return system_prompt, full_prompt

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, Iterator[StreamingChunk]]:
        """Send chat completion request."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        
        system_prompt, prompt = self._convert_messages(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", -1),
                "top_p": kwargs.get("top_p", 0.9),
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if stream:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
            )
            return self._handle_streaming(response)
        
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        
        result = LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            provider=self.provider_type,
            usage=TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            finish_reason="stop" if not data.get("done", False) else None,
            raw_response=data,
        )
        self._emit("chat_complete", {"response": result})
        return result

    def _handle_streaming(self, response) -> Iterator[StreamingChunk]:
        """Handle streaming response."""
        import json
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    text = data.get("response", "")
                    is_done = data.get("done", False)
                    
                    usage = None
                    if is_done:
                        usage = TokenUsage(
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=data.get("eval_count", 0),
                            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                        )
                    
                    yield StreamingChunk(
                        text=text,
                        is_finished=is_done,
                        usage=usage,
                    )
                    
                    if is_done:
                        break
                except json.JSONDecodeError:
                    continue

    async def achat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[LLMResponse, AsyncIterator[StreamingChunk]]:
        """Async chat completion."""
        self._emit("chat_start", {"messages": messages, "stream": stream})
        
        system_prompt, prompt = self._convert_messages(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", -1),
                "top_p": kwargs.get("top_p", 0.9),
            },
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        if stream:
            return self._handle_async_streaming(payload)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                data = await response.json()
        
        result = LLMResponse(
            text=data.get("response", ""),
            model=self.model,
            provider=self.provider_type,
            usage=TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
            finish_reason="stop" if not data.get("done", False) else None,
            raw_response=data,
        )
        self._emit("chat_complete", {"response": result})
        return result

    async def _handle_async_streaming(self, payload: Dict[str, Any]) -> AsyncIterator[StreamingChunk]:
        """Handle async streaming response."""
        import json
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
            ) as response:
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode("utf-8"))
                            text = data.get("response", "")
                            is_done = data.get("done", False)
                            
                            usage = None
                            if is_done:
                                usage = TokenUsage(
                                    prompt_tokens=data.get("prompt_eval_count", 0),
                                    completion_tokens=data.get("eval_count", 0),
                                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                                )
                            
                            yield StreamingChunk(
                                text=text,
                                is_finished=is_done,
                                usage=usage,
                            )
                            
                            if is_done:
                                break
                        except json.JSONDecodeError:
                            continue

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
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": kwargs.get("embedding_model", "nomic-embed-text"),
                    "prompt": text,
                },
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data.get("embedding", []))
        
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
        return False

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_structured_output(self) -> bool:
        return False
