# ruff: noqa: UP006, UP035, UP045
"""Horus Cyber Nano Early Access API provider."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Iterable, Iterator
from typing import Any, Dict, List, Optional

import requests

from ...core import Message
from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType

DEFAULT_API_BASE_URL = "https://api.tokenai.llc/v1"
DEFAULT_MODEL = "horus-cyber-nano-1.0-beta"


class HorusCyberNanoAPIError(RuntimeError):
    """An error returned by the Horus Cyber Nano Early Access API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id
        self.retry_after = retry_after


class HorusCyberNanoProvider(BaseLLMProvider):
    """Client for the gated Horus Cyber Nano 1.0 BETA Text-to-Text API."""

    provider_name = "horus-cyber-nano"
    provider_type = ProviderType.CLOUD_API
    default_model = DEFAULT_MODEL
    model_type = "text-to-text"
    api_key_env_var = "HORUS_CYPER_NANO_API_KEY"
    base_url_env_var = "HORUS_CYPER_NANO_BASE_URL"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            text_completion=True,
            streaming=True,
            function_calling=True,
            tool_use=True,
            vision=False,
            json_mode=False,
            embeddings=False,
            system_prompt=True,
            context_window=32768,
            max_output_tokens=4096,
            async_support=True,
            rate_limit_info={"student_rpm": 10, "developer_researcher_rpm": 20},
        )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_reference: Optional[str] = None,
        client_name: str = "NeuralNode Framework",
        timeout: int = 300,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.getenv(self.api_key_env_var)
        resolved_base_url = base_url or os.getenv(self.base_url_env_var) or DEFAULT_API_BASE_URL
        if not resolved_api_key:
            raise ValueError(
                f"Horus Cyber Nano Access Token is required. Set {self.api_key_env_var} "
                "or pass api_key=..."
            )

        super().__init__(
            model=model or self.default_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            timeout=timeout,
            **kwargs,
        )
        self.conversation_id = self._normalize_conversation_id(conversation_id)
        self.user_reference = user_reference
        self.client_name = client_name
        if not isinstance(max_retries, int) or not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be an integer between 0 and 5.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative.")
        self.max_retries = max_retries
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._session = requests.Session()

    @staticmethod
    def _normalize_conversation_id(value: Optional[str]) -> str:
        if value is None:
            return str(uuid.uuid4())
        try:
            return str(uuid.UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("conversation_id must be a valid UUID.") from exc

    @staticmethod
    def _normalize_messages(messages: Iterable[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for message in messages:
            if isinstance(message, Message):
                role = message.role.value
                content = message.content
                normalized_message: Dict[str, Any] = {
                    "role": str(role),
                    "content": str(content),
                }
            elif isinstance(message, dict):
                raw_role = message.get("role", "user")
                role = getattr(raw_role, "value", raw_role)
                content = message.get("content")
                normalized_message = {"role": str(role), "content": content}
                for key in ("name", "tool_call_id", "tool_calls"):
                    if message.get(key) is not None:
                        normalized_message[key] = message[key]
            else:
                raise TypeError(f"Unsupported message type: {type(message)!r}")

            role = str(role)
            if role not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"Unsupported message role: {role}")
            if content is not None:
                normalized_message["content"] = str(content)
            if not normalized_message.get("content") and not normalized_message.get("tool_calls"):
                raise ValueError("A message must contain content or tool calls.")
            if role == "tool" and not normalized_message.get("tool_call_id"):
                raise ValueError("Tool messages require tool_call_id.")
            normalized.append(normalized_message)

        if not normalized:
            raise ValueError("At least one message is required.")
        return normalized

    @staticmethod
    def _parse_error(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or "API request failed.")
            if error:
                return str(error)
        return f"API request failed with HTTP {response.status_code}."

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "NeuralNode-HorusCyberNano/1.0",
            "X-Client-Name": self.client_name,
        }

    def _iter_stream(
        self,
        response: requests.Response,
        conversation_id: str,
        model: str,
        attempts: int,
    ) -> Iterator[Dict[str, Any]]:
        try:
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                value = raw_line[5:].strip()
                if value == "[DONE]":
                    break
                data = json.loads(value)
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                self.conversation_id = str(data.get("conversation_id") or conversation_id)
                yield {
                    "content": delta.get("content") or "",
                    "model": data.get("model", model),
                    "provider": self.provider_name,
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": delta.get("tool_calls", []),
                    "usage": data.get("usage", {}),
                    "metadata": {
                        "conversation_id": self.conversation_id,
                        "request_id": data.get("request_id")
                        or response.headers.get("X-Request-ID"),
                        "rate_limit": response.headers.get("X-RateLimit-Limit"),
                        "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                        "attempts": attempts,
                    },
                }
        finally:
            response.close()

    def _post_chat(self, payload: Dict[str, Any], stream: bool) -> tuple[requests.Response, int]:
        retryable_statuses = {429, 502, 503}
        last_exception: Optional[requests.RequestException] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
            except requests.RequestException as exc:
                last_exception = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(self.retry_backoff_seconds * (2**attempt), 30.0))
                continue

            should_return = (
                response.ok
                or response.status_code not in retryable_statuses
                or attempt >= self.max_retries
            )
            if should_return:
                return response, attempt + 1

            retry_after_value = response.headers.get("Retry-After")
            retry_after = (
                int(retry_after_value)
                if retry_after_value and retry_after_value.isdigit()
                else self.retry_backoff_seconds * (2**attempt)
            )
            response.close()
            time.sleep(min(float(retry_after), 30.0))

        raise HorusCyberNanoAPIError(
            f"Could not reach the Horus Cyber Nano API: {last_exception}"
        ) from last_exception

    def chat(
        self,
        messages: Iterable[Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        if response_format:
            raise NotImplementedError(
                "Structured response formats are not enabled during Early Access."
            )

        conversation_id = self._normalize_conversation_id(
            kwargs.pop("conversation_id", self.conversation_id)
        )
        payload: Dict[str, Any] = {
            "model": model or self.model or self.default_model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "conversation_id": conversation_id,
            "client_name": kwargs.pop("client_name", self.client_name),
            "stream": stream,
        }
        user_reference = kwargs.pop("user_reference", self.user_reference)
        if user_reference:
            payload["user"] = user_reference
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        elif tool_choice:
            payload["tool_choice"] = tool_choice
        payload.update(kwargs)

        response, attempts = self._post_chat(payload, stream)

        request_id = response.headers.get("X-Request-ID")
        if not response.ok:
            retry_after_value = response.headers.get("Retry-After")
            retry_after = (
                int(retry_after_value)
                if retry_after_value and retry_after_value.isdigit()
                else None
            )
            raise HorusCyberNanoAPIError(
                self._parse_error(response),
                status_code=response.status_code,
                request_id=request_id,
                retry_after=retry_after,
            )

        if stream:
            return self._iter_stream(response, conversation_id, payload["model"], attempts)

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        answer = choice.get("message") or {}
        self.conversation_id = str(data.get("conversation_id") or conversation_id)
        return {
            "content": answer.get("content") or "",
            "model": data.get("model", payload["model"]),
            "provider": self.provider_name,
            "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason"),
            "tool_calls": answer.get("tool_calls", []),
            "metadata": {
                "conversation_id": self.conversation_id,
                "request_id": data.get("request_id") or request_id,
                "rate_limit": response.headers.get("X-RateLimit-Limit"),
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                "attempts": attempts,
            },
        }

    async def achat(self, messages: Iterable[Any], **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    def get_access_info(self) -> Dict[str, Any]:
        """Return the active token status and assigned RPM limit."""
        response = self._session.get(
            f"{self.base_url}/access",
            headers=self._headers(),
            timeout=self.timeout,
        )
        if not response.ok:
            raise HorusCyberNanoAPIError(
                self._parse_error(response),
                status_code=response.status_code,
                request_id=response.headers.get("X-Request-ID"),
            )
        return response.json()

    def new_conversation(self) -> str:
        """Start a new server-side conversation and return its UUID."""
        self.conversation_id = str(uuid.uuid4())
        return self.conversation_id

    def embed(self, texts: Any, model: Optional[str] = None, **kwargs: Any) -> Any:
        raise NotImplementedError("Embeddings are not available for Horus Cyber Nano Early Access.")

    async def aembed(self, texts: Any, model: Optional[str] = None, **kwargs: Any) -> Any:
        raise NotImplementedError("Embeddings are not available for Horus Cyber Nano Early Access.")


HorusCyberNano = HorusCyberNanoProvider


__all__ = [
    "HorusCyberNanoAPIError",
    "HorusCyberNanoProvider",
    "HorusCyberNano",
]
