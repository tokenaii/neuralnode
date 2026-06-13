"""Compatibility shim for historical OpenAI blocking hooks.

The current NeuralNode feature set includes OpenAI-compatible providers, so the
old blocker API remains as a no-op to avoid breaking imports.
"""

from __future__ import annotations

from typing import Optional


def is_openai_key(api_key: Optional[str]) -> bool:
    return False


def is_openai_model(model: Optional[str]) -> bool:
    return False


def check_openai_blocked(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> None:
    return None


__all__ = ["check_openai_blocked", "is_openai_key", "is_openai_model"]
