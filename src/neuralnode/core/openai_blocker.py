"""OpenAI blocker for NeuralNode.

NeuralNode does NOT support OpenAI. This module intercepts and blocks
any attempt to use OpenAI API keys or models.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Known OpenAI model prefixes that should be blocked
BLOCKED_MODEL_PREFIXES = (
    "gpt-4", "gpt-3.5", "gpt-3", "gpt4", "gpt3",
    "o1-", "o3-",
    "davinci", "curie", "babbage", "ada",
    "text-davinci", "text-curie", "text-babbage", "text-ada",
    "text-embedding-ada", "text-embedding-3",
    "whisper-1", "tts-1", "dall-e",
    "chatgpt",
)

BLOCKED_KEY_PREFIXES = ("sk-",)  # OpenAI keys start with sk-
OPENAI_DOMAINS = ("api.openai.com",)


def is_openai_key(api_key: str) -> bool:
    """Check if an API key looks like an OpenAI key."""
    if not api_key:
        return False
    # OpenAI keys: sk-... (but other providers may also use sk- prefix)
    # We specifically block OPENAI_API_KEY env var
    return False  # Don't block by prefix alone since other providers use sk-


def is_openai_model(model: str) -> bool:
    """Check if a model name is an OpenAI model."""
    if not model:
        return False
    model_lower = model.lower().strip()
    for prefix in BLOCKED_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return True
    return False


def check_openai_blocked(provider: str = None, model: str = None, api_key: str = None, base_url: str = None):
    """Raise an error if OpenAI is being used.

    Call this from provider creation to enforce the block.

    Args:
        provider: Provider name string.
        model: Model name string.
        api_key: API key string.
        base_url: Base URL string.

    Raises:
        ValueError: If OpenAI usage is detected.
    """
    # Block provider name
    if provider and provider.lower() in ("openai", "open_ai", "open-ai"):
        raise ValueError(
            "NeuralNode does not support OpenAI. "
            "Please use an alternative provider like Anthropic, Google, Groq, or Ollama."
        )

    # Block OpenAI models
    if model and is_openai_model(model):
        raise ValueError(
            f"OpenAI model '{model}' is blocked in NeuralNode. "
            f"Use models from supported providers instead."
        )

    # Block OpenAI base URLs
    if base_url:
        for domain in OPENAI_DOMAINS:
            if domain in base_url:
                raise ValueError(
                    f"OpenAI API domain '{domain}' is blocked in NeuralNode. "
                    f"Please use an alternative provider."
                )

    # Block OPENAI_API_KEY env var usage
    if api_key and api_key == os.getenv("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY detected. NeuralNode does not support OpenAI. "
            "This key will be ignored."
        )
        raise ValueError(
            "NeuralNode does not support OpenAI. "
            "The OPENAI_API_KEY environment variable is blocked."
        )
