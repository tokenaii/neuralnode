"""Chat Model Providers for NeuralNode.

This module contains all chat-based LLM providers organized by service.
Each provider implements the BaseLLMProvider interface for chat completions.
"""

# Import all providers for convenient access
from .anthropic import AnthropicChatProvider
from .google import GoogleChatProvider
from .groq import GroqChatProvider
from .mistral import MistralChatProvider
from .cohere import CohereChatProvider
from .deepseek import DeepSeekChatProvider
from .perplexity import PerplexityChatProvider
from .ai21 import AI21ChatProvider
from .together import TogetherChatProvider
from .fireworks import FireworksChatProvider

# Convenience aliases
AnthropicProvider = AnthropicChatProvider
GoogleProvider = GoogleChatProvider
GroqProvider = GroqChatProvider
MistralProvider = MistralChatProvider
CohereProvider = CohereChatProvider
DeepSeekProvider = DeepSeekChatProvider

__all__ = [
    # Base chat providers
    "AnthropicChatProvider",
    "GoogleChatProvider",
    "CohereChatProvider",
    "MistralChatProvider",
    "GroqChatProvider",
    "DeepSeekChatProvider",
    "PerplexityChatProvider",
    "AI21ChatProvider",
    "TogetherChatProvider",
    "FireworksChatProvider",
    # Aliases
    "AnthropicProvider",
    "GoogleProvider",
    "GroqProvider",
    "MistralProvider",
    "CohereProvider",
    "DeepSeekProvider",
]
