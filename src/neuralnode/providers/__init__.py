"""
NeuralNode — Comprehensive LLM Provider Registry
Supports 50+ local and cloud providers (OpenAI excluded).
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from ..core import Message, MessageRole, ToolCall, LLMResponse, StreamingChunk, BaseTool
from .base import (
    BaseLLMProvider,
    BaseProvider,
    ProviderCapabilities,
    ProviderType,
    ModelType,
    ModelInfo,
    ProviderRegistry,
    create_provider,
    auto_select_provider,
    get_available_providers,
    get_provider_info,
    validate_provider_config,
)

# ── Chat Model Providers ───────────────────────────────────────────────────────
# Import from the new modular chat package
from .chat import (
    AnthropicChatProvider,
    GoogleChatProvider,
    CohereChatProvider,
    MistralChatProvider,
    GroqChatProvider,
    DeepSeekChatProvider,
    PerplexityChatProvider,
    AI21ChatProvider,
    TogetherChatProvider,
    FireworksChatProvider,
    BedrockChatProvider,
    VertexAIChatProvider,
    OpenAIChatProvider,
    OpenRouterChatProvider,
    AgentRouterChatProvider,
)

# Aliases for backward compatibility
AnthropicProvider = AnthropicChatProvider
GoogleProvider = GoogleChatProvider
CohereProvider = CohereChatProvider
MistralProvider = MistralChatProvider
GroqProvider = GroqChatProvider
DeepSeekProvider = DeepSeekChatProvider
OpenAIProvider = OpenAIChatProvider
OpenRouterProvider = OpenRouterChatProvider
AgentRouterProvider = AgentRouterChatProvider

# ── Text Generation Providers ──────────────────────────────────────────────────
from .text_generation import (
    HuggingFaceTextProvider,
    AI21TextProvider,
    CohereTextProvider,
    AlephAlphaTextProvider,
    NLPCloudTextProvider,
    GooseAITextProvider,
    PetalsTextProvider,
    TextSynthProvider,
    InferKitProvider,
)

# ── Embedding Providers ────────────────────────────────────────────────────────
from .embeddings import (
    CohereEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    GoogleEmbeddingProvider,
    MistralEmbeddingProvider,
    VoyageEmbeddingProvider,
    JinaEmbeddingProvider,
    MixedbreadEmbeddingProvider,
    NomicEmbeddingProvider,
    OllamaEmbeddingProvider,
    TransformersEmbeddingProvider,
    SentenceTransformersProvider,
    InstructorEmbeddingProvider,
    FastEmbedProvider,
    EmbeddingStudioProvider,
    UpstageEmbeddingProvider,
    GeckoEmbeddingProvider,
)

# ── Local Model Providers ──────────────────────────────────────────────────────
from .local_providers import (
    OllamaProvider,
    LlamaCppProvider,
    TransformersProvider,
    LlamaFileProvider,
    KoboldCppProvider,
    TextGenerationWebuiProvider,
    ExLlamaProvider,
    AutoGPTQProvider,
    AutoAWQProvider,
    VLLMProvider,
    TGIProvider,
    DeepSpeedProvider,
    RayServeProvider,
    MLflowProvider,
    BentoMLProvider,
    TritonProvider,
)
from .universal_local import UniversalLocalProvider, load_local_model

# ── Custom Model Providers ─────────────────────────────────────────────────────
try:
    from .horus import HorusProvider, HorusModel, HorusLensModel, Horus, load_horus, load_horus_lens, print_model_list
except ImportError:
    HorusProvider = None  # type: ignore
    HorusModel = None  # type: ignore
    HorusLensModel = None  # type: ignore
    Horus = None  # type: ignore
    load_horus = None  # type: ignore
    load_horus_lens = None  # type: ignore
    print_model_list = None  # type: ignore

# MLX Provider (Apple Silicon — optional)
try:
    from .local_providers import MLXProvider
except ImportError:
    MLXProvider = None  # type: ignore

# ── Convenience aliases ────────────────────────────────────────────────────────

# ── Local provider is-local detection helper ───────────────────────────────────

_LOCAL_PROVIDER_NAMES = {
    "ollama", "llamacpp", "llama.cpp", "transformers", "llamafile",
    "koboldcpp", "textgenwebui", "exllama", "autogptq", "autoawq",
    "vllm", "tgi", "deepspeed", "rayserve",
    "mlflow", "bentoml", "triton", "mlx",
}

# ── Register all providers in ProviderRegistry ────────────────────────────────

_PROVIDER_MAP = {
    # Cloud Chat
    "anthropic":  AnthropicChatProvider,
    "claude":     AnthropicChatProvider,
    "google":     GoogleChatProvider,
    "gemini":     GoogleChatProvider,
    "cohere":     CohereChatProvider,
    "mistral":    MistralChatProvider,
    "groq":       GroqChatProvider,
    "deepseek":   DeepSeekChatProvider,
    "perplexity": PerplexityChatProvider,
    "ai21":       AI21ChatProvider,
    "together":   TogetherChatProvider,
    "fireworks":  FireworksChatProvider,
    "bedrock":    BedrockChatProvider,
    "vertexai":   VertexAIChatProvider,
    "openai":     OpenAIChatProvider,
    "openrouter": OpenRouterChatProvider,
    "agentrouter": AgentRouterChatProvider,
    # Local
    "ollama":        OllamaProvider,
    "llamacpp":      LlamaCppProvider,
    "llama.cpp":     LlamaCppProvider,
    "transformers":  TransformersProvider,
    "llamafile":     LlamaFileProvider,
    "koboldcpp":     KoboldCppProvider,
    "textgenwebui":  TextGenerationWebuiProvider,
    "exllama":       ExLlamaProvider,
    "autogptq":      AutoGPTQProvider,
    "autoawq":       AutoAWQProvider,
    "vllm":          VLLMProvider,
    "tgi":           TGIProvider,
    "deepspeed":     DeepSpeedProvider,
    "rayserve":      RayServeProvider,
    "mlflow":        MLflowProvider,
    "bentoml":       BentoMLProvider,
    "triton":        TritonProvider,
}

if MLXProvider is not None:
    _PROVIDER_MAP["mlx"] = MLXProvider

if HorusProvider is not None:
    _PROVIDER_MAP["horus"] = HorusProvider
    _PROVIDER_MAP["tokenaii/horus"] = HorusProvider
    _LOCAL_PROVIDER_NAMES.add("horus")

for _name, _cls in _PROVIDER_MAP.items():
    ProviderRegistry.register(_name, _cls)

# ── Local provider is-local detection helper ───────────────────────────────────

def is_local_provider(name: str) -> bool:
    """Return True if the given provider name is a local model provider."""
    return name.lower() in _LOCAL_PROVIDER_NAMES


# ── Exports ────────────────────────────────────────────────────────────────────

__all__ = [
    # Base
    "BaseLLMProvider", "BaseProvider",
    "ProviderCapabilities", "ProviderType", "ModelType", "ModelInfo",
    "ProviderRegistry",
    "create_provider",
    "auto_select_provider",
    "get_available_providers",
    "get_provider_info",
    "validate_provider_config",
    "is_local_provider",

    # Chat providers
    "AnthropicChatProvider", "GoogleChatProvider", "CohereChatProvider",
    "MistralChatProvider", "GroqChatProvider", "DeepSeekChatProvider",
    "PerplexityChatProvider", "AI21ChatProvider",
    "TogetherChatProvider", "FireworksChatProvider", "BedrockChatProvider",
    "VertexAIChatProvider", "OpenAIChatProvider", "OpenRouterChatProvider",
    "AgentRouterChatProvider",

    # Aliases
    "AnthropicProvider", "GoogleProvider", "GroqProvider",
    "MistralProvider", "CohereProvider", "DeepSeekProvider",
    "OpenAIProvider", "OpenRouterProvider", "AgentRouterProvider",

    # Text Generation
    "HuggingFaceTextProvider", "AI21TextProvider", "CohereTextProvider",
    "AlephAlphaTextProvider", "NLPCloudTextProvider", "GooseAITextProvider",
    "PetalsTextProvider", "TextSynthProvider", "InferKitProvider",

    # Embeddings
    "CohereEmbeddingProvider", "HuggingFaceEmbeddingProvider",
    "GoogleEmbeddingProvider", "MistralEmbeddingProvider",
    "VoyageEmbeddingProvider", "JinaEmbeddingProvider",
    "MixedbreadEmbeddingProvider", "NomicEmbeddingProvider",
    "OllamaEmbeddingProvider", "TransformersEmbeddingProvider",
    "SentenceTransformersProvider", "InstructorEmbeddingProvider",
    "FastEmbedProvider", "EmbeddingStudioProvider",
    "UpstageEmbeddingProvider", "GeckoEmbeddingProvider",

    # Local
    "OllamaProvider", "LlamaCppProvider", "TransformersProvider",
    "LlamaFileProvider", "KoboldCppProvider", "TextGenerationWebuiProvider",
    "ExLlamaProvider", "AutoGPTQProvider", "AutoAWQProvider",
    "VLLMProvider", "TGIProvider",
    "DeepSpeedProvider", "RayServeProvider", "MLflowProvider",
    "BentoMLProvider", "TritonProvider", "MLXProvider",
    "UniversalLocalProvider", "load_local_model",
    
    # Custom Models
    "HorusProvider", "HorusModel", "HorusLensModel", "Horus", "load_horus", "load_horus_lens", "print_model_list",
]
