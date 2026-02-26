"""Utility modules for NeuralNode."""

from .auto_provider import AutoProvider
from .time_context import inject_time_context, get_current_time_info, format_time_for_prompt
from .chat_templates import (
    ChatTemplate,
    ChatTemplateManager,
    apply_chat_template,
    get_stop_sequences,
    LLAMA3_TEMPLATE,
    QWEN_TEMPLATE,
    MISTRAL_TEMPLATE,
    ZEPHYR_TEMPLATE,
    PHI3_TEMPLATE,
    GEMMA_TEMPLATE,
    CHATML_TEMPLATE,
)
from .retry import (
    with_retry,
    with_rate_limit,
    RetryConfig,
    RateLimiter,
    AdaptiveRateLimiter,
    DEFAULT_RETRY,
    API_RETRY,
    EMBEDDING_RETRY,
)
from .cache import (
    BaseCache,
    InMemoryCache,
    DiskCache,
    SemanticCache,
    cached,
    get_default_cache,
)
from .output_parsers import (
    OutputParser,
    StrOutputParser,
    JSONOutputParser,
    PydanticOutputParser,
    ListOutputParser,
    CommaSeparatedListOutputParser,
    BooleanOutputParser,
    MarkdownOutputParser,
    RegexOutputParser,
    EnumOutputParser,
    parse_json,
    parse_list,
    parse_pydantic,
)

__all__ = [
    # Auto provider
    "AutoProvider",
    # Time context
    "inject_time_context",
    "get_current_time_info",
    "format_time_for_prompt",
    # Chat templates
    "ChatTemplate",
    "ChatTemplateManager",
    "apply_chat_template",
    "get_stop_sequences",
    "LLAMA3_TEMPLATE",
    "QWEN_TEMPLATE",
    "MISTRAL_TEMPLATE",
    "ZEPHYR_TEMPLATE",
    "PHI3_TEMPLATE",
    "GEMMA_TEMPLATE",
    "CHATML_TEMPLATE",
    # Retry
    "with_retry",
    "with_rate_limit",
    "RetryConfig",
    "RateLimiter",
    "AdaptiveRateLimiter",
    "DEFAULT_RETRY",
    "API_RETRY",
    "EMBEDDING_RETRY",
    # Cache
    "BaseCache",
    "InMemoryCache",
    "DiskCache",
    "SemanticCache",
    "cached",
    "get_default_cache",
    # Output parsers
    "OutputParser",
    "StrOutputParser",
    "JSONOutputParser",
    "PydanticOutputParser",
    "ListOutputParser",
    "CommaSeparatedListOutputParser",
    "BooleanOutputParser",
    "MarkdownOutputParser",
    "RegexOutputParser",
    "EnumOutputParser",
    "parse_json",
    "parse_list",
    "parse_pydantic",
]
