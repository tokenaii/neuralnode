"""NeuralNode - A next-generation AI framework for LLMs and Agents.

This package provides a shortcut 'nn' alias for 'neuralnode'.

Usage:
    import neuralnode as nn
    ai = nn.NeuralNode(provider="auto")
    
    # Or use the shortcut alias
    import nn
    ai = nn.NeuralNode(provider="auto")
"""

from .core.ai import NeuralNode
from .core.types import (
    BaseProvider,
    ChatOptions,
    LLMResponse,
    Message,
    MessageRole,
    ProviderType,
    StreamingChunk,
    StructuredOutput,
    TokenUsage,
    ToolDefinition,
    ToolParameter,
)
from .agents.agent import Agent, Tool, create_tool
from .agents.enhanced_agent import EnhancedAgent, MultiStepAgent
from .memory.conversation import ConversationMemory, SlidingWindowMemory
from .chains.pipeline import (
    Chain,
    Pipeline,
    PromptTemplate,
    LLMChain,
    TransformChain,
    create_chain,
    prompt_chain,
)
from .rag.vectorstore import RAG, Document, TextSplitter, SimpleVectorStore, FileLoader
from .rag.advanced_vectorstores import FAISSVectorStore, ChromaDBVectorStore
from .rag.loaders import (
    CSVLoader,
    MarkdownLoader,
    HTMLLoader,
    JSONLoader,
    DirectoryLoader,
    ExtendedFileLoader,
)
from .local.hardware import HardwareDetector
from .local.quantization import ModelQuantizer, ModelDownloader, quantize_model
from .distributed.cluster import DistributedCluster
from .tools.web_search import WebSearch, search
from .tools.browser_automation import BrowserController, BrowserTools
from .tools.system_control import SystemController, FileManager, ProcessManager
from .tools.multimodal import SpeechToText, TextToSpeech, VisionProcessor
from .integrations.telegram import TelegramBot, TelegramAuth
from .utils.monitoring import NeuralMonitor, trace
from .utils.smart_cache import SmartCache, cached
from .training.finetune import FineTuner, RLHFTrainer

__version__ = "1.0.4"
__all__ = [
    # Core
    "NeuralNode",
    "Agent",
    "EnhancedAgent",
    "MultiStepAgent",
    "Tool",
    "create_tool",
    # Types
    "BaseProvider",
    "ChatOptions",
    "ConversationMemory",
    "SlidingWindowMemory",
    "LLMResponse",
    "Message",
    "MessageRole",
    "ProviderType",
    "StreamingChunk",
    "StructuredOutput",
    "TokenUsage",
    "ToolDefinition",
    "ToolParameter",
    # Chains
    "Chain",
    "Pipeline",
    "PromptTemplate",
    "LLMChain",
    "TransformChain",
    "create_chain",
    "prompt_chain",
    # RAG
    "RAG",
    "Document",
    "TextSplitter",
    "SimpleVectorStore",
    "FileLoader",
    "FAISSVectorStore",
    "ChromaDBVectorStore",
    # Loaders
    "CSVLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "JSONLoader",
    "DirectoryLoader",
    "ExtendedFileLoader",
    # Local
    "HardwareDetector",
    "ModelQuantizer",
    "ModelDownloader",
    "quantize_model",
    # Distributed
    "DistributedCluster",
    # Tools
    "WebSearch",
    "search",
    "BrowserController",
    "BrowserTools",
    "SystemController",
    "FileManager",
    "ProcessManager",
    "SpeechToText",
    "TextToSpeech",
    "VisionProcessor",
    # Integrations
    "TelegramBot",
    "TelegramAuth",
    # Monitoring
    "NeuralMonitor",
    "trace",
    # Cache
    "SmartCache",
    "cached",
    # Training
    "FineTuner",
    "RLHFTrainer",
]
