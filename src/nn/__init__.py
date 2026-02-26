"""NeuralNode (nn) - A next-generation AI framework for LLMs and Agents.

This package provides a shortcut 'nn' alias for 'neuralnode'.

Usage:
    import nn
    ai = nn.NeuralNode(provider="auto")
    
    # Or
    from nn import NeuralNode, Agent
"""

# Re-export everything from neuralnode
try:
    from neuralnode import *
    from neuralnode.core.ai import NeuralNode
    from neuralnode.agents import Agent, Tool
    from neuralnode.rag import RAG
    from neuralnode.memory import ConversationMemory
    from neuralnode.local import HardwareDetector
    from neuralnode.core.types import (
        Message,
        MessageRole,
        ProviderType,
        ToolDefinition,
        LLMResponse,
        StreamingChunk,
    )
except ImportError:
    # Fallback for development
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from neuralnode import *

__version__ = "0.1.0"
__all__ = [
    "NeuralNode",
    "Agent",
    "Tool",
    "RAG",
    "ConversationMemory",
    "HardwareDetector",
    "Message",
    "MessageRole",
    "ProviderType",
    "ToolDefinition",
    "LLMResponse",
    "StreamingChunk",
]
