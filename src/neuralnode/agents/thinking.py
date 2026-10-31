"""Agent system for NeuralNode with advanced reasoning and tool calling."""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from ..core import (
    BaseProvider,
    BaseTool,
    Message,
    MessageRole,
    ToolCall,
    BaseAgent as CoreBaseAgent,
    ensure_llm_response,
)

logger = logging.getLogger(__name__)
from .base import *
from .agents import *


_THINKING_SYSTEM_SUFFIX = """

## Reasoning Instructions
Before answering, reason through your response step-by-step inside <think>...</think> tags.
After reasoning, provide your final answer outside the tags.

Example:
<think>
Let me break this down...
Step 1: ...
Conclusion: ...
</think>
[Your actual answer here]
"""


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return text.strip()


class ThinkingAgent:
    """Wraps any agent with a step-by-step reasoning (thinking) layer.

    Injects a Chain-of-Thought instruction into the system prompt so the
    model reasons before answering. Works best with local models.

    Example::

        from neuralnode.agents import ThinkingAgent, ReActAgent

        agent = ThinkingAgent(ReActAgent(llm, tools=my_tools))
        print(agent.run("Explain quantum entanglement"))
    """

    def __init__(
        self,
        agent: "BaseAgent",
        strip_thinking: bool = True,
        thinking_prompt: Optional[str] = None,
    ):
        """
        Args:
            agent: Any BaseAgent instance to wrap.
            strip_thinking: Remove <think> blocks from the returned output.
            thinking_prompt: Custom instruction appended to the system prompt.
        """
        self.agent = agent
        self.strip_thinking = strip_thinking
        self._inject(thinking_prompt or _THINKING_SYSTEM_SUFFIX)

    @property
    def llm(self):
        return self.agent.llm

    @property
    def tools(self):
        if hasattr(self.agent, 'tool_list'):
            return self.agent.tool_list
        return list(self.agent.tools.values())

    def _inject(self, suffix: str) -> None:
        current = getattr(self.agent, 'system_prompt', '') or ''
        if 'Reasoning Instructions' not in current:
            self.agent.system_prompt = current + suffix

    def run(self, task: str, **kwargs) -> Any:
        result = self.agent.run(task, **kwargs)
        if self.strip_thinking:
            if isinstance(result, AgentExecutorOutput):
                result.output = _strip_thinking(result.output)
            elif isinstance(result, str):
                result = _strip_thinking(result)
        return result

    async def arun(self, task: str, **kwargs) -> Any:
        result = await self.agent.arun(task, **kwargs)
        if self.strip_thinking:
            if isinstance(result, AgentExecutorOutput):
                result.output = _strip_thinking(result.output)
            elif isinstance(result, str):
                result = _strip_thinking(result)
        return result


def create_thinking_agent(
    llm: Any,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    strip_thinking: bool = True,
) -> ThinkingAgent:
    """Create a ReActAgent wrapped with a thinking/reasoning layer.

    Args:
        llm: LLM provider.
        tools: List of BaseTool instances.
        system_prompt: Optional custom system prompt.
        strip_thinking: Whether to remove <think> blocks from output.

    Returns:
        ThinkingAgent instance.
    """
    base = ReActAgent(llm, tools or [], system_prompt)
    return ThinkingAgent(base, strip_thinking=strip_thinking)



__all__ = [
    # Dataclasses
    "AgentAction",
    "AgentFinish",
    "AgentStep",
    "AgentExecutorOutput",
    "ReasoningStep",
    "ReasoningStepType",
    # Core classes
    "AgentExecutor",
    "BaseAgent",
    # Agent types
    "SimpleAgent",
    "ReActAgent",
    "ChainOfThoughtAgent",
    "ReflectiveAgent",
    "ToolCallingAgent",
    "StructuredChatAgent",
    "ThinkingAgent",
    # Builder
    "AgentBuilder",
    # Factory functions
    "create_agent",
    "create_react_agent",
    "create_structured_chat_agent",
    "create_tool_calling_agent",
    "create_thinking_agent",
]
