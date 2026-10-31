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


class AgentAction:
    """Represents an action the agent wants to take."""

    def __init__(self, tool: str, tool_input: Union[str, Dict[str, Any]], log: str = ""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

    def __repr__(self):
        return f"AgentAction(tool={self.tool}, tool_input={self.tool_input})"


class AgentFinish:
    """Represents the agent's final answer."""

    def __init__(self, return_values: Dict[str, Any], log: str = ""):
        self.return_values = return_values
        self.log = log

    @property
    def output(self) -> str:
        return self.return_values.get("output", "")


class AgentStep:
    """Represents a single step in the agent's execution."""

    def __init__(self, action: AgentAction, observation: str):
        self.action = action
        self.observation = observation


@dataclass
class AgentExecutorOutput:
    """Output from agent execution."""

    output: str
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReasoningStepType(Enum):
    """Types of reasoning steps."""

    THOUGHT = "thought"
    PLAN = "plan"
    ACTION = "action"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    CONCLUSION = "conclusion"


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""

    step_type: ReasoningStepType
    content: str
    iteration: int
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.step_type.value,
            "content": self.content,
            "iteration": self.iteration,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class AgentExecutor:
    """Executes an agent loop."""

    def __init__(
        self,
        agent: "BaseAgent",
        tools: List[BaseTool],
        max_iterations: int = 15,
        verbose: bool = False,
    ):
        self.agent = agent
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, query: str, **kwargs) -> AgentExecutorOutput:
        intermediate_steps: List[AgentStep] = []
        for i in range(self.max_iterations):
            action = self.agent.plan(intermediate_steps, **kwargs)
            if isinstance(action, AgentFinish):
                return AgentExecutorOutput(
                    output=action.output,
                    intermediate_steps=[
                        {"action": step.action.tool, "observation": step.observation}
                        for step in intermediate_steps
                    ],
                )

            tool_name = action.tool
            tool_input = action.tool_input
            if isinstance(tool_input, str):
                try:
                    tool_input_dict = json.loads(tool_input)
                except:
                    tool_input_dict = {"input": tool_input}
            else:
                tool_input_dict = tool_input

            if tool_name in self.tools:
                try:
                    res = self.tools[tool_name].execute(**tool_input_dict)
                    obs = res.content
                except Exception as e:
                    obs = str(e)
            else:
                obs = f"Tool not found: {tool_name}"

            intermediate_steps.append(AgentStep(action, obs))

        return AgentExecutorOutput(
            output="Max iterations reached without a final answer.",
            intermediate_steps=[
                {"action": step.action.tool, "observation": step.observation}
                for step in intermediate_steps
            ],
        )

    async def arun(self, query: str, **kwargs) -> AgentExecutorOutput:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(query, **kwargs))


class BaseAgent(CoreBaseAgent):
    """Enhanced base agent with planning capabilities."""

    def __init__(
        self,
        llm: Any,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 15,
        verbose: bool = False,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.tool_list = list(tools or [])
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        super().__init__(
            llm=llm,
            tools=self.tool_list,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
        )

    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Union[AgentAction, AgentFinish]:
        """Plan the next action or finish."""
        pass

    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ) -> Union[AgentAction, AgentFinish]:
        """Async plan the next action or finish."""
        pass

    def run(self, query: str, **kwargs) -> AgentExecutorOutput:
        """Run the agent on a query."""
        executor = AgentExecutor(
            agent=self,
            tools=self.tool_list,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )
        return executor.run(query, **kwargs)

    async def arun(self, query: str, **kwargs) -> AgentExecutorOutput:
        """Async run the agent."""
        executor = AgentExecutor(
            agent=self,
            tools=self.tool_list,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )
        return await executor.arun(query, **kwargs)

    def _get_tool_description(self) -> str:
        """Get description of all available tools."""
        descriptions = []
        for tool in self.tool_list:
            schema = tool.get_schema() if hasattr(tool, "get_schema") else {}
            func = schema.get("function", {})
            name = func.get("name", tool.name)
            desc = func.get("description", tool.description)
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)
