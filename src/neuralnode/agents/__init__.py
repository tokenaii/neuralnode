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
    
    def __init__(
        self,
        tool: str,
        tool_input: Union[str, Dict[str, Any]],
        log: str = ""
    ):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log
    
    def __repr__(self):
        return f"AgentAction(tool={self.tool}, tool_input={self.tool_input})"


class AgentFinish:
    """Represents the agent's final answer."""
    
    def __init__(
        self,
        return_values: Dict[str, Any],
        log: str = ""
    ):
        self.return_values = return_values
        self.log = log
    
    @property
    def output(self) -> str:
        return self.return_values.get("output", "")


class AgentStep:
    """Represents a single step in the agent's execution."""
    
    def __init__(
        self,
        action: AgentAction,
        observation: str
    ):
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
            "timestamp": self.timestamp
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
                    ]
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
            ]
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
        super().__init__(llm=llm, tools=self.tool_list, system_prompt=system_prompt, max_iterations=max_iterations)
    
    @abstractmethod
    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """Plan the next action or finish."""
        pass
    
    @abstractmethod
    async def aplan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
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


class ChainOfThoughtAgent(BaseAgent):
    """Agent that uses Chain-of-Thought reasoning.
    
    Breaks down problems into steps and thinks through each one
    before providing an answer.
    """
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        show_reasoning: bool = True
    ):
        super().__init__(llm, tools, system_prompt, max_iterations)
        self.show_reasoning = show_reasoning
        self.reasoning_steps: List[ReasoningStep] = []
    
    def run(self, task: str) -> str:
        """Run with chain-of-thought reasoning."""
        self.reasoning_steps = []
        
        # Build COT prompt
        system = self.system_prompt or "You are a reasoning assistant."
        system += """

When answering, use this format:
REASONING: [Step-by-step thinking process]
PLAN: [What steps you'll take]
EXECUTION: [Actions with tools if needed]
FINAL ANSWER: [Your final response]

Break complex problems into smaller steps."""
        
        messages = [
            Message.system(system),
            Message.user(task)
        ]
        
        # Get initial reasoning
        response = ensure_llm_response(self.llm.chat(messages))
        content = response.content
        
        # Parse and execute any actions
        final_answer = self._process_response(content, task, 0)
        
        return final_answer
    
    def _process_response(self, content: str, task: str, iteration: int) -> str:
        """Process response and execute actions."""
        from datetime import datetime
        
        # Extract reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=PLAN:|EXECUTION:|FINAL ANSWER:|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            self.reasoning_steps.append(ReasoningStep(
                step_type=ReasoningStepType.THOUGHT,
                content=reasoning_match.group(1).strip(),
                iteration=iteration,
                timestamp=datetime.now().isoformat()
            ))
        
        # Extract plan
        plan_match = re.search(
            r'PLAN:\s*(.+?)(?=EXECUTION:|FINAL ANSWER:|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        if plan_match:
            self.reasoning_steps.append(ReasoningStep(
                step_type=ReasoningStepType.PLAN,
                content=plan_match.group(1).strip(),
                iteration=iteration,
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for actions
        execution_match = re.search(
            r'EXECUTION:\s*(.+?)(?=FINAL ANSWER:|$)',
            content, re.DOTALL | re.IGNORECASE
        )
        
        if execution_match and self.tools and iteration < self.max_iterations:
            execution = execution_match.group(1).strip()
            
            # Parse actions
            action_pattern = r'ACTION:\s*(\w+)\s*PARAMETERS:\s*(\{[^}]*\})'
            actions = re.findall(action_pattern, execution, re.IGNORECASE)
            
            if actions:
                results = []
                for tool_name, params_str in actions:
                    try:
                        params = json.loads(params_str)
                    except:
                        params = {}
                    
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(**params)
                        results.append(f"{tool_name}: {result.content}")
                        
                        self.reasoning_steps.append(ReasoningStep(
                            step_type=ReasoningStepType.ACTION,
                            content=f"Executed {tool_name}",
                            iteration=iteration,
                            data={"tool": tool_name, "result": result.content},
                            timestamp=datetime.now().isoformat()
                        ))
                
                # Continue with results
                messages = [
                    Message.system(self.system_prompt or "Continue the task."),
                    Message.user(f"Task: {task}\n\nPrevious results:\n" + "\n".join(results) + "\n\nProvide the FINAL ANSWER.")
                ]
                
                response = ensure_llm_response(self.llm.chat(messages))
                return self._extract_final_answer(response.content)
        
        return self._extract_final_answer(content)
    
    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer from response."""
        final_match = re.search(
            r'FINAL ANSWER:\s*(.+)$',
            content, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            return final_match.group(1).strip()
        return content.strip()
    
    def get_reasoning_trace(self) -> List[Dict]:
        """Get full reasoning trace."""
        return [step.to_dict() for step in self.reasoning_steps]
    
    async def arun(self, task: str) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, task)


class ReflectiveAgent(BaseAgent):
    """Agent with self-reflection capabilities.
    
    Can reflect on its own outputs and improve them.
    """
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        reflection_rounds: int = 2
    ):
        super().__init__(llm, tools, system_prompt, max_iterations)
        self.reflection_rounds = reflection_rounds
        self.reflections: List[Dict] = []
    
    def run(self, task: str) -> str:
        """Run with self-reflection."""
        current_answer = None
        
        for round_num in range(self.reflection_rounds):
            # Generate or improve answer
            if current_answer is None:
                # First round - generate initial answer
                messages = [
                    Message.system(self.system_prompt or "You are a helpful assistant."),
                    Message.user(task)
                ]
                
                if self.tools:
                    # Use agent loop for tool-using tasks
                    current_answer = self._agent_loop(task)
                else:
                    response = ensure_llm_response(self.llm.chat(messages))
                    current_answer = response.content
            else:
                # Reflect and improve
                messages = [
                    Message.system("Reflect on your previous answer and improve it."),
                    Message.user(f"""Task: {task}

Your previous answer: {current_answer}

Reflect on this answer:
1. Is it accurate and complete?
2. Are there any errors or omissions?
3. How can it be improved?

Provide an improved answer.""")
                ]
                
                response = ensure_llm_response(self.llm.chat(messages))
                current_answer = response.content
                
                self.reflections.append({
                    "round": round_num,
                    "improvement": "Applied self-reflection"
                })
        
        return current_answer or "No answer generated."
    
    def _agent_loop(self, task: str) -> str:
        """Simple agent loop with tools."""
        messages = [
            Message.system(self.system_prompt or "Use tools to complete the task."),
            Message.user(task)
        ]
        
        for _ in range(self.max_iterations):
            response = ensure_llm_response(
                self.llm.chat(messages, tools=list(self.tools.values()) if self.tools else None)
            )
            
            if response.tool_calls:
                messages.append(Message.assistant(response.content))
                
                for tc in response.tool_calls:
                    if tc.name in self.tools:
                        result = self.tools[tc.name].execute(**tc.arguments)
                        messages.append(Message(
                            role=MessageRole.TOOL,
                            content=result.content,
                            metadata={"tool_name": tc.name}
                        ))
            else:
                return response.content
        
        return "Maximum iterations reached."
    
    def get_reflections(self) -> List[Dict]:
        """Get reflection history."""
        return self.reflections
    
    async def arun(self, task: str) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, task)


class SimpleAgent(BaseAgent):
    """Simple agent that uses tools via structured prompting."""
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        super().__init__(llm, tools, system_prompt, max_iterations)
        self.messages: List[Message] = []

    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """Compatibility planner so SimpleAgent is instantiable via BaseAgent."""
        query = kwargs.get("query") or kwargs.get("task") or kwargs.get("input") or ""
        output = self.run(query)
        return AgentFinish({"output": output}, log="SimpleAgent direct completion")

    async def aplan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        query = kwargs.get("query") or kwargs.get("task") or kwargs.get("input") or ""
        output = await self.arun(query)
        return AgentFinish({"output": output}, log="SimpleAgent direct completion")
    
    def run(self, task: str) -> str:
        """Run the agent on a task."""
        self.messages = []
        
        # Build system prompt with tool descriptions
        system = self.system_prompt or "You are a helpful AI assistant with access to tools."
        if self.tools:
            system += "\n\nAvailable tools:\n"
            for tool in self.tools.values():
                system += f"\n{tool.name}: {tool.description}"
                if tool.parameters:
                    params = ", ".join(
                        f"{k} ({v.type})" 
                        for k, v in tool.parameters.items()
                    )
                    system += f"\n  Parameters: {params}"
            
            system += """\n\nTo use a tool, respond with:
ACTION: tool_name
PARAMETERS: {"param1": "value1", "param2": "value2"}

The system will execute the tool and return the result."""
        
        self.messages.append(Message.system(system))
        self.messages.append(Message.user(task))
        
        # Main loop
        for iteration in range(self.max_iterations):
            # Get response from LLM
            if self.llm.supports_tools() and self.tools:
                # Use native tool calling if available
                response = ensure_llm_response(self.llm.chat(
                    messages=self.messages,
                    tools=list(self.tools.values())
                ))
                
                # Handle tool calls
                if response.tool_calls:
                    self.messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        metadata={"tool_calls": [tc.to_dict() for tc in response.tool_calls]}
                    ))
                    
                    # Execute each tool call
                    for tc in response.tool_calls:
                        if tc.name in self.tools:
                            tool = self.tools[tc.name]
                            result = tool.execute(**tc.arguments)
                            
                            self.messages.append(Message(
                                role=MessageRole.TOOL,
                                content=result.content,
                                metadata={"tool_name": tc.name, "success": result.success}
                            ))
                        else:
                            self.messages.append(Message(
                                role=MessageRole.TOOL,
                                content=f"Error: Tool '{tc.name}' not found",
                                metadata={"tool_name": tc.name, "success": False}
                            ))
                    
                    # Get next response after tool results
                    continue
                
                # No tool calls, return the response
                return response.content
            
            else:
                # Manual tool parsing
                response = ensure_llm_response(self.llm.chat(messages=self.messages))
                content = response.content
                
                # Check for action pattern
                action_match = re.search(
                    r'ACTION:\s*(\w+)\s*PARAMETERS:\s*(\{[^}]*\})',
                    content,
                    re.IGNORECASE | re.DOTALL
                )
                
                if action_match and self.tools:
                    tool_name = action_match.group(1)
                    params_str = action_match.group(2)
                    
                    try:
                        params = json.loads(params_str)
                    except json.JSONDecodeError:
                        self.messages.append(Message.assistant(content))
                        self.messages.append(Message.user(
                            "Invalid tool parameters format. Please use valid JSON."
                        ))
                        continue
                    
                    self.messages.append(Message.assistant(content))
                    
                    # Execute tool
                    if tool_name in self.tools:
                        tool = self.tools[tool_name]
                        result = tool.execute(**params)
                        self.messages.append(Message(
                            role=MessageRole.TOOL,
                            content=result.content,
                            metadata={"tool_name": tool_name, "success": result.success}
                        ))
                    else:
                        self.messages.append(Message(
                            role=MessageRole.TOOL,
                            content=f"Tool '{tool_name}' not found",
                            metadata={"tool_name": tool_name, "success": False}
                        ))
                    
                    continue
                
                # No action, return response
                return content
        
        return "Maximum iterations reached without completing the task."
    
    async def arun(self, task: str) -> str:
        """Async version of run."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, task)


class ReActAgent(BaseAgent):
    """ReAct agent with reasoning and acting."""
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        super().__init__(llm, tools, system_prompt, max_iterations)
        self.reasoning_steps: List[Dict[str, Any]] = []
        
    def plan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        """Plan the next action or finish (implementation for BaseAgent)."""
        prompt = self._build_react_prompt(kwargs.get("query", "Default Task"), include_history=False)
        
        # Build reasoning based on intermediate steps
        reasoning = []
        for i, step in enumerate(intermediate_steps):
            reasoning.append({
                "iteration": i + 1,
                "thought": step.action.log or "Thinking...",
                "action": step.action.tool,
                "input": step.action.tool_input,
                "observation": step.observation
            })
            
        self.reasoning_steps = reasoning
        if reasoning:
            prompt = self._build_react_prompt(kwargs.get("query", "Default Task"), include_history=True)
            
        messages = [
            Message.system(self.system_prompt or "You are a reasoning agent."),
            Message.user(prompt)
        ]
        
        response = ensure_llm_response(self.llm.chat(messages=messages))
        content = response.content.strip()
        
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', content, re.DOTALL | re.IGNORECASE)
        # Robust regex: matches "Action: tool_name", "Action: tool_name()", or just "tool_name()" if Action: is missing but Input: is present
        action_match = re.search(r'(?:Action:\s*)?(\w+)(?:\(\))?(?:\s*Input:\s*(\{[^}]*\}|[^\n]+))?', content, re.DOTALL | re.IGNORECASE)
        
        thought = thought_match.group(1).strip() if thought_match else ""
        if not thought and not action_match:
            thought = content
        
        if action_match:
            tool_name = action_match.group(1).strip()
            # If we matched just a random word as "Action", but there's no "Action:" prefix and no "Input:", it's probably just text.
            is_explicit = "action:" in content.lower() or "input:" in content.lower()
            if tool_name in self.tools and (is_explicit or "(" in content):
                tool_input = action_match.group(2).strip() if action_match.group(2) else "{}"
                return AgentAction(tool=tool_name, tool_input=tool_input, log=thought)
            # If it wasn't a valid/explicit tool call, we fall through to treat it as Final Answer or general content
            
        answer_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return AgentFinish(return_values={"output": answer_match.group(1).strip()}, log=thought)
            
        return AgentFinish(return_values={"output": content}, log=thought)
        
    async def aplan(
        self,
        intermediate_steps: List[AgentStep],
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.plan(intermediate_steps, callbacks, **kwargs))
    
    def run(self, task: str) -> str:
        """Run the agent using ReAct pattern."""
        self.reasoning_steps = []
        
        # Build ReAct prompt
        prompt = self._build_react_prompt(task)
        
        for iteration in range(self.max_iterations):
            # Get reasoning from LLM
            messages = [
                Message.system(self.system_prompt or "You are a reasoning agent."),
                Message.user(prompt)
            ]
            
            response = ensure_llm_response(self.llm.chat(messages=messages))
            content = response.content.strip()
            
            # Parse thought and action
            thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', content, re.DOTALL | re.IGNORECASE)
            action_match = re.search(r'(?:Action:\s*)?(\w+)(?:\(\))?(?:\s*Input:\s*(\{[^}]*\}|[^\n]+))?', content, re.DOTALL | re.IGNORECASE)
            
            thought = thought_match.group(1).strip() if thought_match else ""
            if not thought and not action_match:
                thought = content
            
            # Record step
            step = {
                "iteration": iteration + 1,
                "thought": thought,
            }
            
            if action_match:
                tool_name = action_match.group(1).strip()
                is_explicit = "action:" in content.lower() or "input:" in content.lower()
                
                if tool_name in self.tools and (is_explicit or "(" in content):
                    tool_input = action_match.group(2).strip() if action_match.group(2) else "{}"
                    
                    # Parse input
                    try:
                        if tool_input.startswith('{'):
                            tool_params = json.loads(tool_input)
                        else:
                            tool_params = {"query": tool_input}
                    except json.JSONDecodeError:
                        tool_params = {"input": tool_input}
                    
                    step["action"] = tool_name
                    step["input"] = tool_params
                    
                    # Execute tool
                    result = self.tools[tool_name].execute(**tool_params)
                    step["observation"] = result.content
                    step["success"] = result.success
                    
                    self.reasoning_steps.append(step)
                    
                    # Build next prompt with history
                    prompt = self._build_react_prompt(task, include_history=True)
                    continue
                else:
                    # Not a valid tool call, treat as final answer or continue
                    pass
            else:
                # Final answer
                step["final_answer"] = content
                self.reasoning_steps.append(step)
                
                # Extract answer
                answer_match = re.search(r'Final Answer:\s*(.+)', content, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    return answer_match.group(1).strip()
                
                return content
        
        return "Maximum iterations reached."
    
    def _build_react_prompt(self, task: str, include_history: bool = False) -> str:
        """Build the ReAct prompt."""
        lines = [
            "You are an AI assistant that can use tools to accomplish tasks.",
            "Think step by step and use the following format:",
            "",
            "Thought: [your reasoning about what to do]",
            "Action: [tool_name] Input: [tool parameters as JSON or text]",
            "",
            "Or provide the final answer:",
            "Thought: [your reasoning]",
            "Final Answer: [your answer]",
            "",
            "Available tools:"
        ]
        
        for name, tool in self.tools.items():
            lines.append(f"  - {name}: {tool.description}")
        
        if include_history and self.reasoning_steps:
            lines.extend(["", "Previous steps:"])
            for step in self.reasoning_steps:
                lines.append(f"\nIteration {step['iteration']}:")
                lines.append(f"Thought: {step['thought']}")
                if "action" in step:
                    lines.append(f"Action: {step['action']} Input: {json.dumps(step['input'])}")
                    lines.append(f"Observation: {step['observation']}")
        
        lines.extend([
            "",
            f"Task: {task}",
            "",
            "What is your next thought?"
        ])
        
        return "\n".join(lines)
    
    async def arun(self, task: str) -> str:
        """Async version of run."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, task)
    
    def get_steps(self) -> List[Dict[str, Any]]:
        """Get reasoning steps for inspection."""
        return self.reasoning_steps


class AgentBuilder:
    """Builder pattern for creating agents."""
    
    def __init__(self):
        self.llm: Optional[BaseProvider] = None
        self.tools: List[BaseTool] = []
        self.system_prompt: Optional[str] = None
        self.max_iterations: int = 10
        self.agent_type: str = "simple"
    
    def with_llm(self, llm: BaseProvider) -> "AgentBuilder":
        """Set the LLM provider."""
        self.llm = llm
        return self
    
    def with_tools(self, tools: List[BaseTool]) -> "AgentBuilder":
        """Set the tools."""
        self.tools = tools
        return self
    
    def add_tool(self, tool: BaseTool) -> "AgentBuilder":
        """Add a single tool."""
        self.tools.append(tool)
        return self
    
    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        """Set the system prompt."""
        self.system_prompt = prompt
        return self
    
    def with_max_iterations(self, n: int) -> "AgentBuilder":
        """Set max iterations."""
        self.max_iterations = n
        return self
    
    def as_react(self) -> "AgentBuilder":
        """Use ReAct agent type."""
        self.agent_type = "react"
        return self
    
    def as_simple(self) -> "AgentBuilder":
        """Use SimpleAgent type."""
        self.agent_type = "simple"
        return self
    
    def as_cot(self) -> "AgentBuilder":
        """Use ChainOfThoughtAgent type."""
        self.agent_type = "cot"
        return self
    
    def as_reflective(self) -> "AgentBuilder":
        """Use ReflectiveAgent type."""
        self.agent_type = "reflective"
        return self
    
    def build_simple(self) -> SimpleAgent:
        """Build a SimpleAgent."""
        if not self.llm:
            raise ValueError("LLM provider required")
        return SimpleAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations
        )
    
    def build_react(self) -> ReActAgent:
        """Build a ReActAgent."""
        if not self.llm:
            raise ValueError("LLM provider required")
        return ReActAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations
        )
    
    def build_cot(self) -> ChainOfThoughtAgent:
        """Build a ChainOfThoughtAgent."""
        if not self.llm:
            raise ValueError("LLM provider required")
        return ChainOfThoughtAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations
        )
    
    def build_reflective(self) -> ReflectiveAgent:
        """Build a ReflectiveAgent."""
        if not self.llm:
            raise ValueError("LLM provider required")
        return ReflectiveAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations
        )
    
    def build(self) -> BaseAgent:
        """Build the agent based on agent_type."""
        if not self.llm:
            raise ValueError("LLM provider required")
        
        from ..providers.base import ProviderType
        
        agent_type = self.agent_type
        # Default to ReAct (thinking) for local models, simple for cloud APIs
        if agent_type == "simple" and getattr(self.llm, "provider_type", None) == ProviderType.LOCAL:
            agent_type = "react"
        
        if agent_type == "react":
            return self.build_react()
        elif agent_type == "cot":
            return self.build_cot()
        elif agent_type == "reflective":
            return self.build_reflective()
        else:
            return self.build_simple()


class ToolCallingAgent(SimpleAgent):
    """Generic agent that uses native tool-calling features."""
    
    def __init__(
        self,
        llm: BaseProvider,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10
    ):
        super().__init__(llm, tools, system_prompt, max_iterations)


class StructuredChatAgent(SimpleAgent):
    """Uses structured JSON format to pass around messages, suitable for models that follow a strict JSON schema."""
    pass


def create_react_agent(
    llm: BaseProvider,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None
) -> ReActAgent:
    """Create a ReAct agent."""
    return ReActAgent(llm, tools, system_prompt)





def create_structured_chat_agent(
    llm: BaseProvider,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None
) -> StructuredChatAgent:
    """Create a structured chat agent."""
    return StructuredChatAgent(llm, tools, system_prompt)


def create_tool_calling_agent(
    llm: BaseProvider,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None
) -> ToolCallingAgent:
    """Create a tool calling agent."""
    return ToolCallingAgent(llm, tools, system_prompt)


def create_agent(
    llm: BaseProvider,
    tools: Optional[List[BaseTool]] = None,
    system_prompt: Optional[str] = None,
    react: Optional[bool] = None
) -> BaseAgent:
    """Quick agent creation."""
    from ..providers.base import ProviderType
    
    # Auto-enable thinking layer for local models by default
    if react is None:
        react = getattr(llm, "provider_type", None) == ProviderType.LOCAL
        
    if react:
        return ReActAgent(llm, tools, system_prompt)
    return SimpleAgent(llm, tools, system_prompt)


# ── Thinking Layer ─────────────────────────────────────────────────────────────

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
