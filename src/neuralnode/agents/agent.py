"""AI Agent system with tool integration."""

import json
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.ai import NeuralNode
from ..core.types import LLMResponse, Message, MessageRole, ToolDefinition


class Tool:
    """A tool that can be used by an agent."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
        required: Optional[List[str]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.required = required or []
        
        # Create ToolDefinition for the LLM
        self.definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            required=self.required,
            handler=handler,
        )

    def execute(self, **params) -> Any:
        """Execute the tool with given parameters."""
        return self.handler(**params)


class Agent:
    """
    An AI agent that can use tools and make decisions.
    """

    def __init__(
        self,
        llm: NeuralNode,
        tools: Optional[List[Union[Tool, ToolDefinition]]] = None,
        system: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """
        Initialize an Agent.
        
        Args:
            llm: NeuralNode instance for LLM calls
            tools: List of tools available to the agent
            system: System prompt for the agent
            max_iterations: Maximum number of tool calls before stopping
            verbose: Whether to print agent thinking process
        """
        self.llm = llm
        self.tools: Dict[str, Tool] = {}
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.system = system or self._default_system_prompt()
        
        # Register tools
        if tools:
            for tool in tools:
                if isinstance(tool, ToolDefinition):
                    # Convert ToolDefinition to Tool if handler exists
                    if tool.handler:
                        tool = Tool(
                            name=tool.name,
                            description=tool.description,
                            parameters=tool.parameters,
                            handler=tool.handler,
                            required=tool.required,
                        )
                    else:
                        continue
                self.tools[tool.name] = tool

    def _default_system_prompt(self) -> str:
        """Generate default system prompt with available tools."""
        return """You are an AI assistant that can use tools to accomplish tasks.
Available tools will be provided in your context.
Think step by step and use tools when necessary."""

    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the prompt."""
        if not self.tools:
            return "No tools available."
        
        descriptions = ["Available tools:"]
        for name, tool in self.tools.items():
            desc = f"\n- {name}: {tool.description}"
            if tool.parameters:
                desc += f"\n  Parameters: {json.dumps(tool.parameters, indent=2)}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Run the agent on a task.
        
        Args:
            task: The task description
            context: Additional context for the task
            
        Returns:
            Final response from the agent
        """
        iterations = 0
        messages: List[Message] = []
        
        # Initial system prompt with tools
        full_system = f"{self.system}\n\n{self._format_tool_descriptions()}"
        messages.append(Message(role=MessageRole.SYSTEM, content=full_system))
        
        # Add context if provided
        if context:
            context_str = json.dumps(context, indent=2)
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=f"Additional context: {context_str}"
            ))
        
        # Add user task
        messages.append(Message(role=MessageRole.USER, content=task))
        
        while iterations < self.max_iterations:
            if self.verbose:
                print(f"\n[Agent] Iteration {iterations + 1}")
            
            # Get LLM response
            tool_definitions = [t.definition for t in self.tools.values()]
            response = self.llm.provider.chat(
                messages,
                tools=tool_definitions if self.llm.supports_tools else None,
            )
            
            if isinstance(response, LLMResponse):
                response_text = response.text
            else:
                # Handle streaming - collect all chunks
                response_text = "".join(chunk.text for chunk in response)
            
            if self.verbose:
                print(f"[Agent] Response: {response_text[:200]}...")
            
            # Check if tool call is needed
            tool_calls = self._extract_tool_calls(response_text)
            
            if not tool_calls:
                # No tool calls, return the response
                return response_text
            
            # Execute tool calls
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_params = tool_call.get("parameters", {})
                
                if tool_name in self.tools:
                    if self.verbose:
                        print(f"[Agent] Calling tool: {tool_name}({tool_params})")
                    
                    try:
                        result = self.tools[tool_name].execute(**tool_params)
                        result_str = str(result) if not isinstance(result, str) else result
                    except Exception as e:
                        result_str = f"Error: {e}"
                    
                    # Add tool result to conversation
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=f"I need to use the {tool_name} tool.",
                    ))
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=f"Tool {tool_name} result: {result_str}",
                    ))
                else:
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content=f"I tried to use tool {tool_name} but it's not available.",
                    ))
            
            iterations += 1
        
        # Max iterations reached
        return "Maximum iterations reached. Task may be incomplete."

    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response.
        
        Supports multiple formats:
        - JSON format: {"tool": "name", "parameters": {...}}
        - XML-like format: <tool name="...">...</tool>
        """
        tool_calls = []
        
        # Try JSON extraction
        try:
            # Look for JSON code blocks
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            for match in matches:
                try:
                    data = json.loads(match)
                    if "tool" in data or "name" in data:
                        tool_calls.append({
                            "name": data.get("tool") or data.get("name"),
                            "parameters": data.get("parameters", data.get("args", {})),
                        })
                except json.JSONDecodeError:
                    pass
            
            # Try direct JSON parsing
            if not tool_calls:
                data = json.loads(response)
                if isinstance(data, dict) and ("tool" in data or "name" in data):
                    tool_calls.append({
                        "name": data.get("tool") or data.get("name"),
                        "parameters": data.get("parameters", data.get("args", {})),
                    })
        except (json.JSONDecodeError, ValueError):
            pass
        
        return tool_calls

    async def arun(self, task: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Async version of run."""
        # For now, run synchronously in async context
        return self.run(task, context)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.tools[tool.name] = tool

    def remove_tool(self, name: str) -> None:
        """Remove a tool from the agent."""
        if name in self.tools:
            del self.tools[name]


def create_tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to create a tool from a function.
    
    Example:
        @create_tool("search", "Search for information", {"query": {"type": "string"}})
        def search(query: str) -> str:
            return f"Results for: {query}"
    """
    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=func,
            required=required or [],
        )
    return decorator
