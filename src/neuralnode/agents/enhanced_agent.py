"""Enhanced Agent with native tool calling and improved parsing."""

import json
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.types import LLMResponse, Message, MessageRole


class ToolCallingFormat(Enum):
    """Supported tool calling formats."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LLAMA = "llama"  # XML format
    JSON = "json"    # Generic JSON
    AUTO = "auto"


@dataclass
class ToolCall:
    """A parsed tool call."""
    name: str
    arguments: Dict[str, Any]
    id: Optional[str] = None
    raw: Optional[str] = None


class NativeToolParser:
    """Parse tool calls from different LLM formats."""
    
    @staticmethod
    def parse_openai_format(content: str) -> List[ToolCall]:
        """Parse OpenAI function calling format."""
        tool_calls = []
        
        # Try to find JSON tool calls
        patterns = [
            # Standard OpenAI format
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
            # Alternative format
            r'"function":\s*\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for name, args_str in matches:
                try:
                    args = json.loads(args_str)
                    tool_calls.append(ToolCall(name=name, arguments=args))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    @staticmethod
    def parse_anthropic_format(content: str) -> List[ToolCall]:
        """Parse Anthropic tool use format."""
        tool_calls = []
        
        # Anthropic uses XML-like tags
        pattern = r'<tool>([^<]+)</tool>\s*<tool_input>([^<]*)</tool_input>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for name, input_str in matches:
            try:
                args = json.loads(input_str.strip()) if input_str.strip() else {}
                tool_calls.append(ToolCall(name=name.strip(), arguments=args))
            except json.JSONDecodeError:
                tool_calls.append(ToolCall(name=name.strip(), arguments={"input": input_str.strip()}))
        
        return tool_calls
    
    @staticmethod
    def parse_llama_format(content: str) -> List[ToolCall]:
        """Parse Llama XML tool format."""
        tool_calls = []
        
        # Llama uses <function_calls> tags
        pattern = r'<function_calls>\s*<invoke>\s*<tool_name>([^<]+)</tool_name>(.*?)<parameters>(.*?)</parameters>\s*</invoke>\s*</function_calls>'
        matches = re.findall(pattern, content, re.DOTALL)
        
        for name, _, params_str in matches:
            try:
                # Parse XML parameters
                args = {}
                param_pattern = r'<([^>]+)>([^<]*)</\1>'
                for key, value in re.findall(param_pattern, params_str):
                    args[key] = value
                tool_calls.append(ToolCall(name=name.strip(), arguments=args))
            except Exception:
                continue
        
        return tool_calls
    
    @staticmethod
    def parse_json_format(content: str) -> List[ToolCall]:
        """Parse generic JSON tool calls."""
        tool_calls = []
        
        # Try to find any JSON objects with 'tool' or 'action' or 'function'
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                if 'tool' in data or 'action' in data or 'function' in data:
                    name = data.get('tool') or data.get('action') or data.get('function') or data.get('name', '')
                    args = {k: v for k, v in data.items() if k not in ['tool', 'action', 'function', 'name']}
                    tool_calls.append(ToolCall(name=name, arguments=args))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        name = item.get('tool') or item.get('action') or item.get('function') or item.get('name', '')
                        if name:
                            args = {k: v for k, v in item.items() if k not in ['tool', 'action', 'function', 'name']}
                            tool_calls.append(ToolCall(name=name, arguments=args))
        except json.JSONDecodeError:
            pass
        
        return tool_calls
    
    @classmethod
    def parse_auto(cls, content: str) -> List[ToolCall]:
        """Auto-detect format and parse."""
        # Try formats in order of likelihood
        parsers = [
            cls.parse_openai_format,
            cls.parse_llama_format,
            cls.parse_anthropic_format,
            cls.parse_json_format,
        ]
        
        for parser in parsers:
            calls = parser(content)
            if calls:
                return calls
        
        return []


class EnhancedAgent:
    """Enhanced agent with native tool calling support."""
    
    def __init__(
        self,
        llm,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = False,
        tool_format: ToolCallingFormat = ToolCallingFormat.AUTO,
    ):
        self.llm = llm
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.system_prompt = system_prompt or "You are a helpful AI assistant with access to tools."
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.tool_format = tool_format
        self.history: List[Message] = []
        self.parser = NativeToolParser()
    
    def _generate_tool_prompt(self) -> str:
        """Generate tool instructions for the LLM."""
        if not self.tools:
            return ""
        
        tool_descriptions = []
        for name, tool in self.tools.items():
            desc = f"\n- {name}: {tool.description}"
            if hasattr(tool, 'parameters') and tool.parameters:
                params = json.dumps(tool.parameters, indent=2)
                desc += f"\n  Parameters: {params}"
            tool_descriptions.append(desc)
        
        return f"""
You have access to the following tools:
{''.join(tool_descriptions)}

To use a tool, respond with a JSON object in this format:
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}

If you need to use multiple tools, return a list of tool calls.
"""
    
    def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run the agent with improved tool calling."""
        messages = []
        
        # System prompt
        full_system = self.system_prompt + self._generate_tool_prompt()
        if context:
            context_str = json.dumps(context, indent=2)
            full_system += f"\n\nContext: {context_str}"
        
        messages.append(Message(role=MessageRole.SYSTEM, content=full_system))
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        iteration = 0
        while iteration < self.max_iterations:
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")
            
            # Get LLM response
            response = self.llm.chat(messages)
            content = response.text if hasattr(response, 'text') else str(response)
            
            if self.verbose:
                print(f"Response: {content[:200]}...")
            
            # Parse tool calls
            tool_calls = self.parser.parse_auto(content)
            
            if not tool_calls:
                # No tool calls, return the response
                return content
            
            # Execute tools
            results = []
            for call in tool_calls:
                if call.name in self.tools:
                    tool = self.tools[call.name]
                    try:
                        if self.verbose:
                            print(f"Executing: {call.name}({call.arguments})")
                        
                        result = tool.execute(**call.arguments)
                        results.append({
                            "tool": call.name,
                            "result": str(result),
                            "success": True,
                        })
                    except Exception as e:
                        results.append({
                            "tool": call.name,
                            "result": str(e),
                            "success": False,
                        })
                else:
                    results.append({
                        "tool": call.name,
                        "result": f"Tool '{call.name}' not found",
                        "success": False,
                    })
            
            # Add to messages
            messages.append(Message(role=MessageRole.ASSISTANT, content=content))
            messages.append(Message(
                role=MessageRole.TOOL,
                content=json.dumps(results),
            ))
            
            iteration += 1
        
        return "Maximum iterations reached."
    
    async def arun(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Async version of run."""
        # For now, delegate to sync version
        # In production, this should be fully async
        return self.run(prompt, context)


class MultiStepAgent(EnhancedAgent):
    """Agent that can plan and execute multi-step tasks."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan: Optional[List[str]] = None
    
    def _generate_plan(self, task: str) -> List[str]:
        """Generate a plan for the task."""
        planning_prompt = f"""
Break down the following task into steps. Return as a JSON list of steps.
Task: {task}
Available tools: {list(self.tools.keys())}

Respond with: ["step 1", "step 2", "step 3"]"""
        
        response = self.llm.chat(planning_prompt)
        content = response.text if hasattr(response, 'text') else str(response)
        
        try:
            # Extract JSON list
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        
        # Fallback: split by newlines
        return [line.strip() for line in content.split('\n') if line.strip()]
    
    def run(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Run with planning."""
        # Generate plan
        if self.verbose:
            print("Generating plan...")
        
        self.plan = self._generate_plan(prompt)
        
        if self.verbose:
            print(f"Plan: {self.plan}")
        
        # Execute plan
        results = []
        for i, step in enumerate(self.plan):
            if self.verbose:
                print(f"\nExecuting step {i+1}: {step}")
            
            step_result = super().run(f"Execute this step: {step}", context)
            results.append(f"Step {i+1}: {step}\nResult: {step_result}")
        
        # Final synthesis
        synthesis_prompt = f"""
Synthesize the following results into a final answer:
Original task: {prompt}

{chr(10).join(results)}

Provide a clear, concise final answer."""
        
        final_response = self.llm.chat(synthesis_prompt)
        return final_response.text if hasattr(final_response, 'text') else str(final_response)
