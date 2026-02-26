"""
Native Tool Parser with proper Function Calling support.
Replaces regex-based JSON parsing with structured function calling.
"""

import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class ToolCallFormat(Enum):
    """Supported tool calling formats."""
    OPENAI = "openai"           # OpenAI function calling format
    ANTHROPIC = "anthropic"     # Anthropic tool use format
    GOOGLE = "google"           # Google function calling
    RAW_JSON = "raw_json"       # Raw JSON format (fallback)
    XML = "xml"                 # XML-based format


@dataclass
class ToolCall:
    """Structured tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ToolDefinition:
    """Definition of a tool for function calling."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    required: List[str]
    examples: Optional[List[Dict]] = None


class NativeToolParser:
    """
    Proper tool parser using structured function calling.
    Supports multiple LLM provider formats.
    """
    
    def __init__(self, format: ToolCallFormat = ToolCallFormat.OPENAI):
        self.format = format
    
    def create_tool_schema(self, tools: List[ToolDefinition]) -> Dict[str, Any]:
        """
        Create tool schema for LLM function calling.
        
        Returns schema in the format expected by the LLM provider.
        """
        if self.format == ToolCallFormat.OPENAI:
            return self._create_openai_schema(tools)
        elif self.format == ToolCallFormat.ANTHROPIC:
            return self._create_anthropic_schema(tools)
        elif self.format == ToolCallFormat.GOOGLE:
            return self._create_google_schema(tools)
        else:
            return self._create_generic_schema(tools)
    
    def _create_openai_schema(self, tools: List[ToolDefinition]) -> Dict[str, Any]:
        """Create OpenAI function calling schema."""
        return {
            "type": "function",
            "functions": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required,
                    },
                }
                for tool in tools
            ],
        }
    
    def _create_anthropic_schema(self, tools: List[ToolDefinition]) -> List[Dict]:
        """Create Anthropic tool use schema."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required,
                },
            }
            for tool in tools
        ]
    
    def _create_google_schema(self, tools: List[ToolDefinition]) -> Dict:
        """Create Google function calling schema."""
        return {
            "function_declarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required,
                    },
                }
                for tool in tools
            ]
        }
    
    def _create_generic_schema(self, tools: List[ToolDefinition]) -> Dict:
        """Create generic schema for other providers."""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "required": tool.required,
                }
                for tool in tools
            ]
        }
    
    def parse_tool_call(self, llm_response: Any) -> Optional[ToolCall]:
        """
        Parse tool call from LLM response.
        
        Args:
            llm_response: Response from LLM (can be dict, object, or string)
        
        Returns:
            ToolCall if found, None otherwise
        """
        # Handle different response types
        if isinstance(llm_response, dict):
            return self._parse_dict_response(llm_response)
        
        if hasattr(llm_response, 'tool_calls'):
            return self._parse_object_with_tool_calls(llm_response)
        
        if isinstance(llm_response, str):
            return self._parse_string_response(llm_response)
        
        # Try to extract from object attributes
        return self._parse_object_response(llm_response)
    
    def _parse_dict_response(self, response: Dict) -> Optional[ToolCall]:
        """Parse dict response (OpenAI/Anthropic format)."""
        # OpenAI format
        if "function_call" in response:
            fc = response["function_call"]
            try:
                args = json.loads(fc.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            
            return ToolCall(
                tool_name=fc.get("name", ""),
                parameters=args,
                call_id=response.get("id"),
            )
        
        # Anthropic format
        if "tool_use" in response:
            tu = response["tool_use"]
            return ToolCall(
                tool_name=tu.get("name", ""),
                parameters=tu.get("input", {}),
                call_id=tu.get("id"),
            )
        
        # Direct tool call format
        if "tool_name" in response or "name" in response:
            return ToolCall(
                tool_name=response.get("tool_name") or response.get("name", ""),
                parameters=response.get("parameters") or response.get("arguments", {}),
                call_id=response.get("id"),
            )
        
        return None
    
    def _parse_object_with_tool_calls(self, response) -> Optional[ToolCall]:
        """Parse object with tool_calls attribute."""
        tool_calls = response.tool_calls
        
        if not tool_calls:
            return None
        
        # Get first tool call
        tc = tool_calls[0]
        
        if hasattr(tc, 'function'):
            # OpenAI format
            func = tc.function
            try:
                args = json.loads(func.arguments) if hasattr(func, 'arguments') else {}
            except:
                args = {}
            
            return ToolCall(
                tool_name=func.name if hasattr(func, 'name') else "",
                parameters=args,
                call_id=tc.id if hasattr(tc, 'id') else None,
            )
        
        return None
    
    def _parse_object_response(self, response) -> Optional[ToolCall]:
        """Parse generic object response."""
        try:
            # Try to convert to dict
            response_dict = vars(response)
            return self._parse_dict_response(response_dict)
        except:
            pass
        
        return None
    
    def _parse_string_response(self, response: str) -> Optional[ToolCall]:
        """
        Parse string response as last resort.
        Uses proper JSON parsing, not regex.
        """
        response = response.strip()
        
        # Try to find JSON block
        if "```json" in response:
            # Extract from markdown code block
            try:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                data = json.loads(json_str)
                return self._create_tool_call_from_dict(data)
            except:
                pass
        
        if "```" in response:
            # Try generic code block
            try:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                data = json.loads(json_str)
                return self._create_tool_call_from_dict(data)
            except:
                pass
        
        # Try to parse as JSON directly
        try:
            data = json.loads(response)
            return self._create_tool_call_from_dict(data)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        try:
            # Find first { and last }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = response[start:end+1]
                data = json.loads(json_str)
                return self._create_tool_call_from_dict(data)
        except:
            pass
        
        return None
    
    def _create_tool_call_from_dict(self, data: Dict) -> Optional[ToolCall]:
        """Create ToolCall from parsed dictionary."""
        # Handle various formats
        tool_name = (
            data.get("tool") or
            data.get("tool_name") or
            data.get("name") or
            data.get("function")
        )
        
        if not tool_name:
            return None
        
        parameters = (
            data.get("parameters") or
            data.get("args") or
            data.get("arguments") or
            data.get("input") or
            {k: v for k, v in data.items() if k not in ['tool', 'tool_name', 'name', 'function']}
        )
        
        return ToolCall(
            tool_name=tool_name,
            parameters=parameters if isinstance(parameters, dict) else {},
            call_id=data.get("id") or data.get("call_id"),
        )


class ToolCallingLLM:
    """
    Wrapper for LLMs to support native tool calling.
    """
    
    def __init__(self, llm, tools: List[ToolDefinition], parser: Optional[NativeToolParser] = None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.parser = parser or NativeToolParser()
        self.tool_implementations: Dict[str, Callable] = {}
    
    def register_tool(self, name: str, implementation: Callable) -> None:
        """Register tool implementation."""
        self.tool_implementations[name] = implementation
    
    def chat_with_tools(self, message: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Chat with automatic tool calling.
        
        Returns:
            Dict with 'response', 'tool_calls', and 'results'
        """
        # Create tool schema
        tool_schema = self.parser.create_tool_schema(list(self.tools.values()))
        
        # Build prompt with tools
        prompt = self._build_tool_prompt(message, tool_schema, history)
        
        # Get LLM response
        response = self.llm.chat(prompt)
        
        # Parse tool calls
        tool_call = self.parser.parse_tool_call(response)
        
        if not tool_call or tool_call.tool_name not in self.tool_implementations:
            # No tool call, return direct response
            return {
                "response": response.text if hasattr(response, 'text') else str(response),
                "tool_calls": [],
                "results": [],
            }
        
        # Execute tool
        result = self._execute_tool(tool_call)
        
        # Get final response with tool result
        final_prompt = self._build_result_prompt(message, tool_call, result, history)
        final_response = self.llm.chat(final_prompt)
        
        return {
            "response": final_response.text if hasattr(final_response, 'text') else str(final_response),
            "tool_calls": [tool_call],
            "results": [result],
        }
    
    def _build_tool_prompt(self, message: str, tool_schema: Dict, history: Optional[List]) -> str:
        """Build prompt with tool definitions."""
        prompt_parts = [
            "You have access to the following tools:",
            "",
            json.dumps(tool_schema, indent=2),
            "",
            "To use a tool, respond with a JSON object:",
            '{"tool": "tool_name", "parameters": {"param1": "value1"}}',
            "",
        ]
        
        if history:
            prompt_parts.append("Conversation history:")
            for msg in history[-5:]:
                prompt_parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            prompt_parts.append("")
        
        prompt_parts.extend([
            f"User: {message}",
            "",
            "Respond with tool JSON or direct answer:",
        ])
        
        return "\n".join(prompt_parts)
    
    def _execute_tool(self, tool_call: ToolCall) -> Any:
        """Execute tool call."""
        tool_name = tool_call.tool_name
        parameters = tool_call.parameters
        
        if tool_name not in self.tool_implementations:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            implementation = self.tool_implementations[tool_name]
            return implementation(**parameters)
        except Exception as e:
            return f"Error executing tool: {str(e)}"
    
    def _build_result_prompt(self, original_message: str, tool_call: ToolCall, result: Any, history: Optional[List]) -> str:
        """Build prompt with tool result."""
        return f"""User asked: {original_message}

You used tool: {tool_call.tool_name} with parameters: {json.dumps(tool_call.parameters)}

Tool result: {result}

Provide your response to the user:
"""


class ToolResultFormatter:
    """Format tool results for LLM consumption."""
    
    @staticmethod
    def format_result(result: Any, max_length: int = 1000) -> str:
        """Format result as string."""
        if isinstance(result, str):
            return result[:max_length]
        
        if isinstance(result, (list, dict)):
            result_str = json.dumps(result, indent=2, ensure_ascii=False)
            if len(result_str) > max_length:
                return result_str[:max_length-3] + "..."
            return result_str
        
        return str(result)[:max_length]
    
    @staticmethod
    def format_error(error: str) -> str:
        """Format error message."""
        return f"[Tool Error: {error}]"


# Convenience functions
def create_tool_definition(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    required: Optional[List[str]] = None,
) -> ToolDefinition:
    """Quick tool definition creation."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters=parameters,
        required=required or [],
    )


def parse_tool_from_response(response: Any, format: ToolCallFormat = ToolCallFormat.OPENAI) -> Optional[ToolCall]:
    """Quick tool parsing."""
    parser = NativeToolParser(format)
    return parser.parse_tool_call(response)
