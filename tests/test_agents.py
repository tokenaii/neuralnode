"""Tests for Agent functionality."""

import pytest
from unittest.mock import MagicMock

from neuralnode.agents.agent import Agent, Tool, create_tool
from neuralnode.core.types import LLMResponse, Message, MessageRole


class TestTool:
    def test_tool_creation(self):
        def handler(x: int) -> int:
            return x * 2
        
        tool = Tool(
            name="double",
            description="Double a number",
            parameters={"x": {"type": "integer"}},
            handler=handler,
            required=["x"],
        )
        
        assert tool.name == "double"
        assert tool.description == "Double a number"
        assert tool.required == ["x"]

    def test_tool_execute(self):
        def handler(x: int) -> int:
            return x * 2
        
        tool = Tool(
            name="double",
            description="Double a number",
            parameters={"x": {"type": "integer"}},
            handler=handler,
        )
        
        result = tool.execute(x=5)
        assert result == 10


class TestCreateToolDecorator:
    def test_decorator(self):
        @create_tool(
            name="add",
            description="Add two numbers",
            parameters={
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            required=["a", "b"],
        )
        def add(a: int, b: int) -> int:
            return a + b
        
        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.execute(a=2, b=3) == 5


class TestAgent:
    def test_agent_creation(self):
        mock_llm = MagicMock()
        agent = Agent(llm=mock_llm)
        assert agent.llm is mock_llm
        assert agent.max_iterations == 10

    def test_agent_with_tools(self):
        mock_llm = MagicMock()
        
        tool = Tool(
            name="test",
            description="Test tool",
            parameters={},
            handler=lambda: "result",
        )
        
        agent = Agent(llm=mock_llm, tools=[tool])
        assert "test" in agent.tools

    def test_add_tool(self):
        mock_llm = MagicMock()
        agent = Agent(llm=mock_llm)
        
        tool = Tool(
            name="new_tool",
            description="A new tool",
            parameters={},
            handler=lambda: "done",
        )
        
        agent.add_tool(tool)
        assert "new_tool" in agent.tools

    def test_remove_tool(self):
        mock_llm = MagicMock()
        
        tool = Tool(
            name="removable",
            description="Tool to remove",
            parameters={},
            handler=lambda: "done",
        )
        
        agent = Agent(llm=mock_llm, tools=[tool])
        assert "removable" in agent.tools
        
        agent.remove_tool("removable")
        assert "removable" not in agent.tools

    def test_format_tool_descriptions(self):
        mock_llm = MagicMock()
        
        tool = Tool(
            name="calculator",
            description="Do math",
            parameters={"expr": {"type": "string"}},
            handler=lambda expr: eval(expr),
        )
        
        agent = Agent(llm=mock_llm, tools=[tool])
        desc = agent._format_tool_descriptions()
        
        assert "calculator" in desc
        assert "Do math" in desc


class TestToolCallExtraction:
    def test_extract_json_tool_call(self):
        mock_llm = MagicMock()
        agent = Agent(llm=mock_llm)
        
        response = '''```json
        {"tool": "search", "parameters": {"query": "test"}}
        ```'''
        
        calls = agent._extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "search"

    def test_extract_no_tool_calls(self):
        mock_llm = MagicMock()
        agent = Agent(llm=mock_llm)
        
        response = "This is just a regular response."
        calls = agent._extract_tool_calls(response)
        assert len(calls) == 0
