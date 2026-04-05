"""Test core components: Message, LLMResponse, ToolResult, BaseProvider, BaseTool, BaseAgent."""

import pytest
from unittest.mock import MagicMock, patch
import neuralnode as nn
from neuralnode.core import Message, MessageRole, LLMResponse, ToolResult, ToolCall


class TestMessage:
    """Test Message dataclass and factory methods."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}
    
    def test_message_with_metadata(self):
        """Test message with metadata."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hi",
            metadata={"timestamp": "2024-01-01", "token_count": 10}
        )
        assert msg.metadata["token_count"] == 10
    
    def test_message_factory_methods(self):
        """Test factory methods: system, user, assistant, tool."""
        system_msg = Message.system("System prompt")
        assert system_msg.role == MessageRole.SYSTEM
        
        user_msg = Message.user("User query")
        assert user_msg.role == MessageRole.USER
        
        assistant_msg = Message.assistant("Assistant response")
        assert assistant_msg.role == MessageRole.ASSISTANT
        
        tool_msg = Message.tool("Tool result", tool_name="search")
        assert tool_msg.role == MessageRole.TOOL
        assert tool_msg.metadata["tool_name"] == "search"
    
    def test_message_to_dict(self):
        """Test message serialization."""
        msg = Message.user("Test")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Test"
    
    def test_message_from_dict(self):
        """Test message deserialization."""
        data = {"role": "assistant", "content": "Hello", "metadata": {}}
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hello"


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_response_creation(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Test response",
            model="claude-3-haiku",
            provider="anthropic"
        )
        assert response.content == "Test response"
        assert response.model == "claude-3-haiku"
        assert response.provider == "anthropic"
    
    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tool_call = ToolCall(name="search", arguments={"query": "test"}, call_id="123")
        response = LLMResponse(
            content="",
            model="claude-3-haiku",
            provider="anthropic",
            tool_calls=[tool_call]
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "search"
    
    def test_response_with_usage(self):
        """Test response with token usage."""
        response = LLMResponse(
            content="Test",
            model="claude-3-haiku",
            provider="anthropic",
            usage={"prompt_tokens": 10, "completion_tokens": 20}
        )
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
    
    def test_response_to_dict(self):
        """Test response serialization."""
        response = LLMResponse(content="Test", model="claude-3-haiku", provider="anthropic")
        data = response.to_dict()
        assert data["content"] == "Test"
        assert data["provider"] == "anthropic"


class TestToolResult:
    """Test ToolResult dataclass."""
    
    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult.ok(content="Success", data={"key": "value"})
        assert result.success is True
        assert result.content == "Success"
        assert result.data["key"] == "value"
    
    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult.error(message="Failed", error_code=500)
        assert result.success is False
        assert "Failed" in result.content
    
    def test_result_with_output(self):
        """Test result with output data."""
        result = ToolResult.ok(
            content="Done",
            stdout="output",
            stderr="",
            return_code=0
        )
        assert result.output["stdout"] == "output"
        assert result.output["return_code"] == 0


class TestBaseProvider:
    """Test BaseProvider abstract class."""
    
    def test_provider_initialization(self, mock_provider):
        """Test provider can be initialized."""
        assert mock_provider.model is not None
    
    def test_provider_chat(self, mock_provider):
        """Test provider chat method."""
        messages = [Message.user("Hello")]
        response = mock_provider.chat(messages)
        assert isinstance(response, LLMResponse)
        mock_provider.chat.assert_called_once()
    
    def test_provider_embed(self, mock_provider):
        """Test provider embed method."""
        embeddings = mock_provider.embed(["test"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
    
    def test_provider_capabilities(self, mock_provider):
        """Test provider capabilities property."""
        caps = mock_provider.capabilities
        assert caps["tools"] is True
        assert caps["streaming"] is True
        assert caps["embeddings"] is True


class TestBaseTool:
    """Test BaseTool abstract class."""
    
    def test_tool_initialization(self, mock_tool):
        """Test tool initialization."""
        assert mock_tool.name == "mock_tool"
        assert mock_tool.description is not None
    
    def test_tool_execute(self, mock_tool):
        """Test tool execution."""
        result = mock_tool.execute(param="value")
        assert isinstance(result, ToolResult)
        mock_tool.execute.assert_called_once()
    
    def test_tool_schema(self, mock_tool):
        """Test tool schema generation."""
        mock_tool.get_schema.return_value = {
            "type": "function",
            "function": {
                "name": "mock_tool",
                "description": "A mock tool"
            }
        }
        schema = mock_tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "mock_tool"


class TestBaseAgent:
    """Test BaseAgent abstract class."""
    
    def test_agent_initialization(self, mock_provider, mock_tool):
        """Test agent initialization."""
        agent = nn.SimpleAgent(llm=mock_provider, tools=[mock_tool])
        assert agent.llm == mock_provider
        assert len(agent.tools) == 1
    
    def test_agent_run(self, mock_provider, mock_tool):
        """Test agent run method."""
        mock_provider.chat.return_value = LLMResponse(
            content="Task complete",
            model="mock",
            provider="mock"
        )
        agent = nn.SimpleAgent(llm=mock_provider, tools=[mock_tool])
        result = agent.run("Test task")
        assert isinstance(result, str)
    
    def test_agent_tool_registration(self, mock_provider):
        """Test agent tool registration."""
        agent = nn.SimpleAgent(llm=mock_provider)
        tool = MagicMock()
        tool.name = "new_tool"
        agent.register_tool(tool)
        assert "new_tool" in agent.tools


class TestToolCall:
    """Test ToolCall dataclass."""
    
    def test_tool_call_creation(self):
        """Test tool call creation."""
        call = ToolCall(name="search", arguments={"query": "test"}, call_id="123")
        assert call.name == "search"
        assert call.arguments["query"] == "test"
        assert call.call_id == "123"
    
    def test_tool_call_to_dict(self):
        """Test tool call serialization."""
        call = ToolCall(name="search", arguments={}, call_id="123")
        data = call.to_dict()
        assert data["name"] == "search"
        assert data["call_id"] == "123"
