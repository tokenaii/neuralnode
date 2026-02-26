"""Tests for NeuralNode core functionality."""

import pytest
from unittest.mock import MagicMock, patch

from neuralnode.core.types import (
    Message,
    MessageRole,
    ProviderType,
    TokenUsage,
    ToolDefinition,
)
from neuralnode.memory.conversation import ConversationMemory


class TestMessage:
    def test_message_creation(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_to_dict(self):
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            name="test",
        )
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Hello"
        assert data["name"] == "test"


class TestTokenUsage:
    def test_usage_creation(self):
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_cost_calculation(self):
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        # Test with known model rates
        cost = usage.cost("gpt-4")
        assert cost > 0


class TestToolDefinition:
    def test_tool_to_openai_format(self):
        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "param1": {"type": "string"},
            },
            required=["param1"],
        )
        
        openai_format = tool.to_openai_format()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert "param1" in openai_format["function"]["parameters"]["properties"]


class TestConversationMemory:
    def test_add_message(self):
        memory = ConversationMemory()
        msg = Message(role=MessageRole.USER, content="Hello")
        memory.add(msg)
        assert memory.message_count == 1

    def test_get_messages(self):
        memory = ConversationMemory()
        memory.add(Message(role=MessageRole.USER, content="Hello"))
        memory.add(Message(role=MessageRole.ASSISTANT, content="Hi"))
        
        messages = memory.get_messages()
        assert len(messages) == 2

    def test_memory_limit(self):
        memory = ConversationMemory(max_messages=3)
        
        for i in range(5):
            memory.add(Message(role=MessageRole.USER, content=f"Message {i}"))
        
        assert memory.message_count <= 3

    def test_clear_memory(self):
        memory = ConversationMemory()
        memory.add(Message(role=MessageRole.USER, content="Hello"))
        memory.clear()
        assert memory.message_count == 0

    def test_metadata(self):
        memory = ConversationMemory()
        memory.set_metadata("key", "value")
        assert memory.get_metadata("key") == "value"
        assert memory.get_metadata("missing", "default") == "default"


class TestProviderType:
    def test_provider_enum(self):
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        assert ProviderType.AUTO.value == "auto"
