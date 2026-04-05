"""Test memory system."""

import pytest
import json
from pathlib import Path

import neuralnode as nn
from neuralnode.core import Message, MessageRole


class TestConversationMemory:
    """Test ConversationMemory."""
    
    def test_init(self):
        """Test initialization."""
        memory = nn.ConversationMemory()
        assert len(memory.messages) == 0
        assert memory.max_messages == 100
    
    def test_add_message(self):
        """Test adding messages."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("Hello"))
        
        assert len(memory.messages) == 1
        assert memory.messages[0].role == MessageRole.USER
    
    def test_add_multiple_messages(self):
        """Test adding multiple messages."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("Hello"))
        memory.add(Message.assistant("Hi there!"))
        memory.add(Message.user("How are you?"))
        
        assert len(memory.messages) == 3
    
    def test_get_all_messages(self):
        """Test retrieving all messages."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("Test"))
        
        messages = memory.get()
        assert len(messages) == 1
        assert messages[0].content == "Test"
    
    def test_get_recent_messages(self):
        """Test retrieving recent messages only."""
        memory = nn.ConversationMemory()
        for i in range(10):
            memory.add(Message.user(f"Message {i}"))
        
        recent = memory.get(n=5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 9"
    
    def test_clear(self):
        """Test clearing memory."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("Test"))
        memory.clear()
        
        assert len(memory.messages) == 0
    
    def test_max_messages_limit(self):
        """Test that max messages limit is enforced."""
        memory = nn.ConversationMemory(max_messages=5)
        for i in range(10):
            memory.add(Message.user(f"Message {i}"))
        
        assert len(memory.messages) == 5
    
    def test_to_dict(self):
        """Test serialization."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("Test"))
        
        data = memory.to_dict()
        assert "messages" in data
        assert len(data["messages"]) == 1
    
    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello", "metadata": {}}
            ],
            "max_messages": 50
        }
        
        memory = nn.ConversationMemory.from_dict(data)
        assert len(memory.messages) == 1
        assert memory.max_messages == 50


class TestFileMemory:
    """Test FileMemory."""
    
    def test_init(self, tmp_path):
        """Test initialization with file path."""
        file_path = tmp_path / "memory.json"
        memory = nn.FileMemory(file_path=str(file_path))
        assert memory.file_path == file_path
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading from file."""
        file_path = tmp_path / "memory.json"
        memory = nn.FileMemory(file_path=str(file_path))
        
        memory.add(Message.user("Hello"))
        memory.add(Message.assistant("Hi!"))
        memory.save()
        
        # Load in new instance
        new_memory = nn.FileMemory(file_path=str(file_path))
        new_memory.load()
        
        assert len(new_memory.messages) == 2
    
    def test_auto_save_on_add(self, tmp_path):
        """Test auto-saving when adding messages."""
        file_path = tmp_path / "memory.json"
        memory = nn.FileMemory(file_path=str(file_path))
        memory.add(Message.user("Test"))
        
        # Check file was created
        assert file_path.exists()
        
        # Verify content
        data = json.loads(file_path.read_text())
        assert len(data["messages"]) == 1


class TestSlidingWindowMemory:
    """Test SlidingWindowMemory."""
    
    def test_init(self):
        """Test initialization."""
        memory = nn.SlidingWindowMemory(window_size=4)
        assert memory.window_size == 4
    
    def test_window_trimming(self):
        """Test that old messages are trimmed."""
        memory = nn.SlidingWindowMemory(window_size=4)
        
        # Add system message (should be preserved)
        memory.add(Message.system("System prompt"))
        
        # Add conversation
        for i in range(10):
            memory.add(Message.user(f"User {i}"))
            memory.add(Message.assistant(f"Assistant {i}"))
        
        # Should keep system + recent messages
        messages = memory.get()
        assert messages[0].role == MessageRole.SYSTEM
        # Total should be <= window_size + 1 (for system)


class TestMemoryIntegration:
    """Test memory integration with agents."""
    
    def test_agent_with_memory(self, mock_provider):
        """Test agent using memory."""
        memory = nn.ConversationMemory()
        memory.add(Message.user("My name is John"))
        memory.add(Message.assistant("Nice to meet you John!"))
        
        agent = nn.SimpleAgent(llm=mock_provider, memory=memory)
        # Agent should have access to previous context
        assert agent.memory is not None
    
    def test_memory_context_in_chat(self, mock_provider):
        """Test that memory provides context in chat."""
        mock_provider.chat.return_value = MagicMock(
            content="Response",
            model="mock",
            provider="mock"
        )
        
        nn_interface = nn.NeuralNode.__new__(nn.NeuralNode)
        nn_interface._llm = mock_provider
        nn_interface._memory = nn.ConversationMemory()
        nn_interface._tools = []
        nn_interface._system_tools = []
        
        # Add some history
        nn_interface._memory.add(Message.user("Previous"))
        nn_interface._memory.add(Message.assistant("Previous response"))
        
        # Chat should include history
        nn_interface.chat("New message")
        
        # Verify provider was called with context
        call_args = mock_provider.chat.call_args
        messages = call_args[0][0]  # First positional argument
        assert len(messages) > 1  # Should include history + new message
