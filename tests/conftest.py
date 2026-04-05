"""Pytest configuration and fixtures for NeuralNode testing."""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Generator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import neuralnode as nn
from neuralnode.core import Message, MessageRole, LLMResponse, ToolResult


# Fixtures

@pytest.fixture
def mock_provider():
    """Create a mock provider for testing."""
    provider = MagicMock()
    provider.model = "mock-model"
    provider.chat.return_value = LLMResponse(
        content="Mock response",
        model="mock-model",
        provider="mock"
    )
    provider.embed.return_value = [[0.1, 0.2, 0.3]]
    provider.capabilities = {"tools": True, "streaming": True, "embeddings": True}
    provider.supports_tools.return_value = True
    return provider


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        Message.system("You are a helpful assistant."),
        Message.user("Hello!"),
        Message.assistant("Hi there!"),
    ]


@pytest.fixture
def temp_memory():
    """Create temporary memory for testing."""
    memory = nn.ConversationMemory()
    yield memory
    memory.clear()


@pytest.fixture
def mock_tool():
    """Create a mock tool for testing."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.description = "A mock tool for testing"
    tool.execute.return_value = ToolResult.ok(content="Mock tool result")
    return tool


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for testing."""
    return tmp_path


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment."""
    # Set test environment variables
    os.environ.setdefault("NEURALNODE_TEST_MODE", "true")
    yield
    # Cleanup
    os.environ.pop("NEURALNODE_TEST_MODE", None)


# Markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_api_key: marks tests that require API keys")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
