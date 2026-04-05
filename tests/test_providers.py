"""Test all LLM providers."""

import pytest
import os
from unittest.mock import MagicMock, patch, Mock
import json

import neuralnode as nn
from neuralnode.core import Message, LLMResponse


class TestAnthropicProvider:
    """Test Anthropic Claude provider."""
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        provider = nn.AnthropicProvider(api_key="test-key", model="claude-3-haiku")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-haiku"
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="Anthropic API key required"):
            nn.AnthropicProvider(api_key=None)
    
    def test_messages_to_anthropic(self):
        """Test message format conversion."""
        provider = nn.AnthropicProvider(api_key="test")
        messages = [
            Message.system("System"),
            Message.user("Hello")
        ]
        system, anthropic_msgs = provider._prepare_messages(messages)
        assert system == "System"
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"
    
    @patch('neuralnode.providers.chat.anthropic.requests.post')
    def test_chat_success(self, mock_post):
        """Test successful chat completion."""
        mock_post.return_value.json.return_value = {
            "content": [{"text": "Hello!"}],
            "model": "claude-3-haiku",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn"
        }
        mock_post.return_value.raise_for_status = Mock()
        
        provider = nn.AnthropicProvider(api_key="test")
        messages = [Message.user("Hi")]
        response = provider.chat(messages)
        
        assert response["content"] == "Hello!"
        assert response["provider"] == "anthropic"
        mock_post.assert_called_once()
    
    def test_capabilities(self):
        """Test provider capabilities."""
        provider = nn.AnthropicProvider(api_key="test")
        caps = provider.capabilities
        assert caps.streaming is True
        assert caps.tool_use is True
        assert caps.embeddings is False


class TestOllamaProvider:
    """Test Ollama provider."""
    
    def test_init_default(self):
        """Test default initialization."""
        provider = nn.OllamaProvider()
        assert provider.model == "llama3.2"
        assert provider.base_url == "http://localhost:11434"
    
    def test_init_custom(self):
        """Test custom initialization."""
        provider = nn.OllamaProvider(model="mistral", base_url="http://custom:11434")
        assert provider.model == "mistral"
        assert provider.base_url == "http://custom:11434"
    
    def test_format_messages(self):
        """Test message formatting."""
        provider = nn.OllamaProvider()
        messages = [
            Message.system("System"),
            Message.user("Hello"),
            Message.assistant("Hi")
        ]
        result = provider._format_messages(messages)
        assert "system" in result
        assert "prompt" in result
    
    @patch('neuralnode.providers.local_providers.requests.post')
    def test_chat_success(self, mock_post):
        """Test successful chat."""
        mock_post.return_value.json.return_value = {
            "response": "Hello!",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        mock_post.return_value.raise_for_status = Mock()
        
        provider = nn.OllamaProvider()
        messages = [Message.user("Hi")]
        response = provider.chat(messages)
        
        assert response["content"] == "Hello!"
        assert response["provider"] == "ollama"
    
    def test_capabilities(self):
        """Test capabilities."""
        provider = nn.OllamaProvider()
        caps = provider.capabilities
        assert caps.chat is True
        assert caps.streaming is True
        assert caps.embeddings is True


class TestGoogleProvider:
    """Test Google/Gemini provider."""
    
    def test_init_with_api_key(self):
        """Test initialization."""
        provider = nn.GoogleProvider(api_key="test-key")
        assert provider.api_key == "test-key"
    
    def test_messages_to_gemini(self):
        """Test message conversion."""
        provider = nn.GoogleProvider(api_key="test")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi")
        ]
        result = provider._format_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "model"
    
    def test_capabilities(self):
        """Test capabilities."""
        provider = nn.GoogleProvider(api_key="test")
        caps = provider.capabilities
        assert caps.embeddings is True


class TestProviderFactory:
    """Test provider factory functions."""
    
    def test_create_provider_anthropic(self):
        """Test creating Anthropic provider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            provider = nn.create_provider("anthropic", model="claude-3-haiku")
            assert isinstance(provider, nn.AnthropicProvider)
            assert provider.model == "claude-3-haiku"
    
    def test_create_provider_ollama(self):
        """Test creating Ollama provider."""
        provider = nn.create_provider("ollama", model="llama3.2")
        assert isinstance(provider, nn.OllamaProvider)
        assert provider.model == "llama3.2"
    
    def test_create_provider_invalid(self):
        """Test creating invalid provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            nn.create_provider("invalid")
    
    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"})
    def test_auto_select_provider_anthropic(self):
        """Test auto-select with Anthropic key."""
        provider = nn.auto_select_provider()
        assert isinstance(provider, nn.AnthropicProvider)
    
    @patch('neuralnode.providers.local_providers.requests.get')
    def test_auto_select_provider_ollama(self, mock_get):
        """Test auto-select with Ollama running."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"models": []}
        
        with patch.dict(os.environ, {}, clear=True):  # No API keys
            provider = nn.auto_select_provider()
            assert isinstance(provider, nn.OllamaProvider)


class TestUniversalLocalProvider:
    """Test Universal Local Provider."""
    
    def test_init_safetensors_path(self, tmp_path):
        """Test initialization with SafeTensors path."""
        # Create fake model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.safetensors").touch()
        
        with pytest.raises(ImportError):  # transformers not installed in test
            nn.UniversalLocalProvider(str(model_dir))
    
    def test_detect_format_gguf(self, tmp_path):
        """Test format detection for GGUF."""
        model_file = tmp_path / "model.gguf"
        model_file.touch()
        
        provider = nn.UniversalLocalProvider.__new__(nn.UniversalLocalProvider)
        provider.model_path = model_file
        
        format = provider._detect_format()
        assert format.value == "gguf"
    
    def test_detect_format_unknown(self, tmp_path):
        """Test format detection for unknown format."""
        model_file = tmp_path / "model.unknown"
        model_file.touch()
        
        provider = nn.UniversalLocalProvider.__new__(nn.UniversalLocalProvider)
        provider.model_path = model_file
        
        format = provider._detect_format()
        assert format.value == "unknown"
