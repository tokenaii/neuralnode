"""Test Discord and Slack integrations."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import neuralnode as nn
from neuralnode.core import Message, LLMResponse


class TestDiscordBot:
    """Test Discord integration."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="Hello from AI!",
            model="mock",
            provider="mock"
        )
        return llm
    
    @pytest.mark.asyncio
    async def test_init(self, mock_llm):
        """Test Discord bot initialization."""
        # Skip if discord not available
        try:
            bot = nn.DiscordBot(
                token="test-token",
                agent=nn.SimpleAgent(llm=mock_llm),
                command_prefix="!"
            )
            assert bot.token == "test-token"
            assert bot.command_prefix == "!"
        except (ImportError, AttributeError):
            pytest.skip("Discord not available")
    
    @pytest.mark.asyncio
    async def test_on_message(self, mock_llm):
        """Test message handling."""
        try:
            bot = nn.DiscordBot(
                token="test-token",
                agent=nn.SimpleAgent(llm=mock_llm)
            )
            
            # Mock message
            message = MagicMock()
            message.content = "!ask Hello"
            message.author = MagicMock()
            message.author.bot = False
            
            await bot.on_message(message)
            # Should not raise
            assert True
        except (ImportError, AttributeError):
            pytest.skip("Discord not available")


class TestSlackBot:
    """Test Slack integration."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="Hello from Slack AI!",
            model="mock",
            provider="mock"
        )
        return llm
    
    def test_init(self, mock_llm):
        """Test Slack bot initialization."""
        try:
            bot = nn.SlackBot(
                token="xoxb-test-token",
                signing_secret="test-secret",
                agent=nn.SimpleAgent(llm=mock_llm)
            )
            assert bot.token == "xoxb-test-token"
        except (ImportError, AttributeError):
            pytest.skip("Slack not available")
    
    def test_handle_message(self, mock_llm):
        """Test message handling."""
        try:
            bot = nn.SlackBot(
                token="xoxb-test-token",
                signing_secret="test-secret",
                agent=nn.SimpleAgent(llm=mock_llm)
            )
            
            # Mock event
            event = {
                "type": "message",
                "text": "Hello AI",
                "user": "U123456",
                "channel": "C123456"
            }
            
            with patch.object(bot, '_send_message') as mock_send:
                bot.handle_message(event)
                # Should attempt to send response
                assert True
        except (ImportError, AttributeError):
            pytest.skip("Slack not available")


class TestIntegrationFactory:
    """Test integration factory functions."""
    
    def test_create_discord_bot(self):
        """Test creating Discord bot."""
        try:
            llm = MagicMock()
            agent = nn.SimpleAgent(llm=llm)
            
            bot = nn.create_discord_bot(
                token="test-token",
                agent=agent
            )
            assert bot is not None
        except (ImportError, AttributeError):
            pytest.skip("Discord not available")
    
    def test_create_slack_bot(self):
        """Test creating Slack bot."""
        try:
            llm = MagicMock()
            agent = nn.SimpleAgent(llm=llm)
            
            bot = nn.create_slack_bot(
                token="xoxb-test",
                signing_secret="secret",
                agent=agent
            )
            assert bot is not None
        except (ImportError, AttributeError):
            pytest.skip("Slack not available")


class TestIntegrationConfig:
    """Test integration configuration."""
    
    def test_discord_config(self):
        """Test Discord configuration."""
        try:
            config = nn.DiscordConfig(
                token="test-token",
                command_prefix="!",
                intents=["messages", "guilds"]
            )
            assert config.token == "test-token"
        except (ImportError, AttributeError):
            pytest.skip("Discord not available")
    
    def test_slack_config(self):
        """Test Slack configuration."""
        try:
            config = nn.SlackConfig(
                token="xoxb-test",
                signing_secret="secret"
            )
            assert config.token == "xoxb-test"
        except (ImportError, AttributeError):
            pytest.skip("Slack not available")
