"""Test Telegram integration."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import neuralnode as nn
from neuralnode.core import LLMResponse
from neuralnode.integrations.telegram import (
    TelegramBot, TelegramBotBuilder, TelegramBotConfig, 
    TelegramSession, create_telegram_bot
)


# Skip tests if python-telegram-bot is not installed
try:
    from telegram import Update, User, Chat, Message as TelegramMessage
    from telegram.ext import ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="python-telegram-bot not installed")


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestTelegramSession:
    """Test TelegramSession dataclass."""
    
    def test_session_creation(self):
        """Test session creation."""
        session = TelegramSession(
            user_id=123456,
            username="testuser",
            first_name="Test"
        )
        assert session.user_id == 123456
        assert session.username == "testuser"
        assert session.authorized is True
    
    def test_session_with_history(self):
        """Test session with chat history."""
        session = TelegramSession(
            user_id=123456,
            username="testuser",
            first_name="Test",
            chat_history=[
                {"role": "user", "content": "Hello"}
            ]
        )
        assert len(session.chat_history) == 1


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestTelegramBotConfig:
    """Test TelegramBotConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TelegramBotConfig(token="test-token")
        assert config.token == "test-token"
        assert config.allowed_users is None  # Allow all by default
        assert config.enable_voice is True
        assert config.max_message_length == 4096


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestTelegramBotBuilder:
    """Test TelegramBotBuilder."""
    
    def test_builder_basic(self, mock_provider):
        """Test basic builder pattern."""
        builder = TelegramBotBuilder()
        bot = builder \
            .with_token("test-token") \
            .with_agent(nn.SimpleAgent(llm=mock_provider)) \
            .build()
        
        assert isinstance(bot, TelegramBot)
    
    def test_builder_with_allowed_users(self, mock_provider):
        """Test builder with allowed users."""
        builder = TelegramBotBuilder()
        bot = builder \
            .with_token("test-token") \
            .with_agent(nn.SimpleAgent(llm=mock_provider)) \
            .allow_users([123456, 789012]) \
            .build()
        
        assert bot.config.allowed_users == [123456, 789012]
    
    def test_builder_missing_token(self, mock_provider):
        """Test builder fails without token."""
        builder = TelegramBotBuilder()
        with pytest.raises(ValueError, match="Bot token required"):
            builder.with_agent(nn.SimpleAgent(llm=mock_provider)).build()
    
    def test_builder_from_env(self, mock_provider):
        """Test builder loading token from environment."""
        with patch.dict("os.environ", {"TEST_BOT_TOKEN": "env-token"}):
            builder = TelegramBotBuilder()
            bot = builder \
                .with_token_from_env("TEST_BOT_TOKEN") \
                .with_agent(nn.SimpleAgent(llm=mock_provider)) \
                .build()
            
            assert bot.token == "env-token"


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestTelegramBot:
    """Test TelegramBot."""
    
    def test_init(self, mock_provider):
        """Test bot initialization."""
        config = TelegramBotConfig(token="test-token")
        agent = nn.SimpleAgent(llm=mock_provider)
        
        bot = TelegramBot(
            token="test-token",
            agent=agent,
            config=config
        )
        
        assert bot.token == "test-token"
        assert bot.agent == agent
    
    def test_get_or_create_session(self, mock_provider):
        """Test session creation."""
        config = TelegramBotConfig(token="test-token")
        agent = nn.SimpleAgent(llm=mock_provider)
        bot = TelegramBot(token="test-token", agent=agent, config=config)
        
        # Mock user
        user = MagicMock()
        user.id = 123456
        user.username = "testuser"
        user.first_name = "Test"
        
        session = bot._get_or_create_session(user)
        
        assert session.user_id == 123456
        assert session.username == "testuser"
    
    def test_unauthorized_user(self, mock_provider):
        """Test unauthorized user handling."""
        config = TelegramBotConfig(
            token="test-token",
            allowed_users=[999999]  # Different from our test user
        )
        agent = nn.SimpleAgent(llm=mock_provider)
        bot = TelegramBot(token="test-token", agent=agent, config=config)
        
        user = MagicMock()
        user.id = 123456
        user.username = "testuser"
        user.first_name = "Test"
        
        session = bot._get_or_create_session(user)
        assert session.authorized is False


@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_telegram_bot(self, mock_provider):
        """Test create_telegram_bot function."""
        agent = nn.SimpleAgent(llm=mock_provider)
        bot = create_telegram_bot(
            token="test-token",
            agent=agent,
            allowed_users=[123456]
        )
        
        assert isinstance(bot, TelegramBot)
        assert bot.config.allowed_users == [123456]


class TestTelegramReplyMode:
    """Test reply mode normalization independently from telegram runtime."""

    def test_reply_mode_text(self):
        bot = TelegramBot.__new__(TelegramBot)
        bot.config = TelegramBotConfig(token="x", reply_mode="text")
        assert bot._normalized_reply_mode() == "text"

    def test_reply_mode_voice(self):
        bot = TelegramBot.__new__(TelegramBot)
        bot.config = TelegramBotConfig(token="x", reply_mode="voice")
        assert bot._normalized_reply_mode() == "voice"

    def test_reply_mode_backward_compat_voice_reply(self):
        bot = TelegramBot.__new__(TelegramBot)
        bot.config = TelegramBotConfig(token="x", reply_mode="text", voice_reply=True)
        assert bot._normalized_reply_mode() == "both"


# Integration tests (marked as slow)
@pytest.mark.slow
@pytest.mark.skipif(not TELEGRAM_AVAILABLE, reason="Telegram not available")
class TestTelegramIntegration:
    """Integration tests for Telegram bot."""
    
    @pytest.mark.asyncio
    async def test_handle_text_message(self, mock_provider):
        """Test handling text message."""
        config = TelegramBotConfig(token="test-token")
        agent = nn.SimpleAgent(llm=mock_provider)
        bot = TelegramBot(token="test-token", agent=agent, config=config)
        
        # Mock update
        update = MagicMock()
        update.effective_user.id = 123456
        update.effective_user.username = "test"
        update.effective_user.first_name = "Test"
        update.message.text = "Hello bot"
        update.message.reply_text = AsyncMock()
        
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        
        # Set up mock provider response
        mock_provider.chat.return_value = LLMResponse(
            content="Hello user!",
            model="mock",
            provider="mock"
        )
        
        await bot._handle_text_message(update, context)
        
        # Verify response was sent
        update.message.reply_text.assert_called_once()
        call_args = update.message.reply_text.call_args[0][0]
        assert "Hello user!" in call_args
