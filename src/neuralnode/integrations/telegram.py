"""Telegram bot integration for NeuralNode AI agents."""

import asyncio
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
import json

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        ContextTypes,
        filters,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("python-telegram-bot not installed. Install with: pip install python-telegram-bot")
    Update = Any
    Application = Any
    class _MockContextTypes:
        DEFAULT_TYPE = Any
    ContextTypes = _MockContextTypes


@dataclass
class TelegramUser:
    """Telegram user info."""
    id: int
    username: Optional[str]
    first_name: str
    last_name: Optional[str]
    is_authorized: bool = False


class TelegramBot:
    """Telegram bot for NeuralNode AI agent."""
    
    def __init__(
        self,
        token: str,
        llm=None,
        agent=None,
        authorized_users: Optional[List[int]] = None,
    ):
        if not TELEGRAM_AVAILABLE:
            raise RuntimeError("python-telegram-bot not installed")
        
        self.token = token
        self.llm = llm
        self.agent = agent
        self.authorized_users = set(authorized_users or [])
        self.application: Optional[Application] = None
        self.user_sessions: Dict[int, Dict[str, Any]] = {}
        self.message_handlers: List[Callable] = []
    
    def add_authorized_user(self, user_id: int) -> None:
        """Add authorized user."""
        self.authorized_users.add(user_id)
    
    def remove_authorized_user(self, user_id: int) -> None:
        """Remove authorized user."""
        self.authorized_users.discard(user_id)
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        if not self.authorized_users:
            # If no authorized users list, allow all
            return True
        return user_id in self.authorized_users
    
    def _get_user_info(self, update: Update) -> TelegramUser:
        """Extract user info from update."""
        user = update.effective_user
        return TelegramUser(
            id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            is_authorized=self.is_authorized(user.id),
        )
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            await update.message.reply_text(
                "⚠️ You are not authorized to use this bot.\n"
                f"Your user ID: {user.id}"
            )
            return
        
        welcome_text = f"""
🧠 Welcome to NeuralNode AI Bot, {user.first_name}!

Commands:
/chat <message> - Chat with AI
/agent <task> - Use AI agent with tools
/clear - Clear conversation history
/models - List available models
/help - Show help

You can also send voice messages for speech-to-text.
        """
        await update.message.reply_text(welcome_text)
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = """
🧠 NeuralNode Bot Commands:

/chat <message> - Have a conversation with AI
/agent <task> - Delegate a task to the AI agent
/clear - Clear your conversation history
/models - List available AI models
/status - Check bot status
/help - Show this help message

Features:
• Text conversations
• Voice message support
• File uploads
• Web search capability
• System control (authorized users only)
        """
        await update.message.reply_text(help_text)
    
    async def chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /chat command."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        # Get message text after /chat
        message_text = update.message.text
        if message_text.startswith('/chat'):
            query = message_text[5:].strip()
        else:
            query = message_text
        
        if not query:
            await update.message.reply_text("Please provide a message. Example: /chat Hello!")
            return
        
        # Send "typing" indicator
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        try:
            if self.llm:
                # Get or create session
                session = self.user_sessions.get(user.id, {})
                
                # Get response
                response = self.llm.chat(query)
                text = response.text if hasattr(response, 'text') else str(response)
                
                # Update session
                if 'history' not in session:
                    session['history'] = []
                session['history'].append({"user": query, "ai": text})
                self.user_sessions[user.id] = session
                
                # Send response (split if too long)
                if len(text) > 4096:
                    # Telegram message limit
                    for i in range(0, len(text), 4096):
                        await update.message.reply_text(text[i:i+4096])
                else:
                    await update.message.reply_text(text)
            else:
                await update.message.reply_text("❌ LLM not configured")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def agent_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /agent command."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        # Get task
        message_text = update.message.text
        if message_text.startswith('/agent'):
            task = message_text[6:].strip()
        else:
            task = message_text
        
        if not task:
            await update.message.reply_text("Please provide a task. Example: /agent Search for Python tutorials")
            return
        
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing"
        )
        
        try:
            if self.agent:
                # Run agent
                await update.message.reply_text(f"🤖 Working on: {task}")
                
                result = self.agent.run(task)
                
                # Send result
                if len(result) > 4096:
                    for i in range(0, len(result), 4096):
                        await update.message.reply_text(result[i:i+4096])
                else:
                    await update.message.reply_text(f"✅ Result:\n{result}")
            else:
                await update.message.reply_text("❌ Agent not configured")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /clear command."""
        user = self._get_user_info(update)
        
        if user.id in self.user_sessions:
            self.user_sessions[user.id] = {}
        
        await update.message.reply_text("🗑️ Conversation history cleared!")
    
    async def models_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /models command."""
        try:
            from ..local.quantization import ModelQuantizer
            
            quantizer = ModelQuantizer()
            models = quantizer.list_available_models()
            
            text = "📦 Available Models:\n\n"
            for m in models[:10]:  # Limit to 10
                text += f"• {m['name']}\n"
                text += f"  Size: {m['size_gb']} GB | License: {m['license']}\n\n"
            
            await update.message.reply_text(text)
        except Exception as e:
            await update.message.reply_text(f"Error listing models: {e}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command."""
        try:
            from ..local.hardware import HardwareDetector
            
            hw = HardwareDetector()
            
            status_text = f"""
🖥️ System Status:

GPU: {'✅ Available' if hw.has_cuda else '❌ Not available'}
VRAM: {hw.vram_gb:.1f} GB
RAM: {hw.ram_gb:.1f} GB
CPU Cores: {hw.cpu_count}

Bot Status:
Authorized Users: {len(self.authorized_users) if self.authorized_users else 'All users'}
Active Sessions: {len(self.user_sessions)}
LLM: {'✅ Connected' if self.llm else '❌ Not connected'}
Agent: {'✅ Connected' if self.agent else '❌ Not connected'}
            """
            await update.message.reply_text(status_text)
        except Exception as e:
            await update.message.reply_text(f"Error getting status: {e}")
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        await update.message.reply_text("🎤 Processing voice message...")
        
        try:
            # Download voice file
            voice_file = await context.bot.get_file(update.message.voice.file_id)
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                await voice_file.download_to_drive(f.name)
                voice_path = f.name
            
            # Transcribe
            try:
                from ..tools.multimodal import SpeechToText
                stt = SpeechToText()
                text = stt.transcribe(voice_path)
                
                await update.message.reply_text(f"📝 Transcription: {text}")
                
                # Process with AI
                if self.llm:
                    response = self.llm.chat(text)
                    await update.message.reply_text(response.text)
                
            finally:
                import os
                os.unlink(voice_path)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Voice processing error: {str(e)}")
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle document uploads."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        try:
            doc = update.message.document
            file = await context.bot.get_file(doc.file_id)
            
            # Download
            import tempfile
            temp_dir = tempfile.mkdtemp()
            file_path = f"{temp_dir}/{doc.file_name}"
            await file.download_to_drive(file_path)
            
            await update.message.reply_text(
                f"📄 Received: {doc.file_name}\n"
                f"Size: {doc.file_size} bytes\n"
                f"Processing..."
            )
            
            # Read file
            try:
                from ..tools.system_control import read_file
                content = read_file(file_path, limit=5000)
                
                # Summarize with AI
                if self.llm and len(content) > 100:
                    prompt = f"Summarize this document:\n\n{content[:3000]}"
                    response = self.llm.chat(prompt)
                    await update.message.reply_text(f"📋 Summary:\n{response.text}")
                else:
                    await update.message.reply_text(f"📄 Content:\n{content[:1000]}")
            
            finally:
                import shutil
                shutil.rmtree(temp_dir)
        
        except Exception as e:
            await update.message.reply_text(f"❌ Document processing error: {str(e)}")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle regular text messages."""
        user = self._get_user_info(update)
        
        if not user.is_authorized:
            return
        
        # Treat as chat command
        await self.chat_command(update, context)
    
    def setup_handlers(self) -> None:
        """Setup bot handlers."""
        self.application = Application.builder().token(self.token).build()
        
        # Commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("chat", self.chat_command))
        self.application.add_handler(CommandHandler("agent", self.agent_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("models", self.models_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
        # Media handlers
        self.application.add_handler(MessageHandler(filters.VOICE, self.handle_voice))
        self.application.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        
        # Text messages (fallback)
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
    
    def run(self) -> None:
        """Start the bot."""
        if not self.application:
            self.setup_handlers()
        
        print("🤖 Telegram Bot starting...")
        print("Press Ctrl+C to stop")
        
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    async def run_async(self) -> None:
        """Async version for running alongside other tasks."""
        if not self.application:
            self.setup_handlers()
        
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling()


class TelegramAuth:
    """Authentication manager for Telegram users."""
    
    def __init__(self, auth_file: Optional[str] = None):
        self.auth_file = auth_file or "telegram_auth.json"
        self.authorized_users: Dict[int, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load authorized users from file."""
        try:
            with open(self.auth_file, 'r') as f:
                data = json.load(f)
                self.authorized_users = {int(k): v for k, v in data.items()}
        except FileNotFoundError:
            pass
    
    def _save(self) -> None:
        """Save authorized users to file."""
        with open(self.auth_file, 'w') as f:
            json.dump(self.authorized_users, f, indent=2)
    
    def add_user(self, user_id: int, username: Optional[str] = None, role: str = "user") -> None:
        """Add authorized user."""
        self.authorized_users[user_id] = {
            "username": username,
            "role": role,
            "added_at": asyncio.get_event_loop().time(),
        }
        self._save()
    
    def remove_user(self, user_id: int) -> None:
        """Remove authorized user."""
        if user_id in self.authorized_users:
            del self.authorized_users[user_id]
            self._save()
    
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id in self.authorized_users
    
    def get_user_role(self, user_id: int) -> Optional[str]:
        """Get user role."""
        user = self.authorized_users.get(user_id)
        return user.get("role") if user else None
    
    def list_users(self) -> Dict[int, Dict[str, Any]]:
        """List all authorized users."""
        return self.authorized_users.copy()


# Convenience function
def create_bot(
    token: str,
    llm=None,
    agent=None,
    authorized_users: Optional[List[int]] = None,
) -> TelegramBot:
    """Create and configure a Telegram bot."""
    return TelegramBot(
        token=token,
        llm=llm,
        agent=agent,
        authorized_users=authorized_users,
    )
