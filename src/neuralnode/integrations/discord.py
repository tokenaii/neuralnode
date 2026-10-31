"""Discord Bot Integration for NeuralNode.

Phase 4: Ecosystem - Connect your AI agent to Discord.
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..core import BaseAgent, Message
from ..utils.logger import get_logger

# Try to import discord.py
try:
    import discord
    from discord.ext import commands

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False


@dataclass
class DiscordSession:
    """User session in Discord."""

    user_id: int
    username: str
    guild_id: Optional[int] = None
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class DiscordBot:
    """Discord bot integration for NeuralNode agents."""

    def __init__(
        self, token: str, agent: BaseAgent, command_prefix: str = "!", intents: Optional[Any] = None
    ):
        if not DISCORD_AVAILABLE:
            raise ImportError("discord.py required. Install: pip install discord.py")

        self.token = token
        self.agent = agent
        self.command_prefix = command_prefix
        self.logger = get_logger()
        self.sessions: Dict[int, DiscordSession] = {}

        # Setup intents
        if intents is None:
            self.intents = discord.Intents.default()
            self.intents.message_content = True
        else:
            self.intents = intents

        # Create bot
        self.bot = commands.Bot(command_prefix=command_prefix, intents=self.intents)

        self._setup_events()
        self._setup_commands()

    def _setup_events(self):
        """Setup Discord events."""

        @self.bot.event
        async def on_ready():
            """Called when bot is ready."""
            self.logger.info(f"Discord bot logged in as {self.bot.user}")

        @self.bot.event
        async def on_message(message):
            """Handle messages."""
            # Ignore bot's own messages
            if message.author == self.bot.user:
                return

            # Process commands first
            await self.bot.process_commands(message)

            # Handle DMs
            if isinstance(message.channel, discord.DMChannel):
                await self._handle_dm(message)

    def _setup_commands(self):
        """Setup bot commands."""

        @self.bot.command(name="ask")
        async def ask(ctx, *, question: str):
            """Ask the AI a question."""
            async with ctx.typing():
                response = self.agent.run(question)
                await ctx.reply(response[:2000])  # Discord limit

        @self.bot.command(name="clear")
        async def clear(ctx):
            """Clear conversation history."""
            user_id = ctx.author.id
            if user_id in self.sessions:
                self.sessions[user_id].chat_history.clear()
            await ctx.reply("🧹 Conversation history cleared!")

        @self.bot.command(name="helpme")
        async def helpme(ctx):
            """Show help."""
            help_text = f"""🤖 **NeuralNode Bot Commands**

**Commands:**
`{self.command_prefix}ask <question>` - Ask the AI anything
`{self.command_prefix}clear` - Clear your conversation history
`{self.command_prefix}helpme` - Show this help

**Features:**
💬 Natural conversations
🔍 Web search capabilities
📁 File analysis
🧠 Advanced reasoning

Just DM me for private conversations!"""
            await ctx.reply(help_text)

    async def _handle_dm(self, message):
        """Handle direct messages."""
        user_id = message.author.id

        # Get or create session
        if user_id not in self.sessions:
            self.sessions[user_id] = DiscordSession(user_id=user_id, username=str(message.author))

        session = self.sessions[user_id]
        session.last_activity = datetime.now()

        # Build context from history
        context = self._build_context(session)

        # Get response
        if context:
            task = f"Previous context:\n{context}\n\nUser: {message.content}"
        else:
            task = message.content

        try:
            response = self.agent.run(task)

            # Save to history
            session.chat_history.append({"role": "user", "content": message.content})
            session.chat_history.append({"role": "assistant", "content": response})

            # Send response (split if needed)
            await self._send_long_message(message.channel, response)

        except Exception as e:
            self.logger.error(f"Error handling DM: {e}")
            await message.reply(f"❌ Error: {str(e)[:1000]}")

    def _build_context(self, session: DiscordSession) -> str:
        """Build context from chat history."""
        if not session.chat_history:
            return ""

        recent = session.chat_history[-20:]  # Last 10 exchanges
        lines = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:200]  # Truncate
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def _send_long_message(self, channel, text: str):
        """Send long message in chunks."""
        max_length = 2000  # Discord limit

        while text:
            chunk = text[:max_length]
            await channel.send(chunk)
            text = text[max_length:]

    def run(self):
        """Run the Discord bot."""
        self.logger.info("Starting Discord bot...")
        self.bot.run(self.token)

    async def close(self):
        """Close the bot."""
        await self.bot.close()


def create_discord_bot(token: str, agent: BaseAgent, command_prefix: str = "!") -> DiscordBot:
    """Quick function to create Discord bot."""
    return DiscordBot(token=token, agent=agent, command_prefix=command_prefix)
