# Telegram Integration Guide for NeuralNode

Complete guide for connecting your AI agent to Telegram.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup Instructions](#setup-instructions)
3. [Basic Usage](#basic-usage)
4. [Advanced Configuration](#advanced-configuration)
5. [Security & Authentication](#security--authentication)
6. [Custom Commands](#custom-commands)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
import nn
from nn.integrations.telegram import TelegramBot, TelegramAuth

# 1. Create your AI
ai = nn.NeuralNode(provider="openai", model="gpt-4")

# 2. Create bot with your token
bot = TelegramBot(
    token="YOUR_BOT_TOKEN",
    llm=ai,
    authorized_users=[123456789]  # Your Telegram user ID
)

# 3. Run the bot
bot.run()
```

**That's it!** Your AI is now on Telegram.

---

## Setup Instructions

### Step 1: Get Bot Token from Telegram

1. Open Telegram and search for **@BotFather**
2. Start a chat and type `/newbot`
3. Follow instructions to create your bot
4. **Save the token** - it looks like: `REDACTED_TELEGRAM_TOKEN`

### Step 2: Find Your User ID

1. Search for **@userinfobot** on Telegram
2. Start the bot
3. It will reply with your user ID (e.g., `123456789`)

### Step 3: Install Dependencies

```bash
pip install python-telegram-bot
```

### Step 4: Run Your Bot

```python
# bot.py
import nn
from nn.integrations.telegram import TelegramBot

ai = nn.NeuralNode(provider="openai", api_key="your-key")

bot = TelegramBot(
    token="123456789:YOUR_TOKEN_HERE",
    llm=ai,
    authorized_users=[123456789]  # Only you can use it
)

print("Bot is running... Press Ctrl+C to stop")
bot.run()
```

Run it:
```bash
python bot.py
```

---

## Basic Usage

### Simple Chat Bot

```python
import nn
from nn.integrations.telegram import TelegramBot

ai = nn.NeuralNode(provider="auto")
bot = TelegramBot(token="YOUR_TOKEN", llm=ai)

# That's it - bot handles:
# /start - Welcome message
# /chat <message> - Chat with AI
# /help - Show help
# Text messages - Automatic response

bot.run()
```

### Agent with Tools

```python
import nn
from nn.integrations.telegram import TelegramBot
from nn.tools.web_search import WebSearch

# Create AI with tools
ai = nn.NeuralNode(provider="openai")

web_search = WebSearch().as_tool(llm=ai)
agent = ai.agent(tools=[web_search])

# Create bot with agent
bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
    agent=agent,  # Now users can delegate tasks
    authorized_users=[123456789]
)

bot.run()
```

Users can now:
- Type: `/agent Search for latest AI news and summarize`
- The bot will search the web and provide a summary

---

## Advanced Configuration

### Authorized Users Only

```python
from nn.integrations.telegram import TelegramBot, TelegramAuth

# Method 1: Simple list
bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
    authorized_users=[123456789, 987654321]  # Multiple users
)

# Method 2: Persistent auth
auth = TelegramAuth(auth_file="my_bot_auth.json")
auth.add_user(123456789, username="admin", role="admin")
auth.add_user(987654321, username="user1", role="user")

bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
    authorized_users=list(auth.authorized_users.keys())
)
```

### Custom Message Handlers

```python
from nn.integrations.telegram import TelegramBot
from telegram import Update
from telegram.ext import ContextTypes

bot = TelegramBot(token="YOUR_TOKEN", llm=ai)

# Add custom handler
async def custom_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message.text
    
    if message.startswith("/weather"):
        city = message.replace("/weather", "").strip()
        response = f"Weather for {city}: Sunny, 25°C (mock)"
        await update.message.reply_text(response)

# Register custom handler
bot.application.add_handler(
    MessageHandler(filters.TEXT & filters.Regex(r'^/weather'), custom_handler)
)

bot.run()
```

### Voice Messages (Speech-to-Text)

```python
bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
    # Voice is enabled by default!
)

# Users can send voice messages
# Bot automatically:
# 1. Downloads the voice file
# 2. Transcribes using Whisper
# 3. Responds to the transcribed text
```

### Document Processing

```python
bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
)

# Users can send:
# - PDF files
# - Text files
# - Code files
# Bot will read and summarize them
```

---

## Security & Authentication

### Production Security

```python
from nn.integrations.telegram import TelegramBot, TelegramAuth

# Strict authentication
auth = TelegramAuth()

# Only allow specific users
ALLOWED_USERS = [
    123456789,  # Admin
    987654321,  # User 1
]

bot = TelegramBot(
    token="YOUR_TOKEN",
    llm=ai,
    authorized_users=ALLOWED_USERS,
)

# Add logging
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

### Rate Limiting

```python
from nn.utils.retry import RateLimiter

class RateLimitedBot(TelegramBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limiter = RateLimiter(max_requests=30, time_window=60)
    
    async def handle_message(self, update, context):
        user_id = update.effective_user.id
        
        if not self.limiter.acquire():
            await update.message.reply_text(
                "Rate limit exceeded. Please wait a moment."
            )
            return
        
        # Process message
        await super().handle_message(update, context)
```

---

## Custom Commands

### Adding Custom Commands

```python
import nn
from nn.integrations.telegram import TelegramBot
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler

class MyBot(TelegramBot):
    def setup_handlers(self):
        super().setup_handlers()
        
        # Add custom commands
        self.application.add_handler(
            CommandHandler("weather", self.weather_command)
        )
        self.application.add_handler(
            CommandHandler("news", self.news_command)
        )
    
    async def weather_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /weather command."""
        city = ' '.join(context.args) if context.args else "Cairo"
        
        # Use agent to get weather
        response = self.llm.chat(f"What's the weather in {city}?")
        
        await update.message.reply_text(response.text)
    
    async def news_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /news command."""
        from nn.tools.web_search import WebSearch
        
        search = WebSearch()
        results = search.search("latest news", num_results=5)
        
        message = "📰 Latest News:\n\n"
        for i, result in enumerate(results, 1):
            message += f"{i}. {result.title}\n{result.url}\n\n"
        
        await update.message.reply_text(message)

# Run custom bot
bot = MyBot(token="YOUR_TOKEN", llm=ai)
bot.run()
```

---

## Complete Example: AI Assistant Bot

```python
"""
Complete Telegram AI Assistant with NeuralNode
"""

import nn
from nn.integrations.telegram import TelegramBot, TelegramAuth
from nn.tools.web_search import WebSearch
from nn.tools.system_control import SystemController
from nn.tools.browser_automation import BrowserController

class AIBot(TelegramBot):
    """Advanced AI Telegram Bot."""
    
    def __init__(self, token: str):
        # Create AI with all capabilities
        self.ai = nn.NeuralNode(
            provider="openai",
            model="gpt-4",
            memory=nn.CompleteMemorySystem(),
        )
        
        # Create agent with tools
        web_search = WebSearch().as_tool(llm=self.ai)
        system = SystemController()
        
        self.agent = self.ai.agent(
            tools=[web_search],
            system="""You are a helpful AI assistant on Telegram. 
            You can search the web and help with various tasks."""
        )
        
        # Initialize bot
        super().__init__(
            token=token,
            llm=self.ai,
            agent=self.agent,
        )
    
    def setup_handlers(self):
        """Setup all command handlers."""
        super().setup_handlers()
        
        # Add admin commands
        self.application.add_handler(
            CommandHandler("status", self.status_command)
        )

# Run the bot
if __name__ == "__main__":
    import os
    
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TOKEN_HERE")
    
    bot = AIBot(token=TOKEN)
    
    print("🤖 AI Bot is running!")
    print("Available commands:")
    print("  /start - Welcome message")
    print("  /chat - Chat with AI")
    print("  /agent - Use AI agent with tools")
    print("  /status - System status")
    print("  /help - Show help")
    
    bot.run()
```

---

## Troubleshooting

### Common Issues

**Issue: Bot not responding**
```python
# Check if token is correct
print(f"Token: {bot.token[:10]}...")  # Should print first 10 chars

# Check if authorized_users is set
print(f"Authorized: {bot.authorized_users}")
```

**Issue: "Unauthorized" message**
```python
# Your user ID is not in authorized_users
# Get your ID from @userinfobot and add it
```

**Issue: Voice messages not working**
```bash
# Install whisper
pip install openai-whisper

# Or use API
export OPENAI_API_KEY="your-key"
```

**Issue: Web search not working**
```bash
# For DuckDuckGo (default), no API key needed
# For SerpAPI:
export SERPAPI_KEY="your-key"
```

---

## Deployment

### Deploy to Server

```bash
# 1. Copy your bot file to server
scp bot.py user@server:/home/user/

# 2. Install dependencies
ssh user@server "pip install neuralnode python-telegram-bot"

# 3. Set environment variables
export TELEGRAM_BOT_TOKEN="your-token"
export OPENAI_API_KEY="your-key"

# 4. Run with nohup
nohup python bot.py &
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY bot.py .

ENV TELEGRAM_BOT_TOKEN=""
ENV OPENAI_API_KEY=""

CMD ["python", "bot.py"]
```

```yaml
# docker-compose.yml
version: '3'

services:
  telegram-bot:
    build: .
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    restart: always
```

---

## Summary

Your AI agent on Telegram can:

1. ✅ **Chat** - Respond to text messages
2. ✅ **Voice** - Transcribe and respond to voice messages
3. ✅ **Documents** - Read and analyze uploaded files
4. ✅ **Web Search** - Search the internet for information
5. ✅ **Agent Mode** - Execute complex tasks with tools
6. ✅ **Memory** - Remember conversation history
7. ✅ **Security** - Only authorized users can access
8. ✅ **Commands** - Custom commands (/weather, /news, etc.)

---

## Next Steps

1. Add more tools to your agent
2. Create custom commands for your use case
3. Deploy to a server for 24/7 availability
4. Add more authorized users
5. Set up monitoring with NeuralMonitor

---

*Last Updated: 2024-01-15*
*NeuralNode Version: 0.1.0*
