"""
Telegram Bot for Shade AI Agent
Connect your local model agent to Telegram
"""

import sys
import os
import logging
import threading
from typing import Dict, Any

# Add NeuralNode to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TelegramShadeBot:
    """Telegram bot that uses the local Shade AI model."""
    
    def __init__(self, model_path: str, telegram_token: str):
        self.model_path = model_path
        self.telegram_token = telegram_token
        self.agent = None
        self.application = None
        self.user_sessions: Dict[int, Any] = {}  # Store conversation history per user
        
    def _init_agent(self):
        """Initialize the local model agent."""
        try:
            from local_agent_gui import LocalModelAgent, WebSearchTool, CalculatorTool
            from neuralnode import Agent
            from neuralnode.local import LocalLLM
            
            logger.info(f"Loading local model from: {self.model_path}")
            
            # Load local model
            llm = LocalLLM(model_path=self.model_path)
            
            # Create tools
            tools = [
                WebSearchTool(),
                CalculatorTool()
            ]
            
            # System prompt for Telegram
            system_prompt = """You are Shade, a helpful AI assistant on Telegram. You have access to web search and calculator tools.

When you need current information, use the web_search tool.
When you need to perform calculations, use the calculator tool.

Keep your responses concise but informative. Use Telegram formatting (Markdown) when helpful.

Always be helpful, accurate, and friendly!"""
            
            self.agent = Agent(
                llm=llm,
                tools=tools,
                system_prompt=system_prompt,
                max_steps=5
            )
            
            logger.info("Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agent: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_agent_response(self, message: str, user_id: int) -> str:
        """Get response from agent."""
        try:
            if not self.agent:
                return "❌ Error: Agent not initialized"
            
            # Add user message to session history
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            
            self.user_sessions[user_id].append({"role": "user", "content": message})
            
            # Get response
            logger.info(f"Processing message from user {user_id}: {message[:50]}...")
            response = self.agent.run(message)
            
            # Add response to session history
            self.user_sessions[user_id].append({"role": "assistant", "content": response})
            
            # Keep only last 10 messages to manage memory
            if len(self.user_sessions[user_id]) > 20:
                self.user_sessions[user_id] = self.user_sessions[user_id][-20:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting response: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ Sorry, I encountered an error: {str(e)}"
    
    async def start_command(self, update, context):
        """Handle /start command."""
        welcome_message = """🤖 *Welcome to Shade AI!*

I'm your intelligent assistant powered by a local language model. I can:

🔍 *Search the web* for current information
🧮 *Calculate* mathematical expressions
💬 *Chat* about any topic

*Available commands:*
/start - Show this welcome message
/help - Get help and tips
/search <query> - Quick web search
/calc <expression> - Quick calculation
/clear - Clear conversation history

How can I help you today?"""
        
        await update.message.reply_text(
            welcome_message,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update, context):
        """Handle /help command."""
        help_message = """ℹ️ *Shade AI Help*

*Tips for best results:*

1️⃣ *Be specific* - Clear questions get better answers
2️⃣ *Use /search* for current events and facts
3️⃣ *Use /calc* for math problems
4️⃣ *Longer context* - I remember our conversation

*Example queries:*
• "What's the weather in Cairo?"
• "/search latest AI news 2024"
• "/calc (15 * 7) + 32"
• "Explain quantum computing"

*Note:* I'm running on a local model, so responses may take a few moments."""
        
        await update.message.reply_text(
            help_message,
            parse_mode='Markdown'
        )
    
    async def search_command(self, update, context):
        """Handle /search command for quick web search."""
        query = ' '.join(context.args)
        
        if not query:
            await update.message.reply_text(
                "🔍 Please provide a search query.\nExample: `/search latest Python news`",
                parse_mode='Markdown'
            )
            return
        
        # Send "typing" action
        await update.message.chat.send_action(action="typing")
        
        # Use the web search tool directly
        from local_agent_gui import WebSearchTool
        search_tool = WebSearchTool()
        result = search_tool.execute(query)
        
        # Format response
        response = f"🔍 *Search results for:* `{query}`\n\n{result[:3000]}"  # Limit length
        
        await update.message.reply_text(
            response,
            parse_mode='Markdown'
        )
    
    async def calc_command(self, update, context):
        """Handle /calc command for calculations."""
        expression = ' '.join(context.args)
        
        if not expression:
            await update.message.reply_text(
                "🧮 Please provide a mathematical expression.\nExample: `/calc (15 * 7) + 32`",
                parse_mode='Markdown'
            )
            return
        
        # Use calculator tool
        from local_agent_gui import CalculatorTool
        calc_tool = CalculatorTool()
        result = calc_tool.execute(expression)
        
        await update.message.reply_text(
            f"🧮 *Calculation:*\n`{expression}`\n\n{result}",
            parse_mode='Markdown'
        )
    
    async def clear_command(self, update, context):
        """Handle /clear command to reset conversation."""
        user_id = update.message.from_user.id
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id] = []
            await update.message.reply_text("🗑️ Conversation history cleared!")
        else:
            await update.message.reply_text("ℹ️ No conversation history to clear.")
    
    async def message_handler(self, update, context):
        """Handle regular text messages."""
        user_id = update.message.from_user.id
        message_text = update.message.text
        
        # Skip commands
        if message_text.startswith('/'):
            return
        
        # Send "typing" action
        await update.message.chat.send_action(action="typing")
        
        # Get response from agent
        response = self._get_agent_response(message_text, user_id)
        
        # Send response (truncate if too long)
        max_length = 4096  # Telegram message limit
        if len(response) > max_length:
            response = response[:max_length-100] + "\n\n... (message truncated)"
        
        await update.message.reply_text(response)
    
    def run(self):
        """Run the Telegram bot."""
        try:
            # Import python-telegram-bot
            try:
                from telegram.ext import Application, CommandHandler, MessageHandler, filters
            except ImportError:
                logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
                print("❌ python-telegram-bot not installed. Run: pip install python-telegram-bot")
                return
            
            # Initialize agent
            logger.info("Initializing agent...")
            self._init_agent()
            
            # Create application
            logger.info("Starting Telegram bot...")
            self.application = Application.builder().token(self.telegram_token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("search", self.search_command))
            self.application.add_handler(CommandHandler("calc", self.calc_command))
            self.application.add_handler(CommandHandler("clear", self.clear_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.message_handler))
            
            # Start the bot
            logger.info("Bot is running! Press Ctrl+C to stop.")
            print("🤖 Telegram bot is running!")
            print("Send /start to your bot on Telegram to begin.")
            
            self.application.run_polling(allowed_updates=Update.ALL_TYPES)
            
        except Exception as e:
            logger.error(f"Error running bot: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Shade AI Telegram Bot')
    parser.add_argument('--token', required=True, help='Telegram Bot Token')
    parser.add_argument('--model', default=r"D:\Work Space\Shade\Shade Models\best\model.safetensors",
                        help='Path to local model')
    
    args = parser.parse_args()
    
    # Check model path
    if not os.path.exists(args.model):
        print(f"❌ Model not found at: {args.model}")
        return
    
    # Create and run bot
    bot = TelegramShadeBot(
        model_path=args.model,
        telegram_token=args.token
    )
    
    bot.run()


if __name__ == "__main__":
    main()
