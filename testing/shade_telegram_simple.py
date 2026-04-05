"""
Shade AI - Simple Local Model Telegram Bot
===========================================
Quick start version for local model + Telegram
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============== CONFIGURATION ==============
MODEL_PATH = r"D:\Work Space\Shade\Shade Models\best"
TELEGRAM_TOKEN = "REDACTED_TELEGRAM_TOKEN"

# Add neuralnode to path
sys.path.insert(0, r"D:\Work Space\NeuralNode\neuralnode\src")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
except ImportError:
    print("Installing required packages...")
    os.system("pip install torch transformers accelerate -q")
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    print("Installing telegram...")
    os.system("pip install python-telegram-bot -q")
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


class ShadeTelegramBot:
    """
    Telegram bot powered by local Shade model
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.app = None
        
    def load_model(self):
        """Load the local Shade model"""
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            logger.info("Loading model (this may take a few minutes)...")
            
            # Try to load with optimizations
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                device=0 if device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
            # Try alternative loading method
            logger.info("Trying alternative loading method...")
            try:
                self.pipe = pipeline(
                    "text-generation",
                    model=MODEL_PATH,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info("Model loaded with pipeline!")
                return True
            except Exception as e2:
                logger.error(f"Failed to load model: {e2}")
                return False
    
    def generate_response(self, user_message: str, user_id: str = "unknown") -> str:
        """Generate response using local model"""
        
        # Create prompt with system context
        system_prompt = """You are Shade AI, an intelligent and helpful assistant. 
You have access to the local system and can help with various tasks.
Be concise, accurate, and helpful in your responses."""
        
        # Format for chat
        prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
        
        try:
            if self.pipe:
                # Generate using pipeline
                result = self.pipe(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    return_full_text=False
                )[0]["generated_text"]
                
                # Clean up response
                response = result.strip()
                
                # Remove prompt echo if present
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                return response
            else:
                return "Model not loaded properly. Please restart the bot."
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Sorry, I encountered an error: {str(e)[:200]}"
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user = update.effective_user
        
        welcome = f"""
🤖 *Welcome to Shade AI!*

Hello {user.first_name}! I'm an AI assistant powered by a local language model.

*What I can do:*
💬 Have intelligent conversations
📝 Help with writing and analysis
🔍 Answer questions
💡 Provide suggestions and ideas

*Commands:*
/start - Show this welcome message
/help - Show help information
/status - Check system status
/clear - Clear conversation history

Just send me a message and I'll respond!
"""
        await update.message.reply_text(welcome, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_text = """
*Shade AI Help*

*How to use:*
Simply type your message and I'll respond!

*Example questions:*
• "What is machine learning?"
• "Help me write an email"
• "Explain quantum physics"
• "Write Python code for..."
• "Summarize this concept"

*Tips:*
• Be specific for better answers
• I can handle long conversations
• Feel free to ask follow-up questions

*Limitations:*
• I don't have internet access
• I can't access real-time information
• My knowledge is based on my training data
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        import platform
        import psutil
        
        # Get system info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get model info
        model_loaded = "✅ Loaded" if self.model or self.pipe else "❌ Not Loaded"
        device = "GPU" if torch.cuda.is_available() else "CPU"
        
        status = f"""
⚙️ *System Status*

🤖 Model: {model_loaded}
🖥️ Device: {device}
💾 CPU Usage: {cpu_percent}%
🧠 RAM: {memory.percent}% used
💽 Disk: {disk.percent}% used
🐍 Python: {platform.python_version()}
"""
        await update.message.reply_text(status, parse_mode='Markdown')
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        await update.message.reply_text("🧹 Conversation cleared! (Note: Local model doesn't persist chat history between restarts)")
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages"""
        user = update.effective_user
        message = update.message.text
        
        logger.info(f"Message from {user.first_name} ({user.id}): {message[:50]}...")
        
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        # Generate response
        response = self.generate_response(message, str(user.id))
        
        # Send response (handle long messages)
        if len(response) > 4000:
            # Split into chunks
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await update.message.reply_text(chunk)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
        else:
            await update.message.reply_text(response)
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Error: {context.error}")
        if isinstance(update, Update) and update.message:
            await update.message.reply_text("❌ An error occurred. Please try again.")
    
    def run(self):
        """Run the bot"""
        print("=" * 60)
        print("🤖 Shade AI - Telegram Bot")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            print("\n❌ Failed to load model. Please check:")
            print(f"   1. Model path exists: {MODEL_PATH}")
            print("   2. Required packages are installed")
            print("   3. You have enough RAM/VRAM")
            return
        
        print("\n" + "=" * 60)
        print("✅ Model loaded successfully!")
        print("=" * 60)
        
        # Setup Telegram bot
        print("\n🚀 Starting Telegram bot...")
        
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("status", self.status_command))
        self.app.add_handler(CommandHandler("clear", self.clear_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Add error handler
        self.app.add_error_handler(self.error_handler)
        
        # Get bot info
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def get_bot_info():
            bot = await self.app.bot.get_me()
            return bot
        
        try:
            bot_info = loop.run_until_complete(get_bot_info())
            print(f"\n✅ Bot connected: @{bot_info.username}")
            print(f"📱 Bot ID: {bot_info.id}")
            print("\n" + "=" * 60)
            print("🤖 Bot is running! Send /start in Telegram")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"\n❌ Error connecting to Telegram: {e}")
            print("Please check your bot token.")
            return
        finally:
            loop.close()
        
        # Run the bot
        print("Press Ctrl+C to stop\n")
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    bot = ShadeTelegramBot()
    bot.run()
