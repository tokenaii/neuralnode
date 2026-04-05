"""
Shade AI - Local Model Agent with Telegram Integration
======================================================
Intelligent agent using local model with full system control
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Add neuralnode to path if needed
sys.path.insert(0, r"D:\Work Space\NeuralNode\neuralnode\src")

import neuralnode as nn
from neuralnode import (
    NeuralNode, create_react_agent, create_openai_functions_agent,
    GoogleSearchTool, WikipediaTool, WebScraperTool,
    PythonREPLTool, BashTool, SQLDatabaseTool, MongoDBTool,
    FileReadTool, FileWriteTool, ListDirectoryTool, APITool,
    TelegramBot, TelegramSession,
    ConversationBufferMemory, create_vector_store_memory,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============== CONFIGURATION ==============

MODEL_PATH = r"D:\Work Space\Shade\Shade Models\best"
TELEGRAM_TOKEN = "REDACTED_TELEGRAM_TOKEN"
ADMIN_USER_ID = None  # Set your Telegram user ID for admin commands

# Allowed paths for file operations (for security)
ALLOWED_PATHS = [
    r"D:\Work Space",
    r"C:\Users",
    os.path.expanduser("~"),
]


class ShadeAI:
    """
    Shade AI - Intelligent Local Model Agent
    ========================================
    Features:
    - Local LLM powered by HuggingFace Transformers
    - Multi-tool agent with ReAct reasoning
    - Full system control (files, commands, search)
    - Telegram integration
    """
    
    def __init__(self):
        self.llm = None
        self.agent = None
        self.telegram_bot = None
        self.memory = None
        self.tools = []
        
    def load_local_model(self):
        """Load the local Shade model."""
        logger.info("Loading local model from: %s", MODEL_PATH)
        
        try:
            # Use TransformersProvider for local HuggingFace model
            self.llm = nn.TransformersProvider(
                model=MODEL_PATH,  # Path to model directory
                device="auto",     # Auto-detect GPU/CPU
                torch_dtype="float16",  # Use fp16 for efficiency
                load_in_8bit=False,     # Set True for lower VRAM usage
                trust_remote_code=True
            )
            
            # Test the model
            test_response = self.llm.chat([{"role": "user", "content": "Hello"}])
            logger.info("Model loaded successfully! Test response: %s", 
                       test_response.get("content", "")[:100])
            
            return True
            
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            logger.info("Trying with 8-bit quantization...")
            
            try:
                # Retry with 8-bit quantization for lower VRAM
                self.llm = nn.TransformersProvider(
                    model=MODEL_PATH,
                    device="auto",
                    load_in_8bit=True,
                    trust_remote_code=True
                )
                logger.info("Model loaded with 8-bit quantization!")
                return True
            except Exception as e2:
                logger.error("Failed to load model with 8-bit: %s", e2)
                return False
    
    def setup_tools(self):
        """Setup all available tools for the agent."""
        logger.info("Setting up tools...")
        
        # Web & Search Tools
        web_search = GoogleSearchTool()  # Requires SERPAPI_API_KEY
        wikipedia = WikipediaTool()
        web_scraper = WebScraperTool()
        
        # Code Execution Tools
        python_repl = PythonREPLTool(
            timeout=30,
            allowed_modules=["math", "random", "datetime", "json", "re", "collections"]
        )
        
        # System Command Tool (BE CAREFUL!)
        bash = BashTool(
            allowed_commands=["dir", "ls", "echo", "cat", "type", "ping", "ipconfig", "systeminfo"],
            timeout=30
        )
        
        # File System Tools (with security restrictions)
        file_reader = FileReadTool(allowed_paths=ALLOWED_PATHS)
        file_writer = FileWriteTool(allowed_paths=ALLOWED_PATHS)
        dir_lister = ListDirectoryTool(allowed_paths=ALLOWED_PATHS)
        
        # Database Tools (configure connection strings as needed)
        sql_tool = SQLDatabaseTool(
            connection_string=os.getenv("DATABASE_URL", "sqlite:///shade.db")
        )
        mongodb_tool = MongoDBTool(
            connection_string=os.getenv("MONGODB_URI"),
            database="shade_db"
        )
        
        # API Tool
        api_tool = APITool()
        
        # Collect all tools
        self.tools = [
            web_search,
            wikipedia,
            web_scraper,
            python_repl,
            bash,
            file_reader,
            file_writer,
            dir_lister,
            sql_tool,
            mongodb_tool,
            api_tool,
        ]
        
        logger.info("Setup %d tools", len(self.tools))
    
    def setup_memory(self):
        """Setup conversation memory."""
        logger.info("Setting up memory...")
        
        # Use buffer memory for conversation history
        self.memory = ConversationBufferMemory(
            max_messages=20,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def create_agent(self):
        """Create the intelligent agent."""
        logger.info("Creating agent...")
        
        if not self.llm:
            raise ValueError("LLM not loaded. Call load_local_model() first.")
        
        # Create ReAct agent with all tools
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt="""You are Shade AI, an intelligent assistant with full system access.
You can:
- Search the web using Google Search and Wikipedia
- Execute Python code
- Run system commands
- Read and write files
- Query databases
- Make API calls

When asked to perform tasks:
1. Use tools to gather information
2. Execute commands to complete tasks
3. Always confirm before making destructive changes
4. Provide clear explanations of your actions

Be helpful, thorough, and careful with system operations.""",
            max_iterations=15,
            verbose=True
        )
        
        logger.info("Agent created successfully!")
    
    def setup_telegram(self):
        """Setup Telegram bot integration."""
        logger.info("Setting up Telegram bot...")
        
        try:
            from telegram import Update
            from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
        except ImportError:
            logger.error("python-telegram-bot required. Install: pip install python-telegram-bot")
            return False
        
        # Create Telegram bot
        self.telegram_bot = TelegramBot(
            token=TELEGRAM_TOKEN,
            agent=self.agent,
            memory=self.memory
        )
        
        # Build application
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add handlers
        app.add_handler(CommandHandler("start", self._telegram_start))
        app.add_handler(CommandHandler("help", self._telegram_help))
        app.add_handler(CommandHandler("status", self._telegram_status))
        app.add_handler(CommandHandler("clear", self._telegram_clear))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._telegram_message))
        
        self.telegram_app = app
        logger.info("Telegram bot setup complete!")
        return True
    
    async def _telegram_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = """
🤖 *Shade AI* - Your Local Intelligent Agent

I have access to:
🔍 Web Search (Google, Wikipedia)
💻 Code Execution (Python, Bash)
📁 File System (Read/Write/Ls)
🗄️ Databases (SQL, MongoDB)
🌐 API Calls

*Commands:*
/start - Show this message
/help - Show help
/status - Check system status
/clear - Clear conversation history

*Security Notice:* I have limited system access. Admin commands require authentication.
"""
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def _telegram_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
*Available Commands:*

🔍 *Search:*
- "Search for latest AI news"
- "What is quantum computing?"

💻 *Code:*
- "Calculate fibonacci(20) in Python"
- "Run: echo Hello World"

📁 *Files:*
- "Read file D:\\path\\to\\file.txt"
- "List files in D:\\Work Space"
- "Write 'content' to file.txt"

🗄️ *Database:*
- "Query: SELECT * FROM users"

🌐 *Web:*
- "Scrape https://example.com"
- "Get API https://api.example.com/data"

Just type naturally and I'll use the right tools!
"""
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def _telegram_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        import platform
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        status = f"""
⚙️ *System Status*

🖥️ OS: {platform.system()} {platform.release()}
💾 CPU: {cpu_percent}% used
🧠 RAM: {memory.percent}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)
🤖 Model: Local Shade Model (Transformers)
🔧 Tools: {len(self.tools)} available
💬 Memory: {len(self.memory.messages) if self.memory else 0} messages
"""
        await update.message.reply_text(status, parse_mode='Markdown')
    
    async def _telegram_clear(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command."""
        if self.memory:
            self.memory.clear()
        await update.message.reply_text("🧹 Conversation history cleared!")
    
    async def _telegram_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        user_message = update.message.text
        user_id = update.effective_user.id
        
        logger.info("Message from user %s: %s", user_id, user_message[:50])
        
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        try:
            # Run agent
            result = self.agent.run(user_message)
            
            response = result.output
            
            # Add metadata about tools used
            if result.intermediate_steps:
                tools_used = list(set(step.get("action", "") for step in result.intermediate_steps if step.get("action")))
                if tools_used:
                    response += f"\n\n🔧 Tools used: {', '.join(tools_used)}"
            
            # Send response (Telegram has 4096 char limit)
            if len(response) > 4000:
                # Split into chunks
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(response)
                
        except Exception as e:
            logger.error("Error processing message: %s", e)
            await update.message.reply_text(f"❌ Error: {str(e)[:500]}")
    
    def run(self):
        """Initialize and run Shade AI."""
        logger.info("="*50)
        logger.info("Starting Shade AI")
        logger.info("="*50)
        
        # 1. Load model
        if not self.load_local_model():
            logger.error("Failed to load model. Exiting.")
            return False
        
        # 2. Setup tools
        self.setup_tools()
        
        # 3. Setup memory
        self.setup_memory()
        
        # 4. Create agent
        self.create_agent()
        
        # 5. Setup Telegram
        if not self.setup_telegram():
            logger.warning("Telegram bot setup failed. Running in CLI mode.")
            self._run_cli()
            return True
        
        # 6. Start Telegram bot
        logger.info("Starting Telegram bot...")
        print("\n" + "="*50)
        print("🤖 Shade AI is running!")
        print("="*50)
        print(f"📱 Telegram Bot: @{self.telegram_app.bot.username}")
        print("💬 Send /start to begin")
        print("="*50 + "\n")
        
        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)
        return True
    
    def _run_cli(self):
        """Run in CLI mode (fallback)."""
        print("\n" + "="*50)
        print("🤖 Shade AI - CLI Mode")
        print("="*50)
        print("Type 'exit' to quit\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("\nShade AI is thinking...")
                result = self.agent.run(user_input)
                
                print(f"\nShade AI: {result.output}")
                
                if result.intermediate_steps:
                    print("\n📋 Steps taken:")
                    for step in result.intermediate_steps:
                        print(f"  - {step.get('action', 'unknown')}: {step.get('input', '')[:50]}...")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        print("\nGoodbye! 👋")


# ============== MAIN ENTRY POINT ==============

if __name__ == "__main__":
    shade = ShadeAI()
    shade.run()
