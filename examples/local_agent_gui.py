"""
NeuralNode Local Model Agent with GUI
Uses local model with web search capabilities
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import Dict, Any, List
import json

# Add NeuralNode to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import NeuralNode components
try:
    from neuralnode import Agent
    from neuralnode.core.types import MessageRole
    from neuralnode.tools import Tool
    print("NeuralNode imported successfully")
except ImportError as e:
    print(f"NeuralNode import error: {e}")
    sys.exit(1)


class WebSearchTool(Tool):
    """Web search tool for the agent."""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for current information",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            }
        )
    
    def execute(self, query: str) -> str:
        """Execute web search."""
        try:
            import requests
            from urllib.parse import quote
            
            # Use DuckDuckGo HTML version (no API key needed)
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Simple parsing - extract titles and snippets
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                results = []
                # DuckDuckGo results are in .result class
                for result in soup.select('.result')[:5]:
                    title_elem = result.select_one('.result__a')
                    snippet_elem = result.select_one('.result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text(strip=True)
                        snippet = snippet_elem.get_text(strip=True)
                        results.append(f"{title}: {snippet}")
                
                if results:
                    return "\n\n".join(results)
                else:
                    return "No results found. Try a different query."
            else:
                return f"Search failed with status {response.status_code}"
                
        except ImportError:
            return "Web search requires 'requests' and 'beautifulsoup4'. Install: pip install requests beautifulsoup4"
        except Exception as e:
            return f"Search error: {str(e)}"


class CalculatorTool(Tool):
    """Calculator tool for math operations."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            }
        )
    
    def execute(self, expression: str) -> str:
        """Safely evaluate mathematical expression."""
        try:
            # Only allow safe math operations
            allowed_names = {
                'abs': abs, 'max': max, 'min': min,
                'sum': sum, 'len': len,
            }
            
            # Clean the expression
            expression = expression.replace('^', '**')
            
            # Evaluate safely
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"


class LocalModelAgent:
    """Agent that uses local model with tools."""
    
    def __init__(self, model_path: str, thinking_callback=None):
        self.model_path = model_path
        self.thinking_callback = thinking_callback
        self.agent = None
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the agent with local model."""
        try:
            # Import local LLM provider
            from neuralnode.local import LocalLLM
            
            print(f"Loading local model from: {self.model_path}")
            
            # Load the local model
            llm = LocalLLM(model_path=self.model_path)
            
            # Create tools
            tools = [
                WebSearchTool(),
                CalculatorTool()
            ]
            
            # Create agent with system prompt
            system_prompt = """You are Shade, a helpful AI assistant. You have access to web search and calculator tools.

When you need current information, use the web_search tool.
When you need to perform calculations, use the calculator tool.

Always think step by step:
1. Understand the user's question
2. Decide if you need tools to answer
3. Use tools if needed
4. Provide a clear, helpful response

Show your thinking process before giving the final answer."""
            
            self.agent = Agent(
                llm=llm,
                tools=tools,
                system_prompt=system_prompt,
                max_steps=5
            )
            
            print("Agent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing agent: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def chat(self, message: str) -> str:
        """Send message to agent and get response."""
        if not self.agent:
            return "Error: Agent not initialized"
        
        try:
            # Show thinking
            if self.thinking_callback:
                self.thinking_callback("🤔 Thinking...")
            
            # Get response from agent
            response = self.agent.run(message)
            
            return response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg


class AgentGUI:
    """GUI for the Local Model Agent."""
    
    def __init__(self, root: tk.Tk, model_path: str):
        self.root = root
        self.root.title("Shade AI Agent - Local Model")
        self.root.geometry("900x700")
        
        self.model_path = model_path
        self.agent: LocalModelAgent = None
        self.conversation_history: List[Dict[str, str]] = []
        
        self._setup_ui()
        self._init_agent()
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)  # Chat area
        main_frame.rowconfigure(2, weight=1)  # Thinking area
        main_frame.rowconfigure(3, weight=0)  # Input area
        
        # Header
        header = ttk.Label(
            main_frame,
            text="🤖 Shade AI Agent (Local Model)",
            font=('Helvetica', 16, 'bold')
        )
        header.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Model info
        model_info = ttk.Label(
            main_frame,
            text=f"Model: {os.path.basename(self.model_path)}",
            font=('Helvetica', 9)
        )
        model_info.grid(row=0, column=0, pady=(0, 10), sticky=tk.E)
        
        # Chat display
        chat_frame = ttk.LabelFrame(main_frame, text="Conversation", padding="5")
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#f5f5f5'
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_display.config(state=tk.DISABLED)
        
        # Thinking display
        thinking_frame = ttk.LabelFrame(main_frame, text="Agent Thinking", padding="5")
        thinking_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        thinking_frame.columnconfigure(0, weight=1)
        thinking_frame.rowconfigure(0, weight=1)
        
        self.thinking_display = scrolledtext.ScrolledText(
            thinking_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#fff8dc',
            height=8
        )
        self.thinking_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.thinking_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.input_field = ttk.Entry(input_frame, font=('Consolas', 11))
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.input_field.bind('<Return>', self._on_send)
        
        send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._on_send,
            width=10
        )
        send_btn.grid(row=0, column=1)
        
        # Tools frame
        tools_frame = ttk.LabelFrame(main_frame, text="Tools", padding="5")
        tools_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.web_search_var = tk.BooleanVar(value=True)
        web_search_cb = ttk.Checkbutton(
            tools_frame,
            text="Web Search",
            variable=self.web_search_var
        )
        web_search_cb.pack(side=tk.LEFT, padx=5)
        
        self.calculator_var = tk.BooleanVar(value=True)
        calculator_cb = ttk.Checkbutton(
            tools_frame,
            text="Calculator",
            variable=self.calculator_var
        )
        calculator_cb.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
    
    def _init_agent(self):
        """Initialize the agent in a separate thread."""
        def init():
            try:
                self.status_var.set("Loading local model...")
                self.agent = LocalModelAgent(
                    model_path=self.model_path,
                    thinking_callback=self._update_thinking
                )
                self.status_var.set("Ready - Model loaded")
                self._append_to_chat("System", "✅ Agent ready! I can search the web and perform calculations. What would you like to know?")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self._append_to_chat("System", f"❌ Error loading model: {str(e)}")
        
        thread = threading.Thread(target=init, daemon=True)
        thread.start()
    
    def _update_thinking(self, text: str):
        """Update the thinking display."""
        self.root.after(0, lambda: self._append_to_thinking(text))
    
    def _append_to_thinking(self, text: str):
        """Append text to thinking display."""
        self.thinking_display.config(state=tk.NORMAL)
        self.thinking_display.insert(tk.END, f"{text}\n")
        self.thinking_display.see(tk.END)
        self.thinking_display.config(state=tk.DISABLED)
    
    def _append_to_chat(self, sender: str, message: str):
        """Append message to chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        if sender == "You":
            self.chat_display.insert(tk.END, f"\n👤 You:\n{message}\n", "user")
            self.chat_display.tag_config("user", foreground="#0066cc", font=('Consolas', 10, 'bold'))
        elif sender == "Shade":
            self.chat_display.insert(tk.END, f"\n🤖 Shade:\n{message}\n", "agent")
            self.chat_display.tag_config("agent", foreground="#006600", font=('Consolas', 10))
        else:
            self.chat_display.insert(tk.END, f"\n🌐 {sender}:\n{message}\n", "system")
            self.chat_display.tag_config("system", foreground="#666666", font=('Consolas', 9, 'italic'))
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def _on_send(self, event=None):
        """Handle send button click."""
        message = self.input_field.get().strip()
        if not message:
            return
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # Add to chat
        self._append_to_chat("You", message)
        
        # Clear thinking
        self.thinking_display.config(state=tk.NORMAL)
        self.thinking_display.delete(1.0, tk.END)
        self.thinking_display.config(state=tk.DISABLED)
        
        # Process in thread
        def process():
            self.status_var.set("Thinking...")
            if self.agent:
                response = self.agent.chat(message)
                self.root.after(0, lambda: self._append_to_chat("Shade", response))
            else:
                self.root.after(0, lambda: self._append_to_chat("System", "Agent not ready yet. Please wait..."))
            self.status_var.set("Ready")
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()


def main():
    """Main entry point."""
    # Model path
    model_path = r"D:\Work Space\Shade\Shade Models\best\model.safetensors"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please check the path and try again.")
        return
    
    # Create GUI
    root = tk.Tk()
    app = AgentGUI(root, model_path)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()
