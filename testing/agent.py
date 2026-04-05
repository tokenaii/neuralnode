import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import asyncio
import json
import requests
import sys
import io
# Import NeuralNode components
from neuralnode.core.ai import NeuralNode
from neuralnode.agents.enhanced_agent import EnhancedAgent

# Import all available tools
try:
    from neuralnode.tools.builtin import CalculatorTool, WeatherTool, WebSearchTool
except ImportError:
    pass

try:
    from neuralnode.tools.code_interpreter import CodeInterpreterTool
except ImportError:
    pass
    
try:
    from neuralnode.tools.system_control import SystemControlTool
except ImportError:
    pass
    
try:
    from neuralnode.tools.web_search import AdvancedWebSearchTool
except ImportError:
    pass


class NeuralNodeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuralNode Advanced Agent GUI")
        self.root.geometry("900x700")
        self.root.configure(bg="#1e1e2e")
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#89b4fa")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), background="#89b4fa", foreground="#11111b")
        style.map("TButton", background=[("active", "#b4befe")])
        style.configure("TCombobox", fieldbackground="#313244", background="#313244", foreground="#cdd6f4")
        
        # Agent state
        self.agent = None
        self.tools = []
        
        self.setup_ui()
        self.load_ollama_models()
        self.load_available_tools()
        
    def setup_ui(self):
        # Top Frame (Settings)
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Label(top_frame, text="NeuralNode Agent Setup", style="Header.TLabel").pack(side=tk.LEFT, padx=5)
        
        # Agent Mode Toggle
        self.agent_mode_var = tk.BooleanVar(value=True)
        self.agent_mode_check = ttk.Checkbutton(top_frame, text="Agent Mode (Iterative Thinking)", variable=self.agent_mode_var)
        self.agent_mode_check.pack(side=tk.LEFT, padx=15)
        
        # Model Selection
        model_frame = ttk.Frame(top_frame)
        model_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(model_frame, text="Model/Path:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=r"D:\Work Space\Shade\Shade Models\best")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=45)
        self.model_combo.pack(side=tk.LEFT, padx=5)
        
        self.connect_btn = ttk.Button(model_frame, text="Initialize Agent", command=self.initialize_agent)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        # Main Chat Area
        chat_frame = ttk.Frame(self.root, padding=10)
        chat_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            bg="#313244", 
            fg="#cdd6f4", 
            font=("Consolas", 11),
            insertbackground="#cdd6f4",
            state=tk.DISABLED
        )
        self.chat_display.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Configure tags for text coloring
        self.chat_display.tag_config("user", foreground="#a6e3a1", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("agent", foreground="#89b4fa", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("system", foreground="#f38ba8", font=("Consolas", 10, "italic"))
        self.chat_display.tag_config("tool", foreground="#f9e2af", font=("Consolas", 10))
        
        # Bottom Frame (Input)
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.input_text = tk.Text(bottom_frame, height=3, bg="#313244", fg="#cdd6f4", font=("Consolas", 11), insertbackground="#cdd6f4")
        self.input_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.input_text.bind("<Return>", self.on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: None) # allow newline
        
        self.send_btn = ttk.Button(bottom_frame, text="Send", command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.append_to_chat("System", "Welcome to NeuralNode Agent GUI.\nPlease select a model path or Ollama name and click 'Initialize Agent'.", "system")

    def load_ollama_models(self):
        """Fetch available models from local Ollama instance."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = [model["name"] for model in response.json().get("models", [])]
                
                # Prepend the user's local model at the start
                local_hf = r"D:\Work Space\Shade\Shade Models\best"
                if local_hf not in models:
                    models.insert(0, local_hf)
                    
                if models:
                    self.model_combo['values'] = models
                    # self.model_var.set(models[0]) is already set to local_hf
                    self.append_to_chat("System", f"Found {len(models)-1} Ollama models + Local Model.", "system")
                else:
                    self.append_to_chat("System", "No models found in Ollama. Make sure you have pulled some models (e.g., 'ollama run llama3').", "system")
        except requests.exceptions.RequestException:
            self.append_to_chat("System", "Warning: Could not connect to Ollama at localhost:11434. Is it running?", "system")

    def load_available_tools(self):
        """Load and instantiate all available NeuralNode tools."""
        self.append_to_chat("System", "Loading tools...", "system")
        
        tool_instances = []
        
        # Dynamically load tools based on what's available
        try:
            tool_instances.append(CalculatorTool())
            tool_instances.append(WeatherTool())
        except NameError: pass
        
        try:
            tool_instances.append(CodeInterpreterTool())
        except NameError: pass
            
        try:
            tool_instances.append(SystemControlTool())
        except NameError: pass
            
        try:
            tool_instances.append(AdvancedWebSearchTool())
        except NameError:
            try:
                tool_instances.append(WebSearchTool())
            except NameError: pass
            
        self.tools = tool_instances
        tool_names = [t.name for t in self.tools]
        self.append_to_chat("System", f"Loaded Tools: {', '.join(tool_names)}", "system")

    def initialize_agent(self):
        model_name = self.model_var.get()
        if not model_name:
            messagebox.showerror("Error", "Please select or type a model name.")
            return
            
        self.append_to_chat("System", f"Initializing agent with model: {model_name}...", "system")
        
        try:
            # Check if it's a file path (HuggingFace) or purely a model name (Ollama)
            if "\\" in model_name or "/" in model_name:
                self.append_to_chat("System", "Using HuggingFace pipeline for local safetensors. Loading into VRAM...", "system")
                # Initialize core NeuralNode directly with provider enum string
                node = NeuralNode(provider="huggingface", model=model_name, device="cuda")
            else:
                node = NeuralNode(provider="ollama", model=model_name)
            
            # Initialize EnhancedAgent
            self.agent = EnhancedAgent(
                llm=node,
                tools=self.tools,
                system_prompt=(
                    "You are a highly capable and intelligent AI agent built on the NeuralNode framework. "
                    "You have access to a variety of tools. Use them whenever necessary to fulfill the user's request. "
                    "Provide clear, comprehensive, and helpful answers."
                ),
                max_iterations=5,
                verbose=True
            )
            
            # IMPORTANT: Since Ollama doesn't natively support ToolCalling in the same strict way OpenAI does,
            # EnhancedAgent will fallback to the 'NativeToolParser' extracting JSON from the response.
            
            self.append_to_chat("System", "Agent successfully initialized and ready!", "system")
            self.send_btn.config(state=tk.NORMAL)
            self.connect_btn.config(text="Re-Initialize")
            
        except Exception as e:
            self.append_to_chat("System", f"Failed to initialize agent: {str(e)}", "system")

    def append_to_chat(self, sender, message, tag=None):
        self.chat_display.config(state=tk.NORMAL)
        
        self.chat_display.insert(tk.END, f"\n[{sender}]\n", tag)
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)

    def on_enter(self, event):
        if not event.state & 0x1: # If Shift is not pressed
            self.send_message()
            return "break" # Prevent newline insertion

    def send_message(self):
        if not self.agent:
            messagebox.showwarning("Warning", "Please initialize the agent first.")
            return
            
        user_input = self.input_text.get("1.0", tk.END).strip()
        if not user_input:
            return
            
        self.input_text.delete("1.0", tk.END)
        self.append_to_chat("User", user_input, "user")
        self.send_btn.config(state=tk.DISABLED)
        
        # Run agent in a separate thread to avoid freezing GUI
        threading.Thread(target=self.run_agent_task, args=(user_input,), daemon=True).start()

    def run_agent_task(self, prompt):
        try:
            self.root.after(0, self.append_to_chat, "System", "[Started processing...]", "system")
            
            if self.agent_mode_var.get():
                # Agent Mode - Iterative
                # Capture standard output to show iterations
                class StdoutRedirector(io.StringIO):
                    def __init__(self, gui_append_fn):
                        super().__init__()
                        self.gui_append_fn = gui_append_fn
                        self.buffer = ""
                        
                    def write(self, s):
                        self.buffer += s
                        if '\n' in self.buffer:
                            lines = self.buffer.split('\n')
                            for line in lines[:-1]:
                                if line.strip():
                                    self.gui_append_fn("Agent Thoughts", line.strip(), "system")
                            self.buffer = lines[-1]
                            
                redirector = StdoutRedirector(lambda s, m, t: self.root.after(0, self.append_to_chat, s, m, t))
                old_stdout = sys.stdout
                sys.stdout = redirector
                
                try:
                    response = self.agent.run(prompt=prompt)
                finally:
                    sys.stdout = old_stdout
                    
                self.root.after(0, self.append_to_chat, "NeuralNode Agent", response, "agent")
            else:
                # Direct Mode - Single Pass
                 response = self.agent.llm.complete(prompt=prompt)
                 self.root.after(0, self.append_to_chat, "Model (Direct)", response.text, "agent")
                 
        except Exception as e:
            self.root.after(0, self.append_to_chat, "System", f"Error during execution: {str(e)}", "system")
        finally:
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNodeGUI(root)
    root.mainloop()
