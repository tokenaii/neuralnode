"""Command Line Interface for NeuralNode."""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional


def chat_command(args):
    """Interactive chat command."""
    try:
        import nn
        from nn.utils.monitoring import get_monitor
        
        print("🧠 NeuralNode Chat")
        print("Type 'exit' to quit, 'help' for commands\n")
        
        # Initialize
        ai = nn.NeuralNode(provider=args.provider or "auto", model=args.model)
        
        # Enable monitoring
        monitor = get_monitor()
        
        # Enable features
        if args.browser:
            print("Browser automation enabled")
        if args.web:
            print("Web search enabled")
        
        # Chat loop
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  exit - Quit chat")
                    print("  clear - Clear conversation")
                    print("  model - Show current model")
                    print("  save <file> - Save conversation")
                    print("\nShortcuts:")
                    print("  !search <query> - Web search")
                    print("  !browser <url> - Navigate to URL")
                    continue
                
                if user_input.lower() == 'clear':
                    ai.clear_memory()
                    print("Conversation cleared.")
                    continue
                
                if user_input.lower().startswith('save '):
                    filename = user_input[5:].strip()
                    # Save conversation logic here
                    print(f"Conversation saved to {filename}")
                    continue
                
                # Start monitoring trace
                run_id = monitor.start_trace(name="chat_turn", inputs={"prompt": user_input})
                
                # Get response
                if args.stream:
                    print("\n🤖 Assistant: ", end="", flush=True)
                    for chunk in ai.chat(user_input, stream=True):
                        print(chunk.text, end="", flush=True)
                    print()
                else:
                    response = ai.chat(user_input)
                    print(f"\n🤖 Assistant: {response.text}")
                
                # End trace
                monitor.end_trace(run_id)
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure NeuralNode is installed: pip install -e .")
        sys.exit(1)


def run_command(args):
    """Run a single prompt."""
    try:
        import nn
        
        ai = nn.NeuralNode(provider=args.provider or "auto", model=args.model)
        
        # Read from file if specified
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                prompt = f.read()
        else:
            prompt = args.prompt
        
        # Execute
        if args.json:
            # Structured output
            from pydantic import BaseModel, Field
            
            class Output(BaseModel):
                result: str = Field(description="The main result")
                details: Optional[str] = Field(description="Additional details")
            
            response = ai.structured_output(prompt, output_schema=Output)
            print(json.dumps(response.dict(), indent=2))
        else:
            response = ai.chat(prompt)
            print(response.text)
    
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def agent_command(args):
    """Run agent with tools."""
    try:
        import nn
        from nn.tools.web_search import WebSearch
        from nn.tools.browser_automation import BrowserController
        
        print("🤖 NeuralNode Agent Mode")
        print("Available tools: web_search, browser, file, shell\n")
        
        ai = nn.NeuralNode(provider=args.provider or "auto", model=args.model)
        
        # Create tools
        tools = []
        
        if args.web:
            web_search = WebSearch().as_tool(llm=ai)
            tools.append(web_search)
        
        if args.browser:
            # Browser tool would be added here
            pass
        
        # Create agent
        agent = ai.agent(tools=tools, verbose=True)
        
        # Read task
        if args.file:
            with open(args.file, 'r') as f:
                task = f.read()
        else:
            task = args.task
        
        print(f"Task: {task}\n")
        result = agent.run(task)
        print(f"\n✅ Result: {result}")
    
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def models_command(args):
    """List available models."""
    try:
        from nn.local.quantization import ModelQuantizer, ModelDownloader
        
        if args.list:
            # List popular models
            quantizer = ModelQuantizer()
            models = quantizer.list_available_models()
            
            print("\n📦 Available Models:\n")
            for m in models:
                print(f"  {m['id']}")
                print(f"    Name: {m['name']}")
                print(f"    Size: {m['size_gb']} GB")
                print(f"    License: {m['license']}")
                print()
        
        elif args.local:
            # List local models
            downloader = ModelDownloader()
            models = downloader.list_local_models()
            
            print("\n💾 Local Models:\n")
            for m in models:
                print(f"  {m['name']} ({m['format']}) - {m['size_gb']} GB")
                print(f"    Path: {m['path']}")
                print()
        
        elif args.download:
            # Download model
            print(f"Downloading {args.download}...")
            quantizer = ModelQuantizer()
            path = quantizer.download_model(args.download)
            if path:
                print(f"✅ Downloaded to: {path}")
        
        elif args.quantize:
            # Quantize model
            print(f"Quantizing {args.quantize} to {args.method or 'Q4_K_M'}...")
            from nn.local.quantization import quantize_model
            result = quantize_model(args.quantize, method=args.method or "Q4_K_M")
            if result:
                print(f"✅ Quantized: {result}")
    
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def monitor_command(args):
    """Show monitoring dashboard."""
    try:
        from nn.utils.monitoring import get_monitor
        
        monitor = get_monitor()
        
        if args.dashboard:
            monitor.print_dashboard()
        elif args.export:
            data = monitor.get_dashboard_data()
            with open(args.export, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Dashboard data exported to {args.export}")
    
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nn',
        description='NeuralNode CLI - AI framework for LLMs',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--provider', '-p', help='LLM provider')
    chat_parser.add_argument('--model', '-m', help='Model name')
    chat_parser.add_argument('--stream', '-s', action='store_true', help='Stream responses')
    chat_parser.add_argument('--browser', action='store_true', help='Enable browser')
    chat_parser.add_argument('--web', '-w', action='store_true', help='Enable web search')
    chat_parser.set_defaults(func=chat_command)
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single prompt')
    run_parser.add_argument('prompt', nargs='?', help='Prompt to run')
    run_parser.add_argument('--file', '-f', help='Read prompt from file')
    run_parser.add_argument('--provider', '-p', help='LLM provider')
    run_parser.add_argument('--model', '-m', help='Model name')
    run_parser.add_argument('--json', action='store_true', help='Output as JSON')
    run_parser.set_defaults(func=run_command)
    
    # Agent command
    agent_parser = subparsers.add_parser('agent', help='Run agent with tools')
    agent_parser.add_argument('task', nargs='?', help='Task for agent')
    agent_parser.add_argument('--file', '-f', help='Read task from file')
    agent_parser.add_argument('--provider', '-p', help='LLM provider')
    agent_parser.add_argument('--model', '-m', help='Model name')
    agent_parser.add_argument('--web', '-w', action='store_true', help='Enable web search')
    agent_parser.add_argument('--browser', '-b', action='store_true', help='Enable browser')
    agent_parser.set_defaults(func=agent_command)
    
    # Models command
    models_parser = subparsers.add_parser('models', help='Model management')
    models_parser.add_argument('--list', '-l', action='store_true', help='List available models')
    models_parser.add_argument('--local', action='store_true', help='List local models')
    models_parser.add_argument('--download', '-d', help='Download a model')
    models_parser.add_argument('--quantize', '-q', help='Quantize a model')
    models_parser.add_argument('--method', help='Quantization method (Q4_K_M, Q5_K_M, etc.)')
    models_parser.set_defaults(func=models_command)
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitoring dashboard')
    monitor_parser.add_argument('--dashboard', '-d', action='store_true', help='Show dashboard')
    monitor_parser.add_argument('--export', '-e', help='Export dashboard data to file')
    monitor_parser.set_defaults(func=monitor_command)
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
