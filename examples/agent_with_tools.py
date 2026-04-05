"""Example: Agent with web search and file system tools."""

import neuralnode as nn
from pathlib import Path


def main():
    """Agent with tools example."""
    print("NeuralNode Agent Example")
    print("=" * 50)
    
    # Initialize NeuralNode
    ai = nn.NeuralNode(provider="auto")
    print(f"Connected to: {ai.provider} ({ai.model})")
    print()
    
    # Create tools
    tools = [
        nn.WebSearch(),           # Search the web
        nn.FileReader(),          # Read files
        nn.FileWriter(),          # Write files
        nn.FileManager(),         # List directories
        nn.SystemInfo(),          # Get system info
    ]
    
    # Create agent
    agent = ai.agent(
        tools=tools,
        system_prompt="""You are a helpful AI assistant with access to tools.
When the user asks you to search, use the web_search tool.
When they ask about files, use the file tools.
Always explain what you're doing.""",
        react=True  # Use ReAct reasoning
    )
    
    print("Agent created with tools:")
    print("  - WebSearch: Search the internet")
    print("  - FileReader: Read files")
    print("  - FileWriter: Write files")
    print("  - FileManager: Browse directories")
    print("  - SystemInfo: Get system information")
    print()
    
    # Example tasks
    tasks = [
        "Search for the latest Python version and tell me about its features",
        "What files are in the current directory?",
        "Get system information",
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\nTask {i}: {task}")
        print("-" * 50)
        
        try:
            result = agent.run(task)
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("You can now enter your own tasks (or 'quit' to exit):")
    
    while True:
        task = input("\nTask: ").strip()
        
        if task.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not task:
            continue
        
        try:
            result = agent.run(task)
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
