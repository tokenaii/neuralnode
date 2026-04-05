"""Example: Basic chat with NeuralNode."""

import neuralnode as nn


def main():
    """Basic chat example."""
    # Initialize with auto-selected provider
    # Prefers configured supported cloud providers, then falls back to Ollama
    print("Initializing NeuralNode...")
    ai = nn.NeuralNode(provider="auto")
    
    print(f"Using provider: {ai.provider}")
    print(f"Using model: {ai.model}")
    print()
    
    # Simple chat
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = ai.chat(user_input)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
