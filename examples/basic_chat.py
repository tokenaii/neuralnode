"""
Example: Basic chat with NeuralNode

This example demonstrates the basic usage of NeuralNode
for simple chat interactions with different providers.
"""

from neuralnode import NeuralNode


def main():
    # Example 1: Auto-detect provider
    print("=== Example 1: Auto-detect Provider ===")
    ai = NeuralNode(provider="auto")
    print(f"Using provider: {ai.provider}")
    
    # Example 2: Simple chat
    print("\n=== Example 2: Simple Chat ===")
    response = ai.chat("Hello! What can you help me with?")
    print(f"Response: {response.text}")
    
    # Example 3: Chat with system prompt
    print("\n=== Example 3: Chat with System Prompt ===")
    response = ai.chat(
        "Explain quantum computing",
        system="You are a helpful physics professor. Keep answers simple and engaging."
    )
    print(f"Response: {response.text}")
    
    # Example 4: Streaming response
    print("\n=== Example 4: Streaming Response ===")
    print("Response: ", end="", flush=True)
    for chunk in ai.chat("Write a short poem about AI", stream=True):
        print(chunk.text, end="", flush=True)
    print()
    
    # Example 5: Multi-turn conversation with memory
    print("\n=== Example 5: Multi-turn Conversation ===")
    ai2 = NeuralNode(provider="auto")  # New instance with fresh memory
    
    r1 = ai2.chat("My name is Alice")
    print(f"AI: {r1.text}")
    
    r2 = ai2.chat("What's my name?")
    print(f"AI: {r2.text}")
    
    # Example 6: Using specific providers
    print("\n=== Example 6: Specific Providers ===")
    
    # OpenAI
    # ai_openai = NeuralNode(provider="openai", model="gpt-4")
    
    # Anthropic
    # ai_anthropic = NeuralNode(provider="anthropic", model="claude-3-sonnet-20240229")
    
    # Local Ollama
    # ai_local = NeuralNode(provider="ollama", model="llama3")
    
    print("(Uncomment to try specific providers)")


if __name__ == "__main__":
    main()
