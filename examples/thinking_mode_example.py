"""
Thinking Mode Usage Example for NeuralNode
==========================================

Complete guide to using Thinking Mode with ANY model.
"""

from pathlib import Path

import neuralnode as nn

# Configuration
MODEL_ID = "Horus-1.0-4B-Q4_K_M.gguf"
CACHE_DIR = str(Path.home() / ".cache" / "neuralnode" / "models")


def example_1_basic_thinking():
    """Example 1: Basic Thinking Mode Usage"""
    print("=" * 70)
    print("Example 1: Basic Thinking Mode")
    print("=" * 70)
    
    # Load any model
    model = nn.HorusModel(
        model_id=MODEL_ID,
        device="cpu",
        cache_dir=CACHE_DIR,
    ).load()
    
    # Method 1: Using enable_thinking() function
    thinker = nn.enable_thinking(model, show_thinking=True)
    
    # Ask a question
    response = thinker.chat("What is 15 × 23?")
    
    print(f"\nQuestion: What is 15 × 23?")
    print(f"\n🧠 Thinking Process:\n{response.thinking}")
    print(f"\n✅ Final Answer: {response.content}")
    print(f"📊 Confidence: {response.confidence:.0%}")


def example_2_custom_steps():
    """Example 2: Custom Reasoning Steps"""
    print("\n" + "=" * 70)
    print("Example 2: Custom Reasoning Steps")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id=MODEL_ID,
        device="cpu",
        cache_dir=CACHE_DIR,
    ).load()
    
    # Method 2: Using ThinkingMode class with custom steps
    thinker = nn.ThinkingMode(
        model,
        steps=["analyze", "brainstorm", "evaluate", "conclude"],
        show_thinking=True,
        concise=True,
        max_tokens_per_step=80,
        max_final_tokens=150,
    )
    
    response = thinker.chat("Should I learn Python or JavaScript first?")
    
    print(f"\nQuestion: Should I learn Python or JavaScript first?")
    print(f"\n🔍 Reasoning Steps ({len(response.steps)} steps):")
    for i, step in enumerate(response.steps, 1):
        print(f"  Step {i}: {step[:60]}...")
    print(f"\n✅ Answer: {response.content}")


def example_3_no_cutoff_protection():
    """Example 3: No Cutoff - Long Thinking"""
    print("\n" + "=" * 70)
    print("Example 3: Long Reasoning (No Token Cutoff)")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id=MODEL_ID,
        device="cpu",
        cache_dir=CACHE_DIR,
    ).load()
    
    # Dynamic token calculation prevents cutoff
    thinker = nn.ThinkingMode(
        model,
        steps=["understand", "plan", "execute", "verify", "summarize"],
        max_tokens_per_step=120,  # More tokens per step
        max_final_tokens=300,     # Longer final answer
    )
    
    response = thinker.chat(
        "Explain how photosynthesis works, step by step, in detail."
    )
    
    print(f"\nQuestion: Explain photosynthesis in detail")
    print(f"\n🧠 Thinking ({len(response.thinking)} chars):")
    print(response.thinking[:300] + "...")
    print(f"\n✅ Detailed Answer ({len(response.content)} chars):")
    print(response.content)


def example_4_concise_mode():
    """Example 4: Concise Responses"""
    print("\n" + "=" * 70)
    print("Example 4: Concise Mode (Brief Responses)")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id=MODEL_ID,
        device="cpu",
        cache_dir=CACHE_DIR,
    ).load()
    
    # Enable concise mode for brief answers
    thinker = nn.ThinkingMode(
        model,
        concise=True,  # Brief responses
        max_tokens_per_step=50,
        max_final_tokens=100,
    )
    
    response = thinker.chat("What is the capital of France?")
    
    print(f"\nQuestion: What is the capital of France?")
    print(f"\n✅ Concise Answer: {response.content}")


def example_5_streaming_thinking():
    """Example 5: Streaming with Thinking"""
    print("\n" + "=" * 70)
    print("Example 5: Streaming Response")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id=MODEL_ID,
        device="cpu",
        cache_dir=CACHE_DIR,
    ).load()
    
    thinker = nn.ThinkingMode(model, show_thinking=True)
    
    print(f"\nQuestion: Count from 1 to 5")
    print(f"\n🧠 Thinking: ", end="")
    
    # Stream the response
    full_response = ""
    for chunk in thinker.stream_chat("Count from 1 to 5"):
        if chunk.metadata.get("is_thinking"):
            print(f"[Thinking: {len(chunk.thinking)} chars]")
        else:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content
    
    print(f"\n\n✅ Complete Answer: {full_response}")


def example_6_with_any_provider():
    """Example 6: Works with ANY Provider"""
    print("\n" + "=" * 70)
    print("Example 6: Thinking Mode with Any Model")
    print("=" * 70)
    
    print("""
Thinking Mode works with ANY NeuralNode provider:

1. Local Models:
   model = nn.HorusModel().load()
   thinker = nn.ThinkingMode(model)
   
2. Cloud APIs:
   ai = nn.NeuralNode(provider="anthropic")
   thinker = nn.ThinkingMode(ai._llm)
   
3. Ollama:
   model = nn.create_provider("ollama", model="llama3.2")
   thinker = nn.ThinkingMode(model)
   
4. Transformers:
   model = nn.create_provider("transformers", model="gpt2")
   thinker = nn.ThinkingMode(model)

All work the same way!
    """)


def main():
    """Run all examples."""
    print("\n" + "█" * 70)
    print("██" + " " * 66 + "██")
    print("██" + "  NeuralNode Thinking Mode - Complete Examples".center(66) + "██")
    print("██" + " " * 66 + "██")
    print("█" * 70)
    
    examples = [
        ("Basic Thinking", example_1_basic_thinking),
        ("Custom Steps", example_2_custom_steps),
        ("No Cutoff", example_3_no_cutoff_protection),
        ("Concise Mode", example_4_concise_mode),
        ("Streaming", example_5_streaming_thinking),
        ("Any Provider", example_6_with_any_provider),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run All")
    
    choice = input("\nSelect example (0-6): ").strip()
    
    if choice == "0":
        for name, func in examples:
            if name != "Any Provider":  # Skip the info-only one
                try:
                    func()
                    input("\nPress Enter for next example...")
                except Exception as e:
                    print(f"\nError in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= 6:
        examples[int(choice)-1][1]()
    else:
        print("Invalid choice")
    
    print("\n" + "=" * 70)
    print("Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
