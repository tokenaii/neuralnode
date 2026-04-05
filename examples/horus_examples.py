"""
Horus Model Usage Examples for NeuralNode
=========================================

This file demonstrates how to use the Horus model with NeuralNode.

Horus is a custom AI model trained by the user, available on HuggingFace:
https://huggingface.co/tokenaii/horus

Basic Usage:
    import neuralnode as nn
    
    # Simple loading with default settings
    model = nn.HorusModel().load()
    
    # Chat with the model
    response = model.chat([{"role": "user", "content": "Hello!"}])
    print(response.content)

Advanced Usage:
    # Load with custom settings
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        device="cuda",           # Use GPU
        load_in_8bit=True,        # Save memory with 8-bit quantization
        use_flash_attention=True, # Faster inference
    ).load()
    
    # Stream responses
    for token in model.stream_chat([{"role": "user", "content": "Tell me a story"}]):
        print(token, end="")
"""

import neuralnode as nn


def example_basic_usage():
    """Basic usage example - simplest way to use Horus."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Load the model (downloads from HuggingFace if not cached)
    print("Loading Horus model...")
    model = nn.HorusModel().load()
    
    # Simple chat
    messages = [
        {"role": "user", "content": "What is artificial intelligence?"}
    ]
    
    print("User: What is artificial intelligence?")
    response = model.chat(messages)
    print(f"Horus: {response.content}")
    print()


def example_with_system_prompt():
    """Example with system prompt to customize behavior."""
    print("=" * 60)
    print("Example 2: With System Prompt")
    print("=" * 60)
    
    model = nn.HorusModel().load()
    
    # Chat with system context
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate factorial."}
    ]
    
    response = model.chat(messages)
    print(f"Response: {response.content}")
    print()


def example_advanced_settings():
    """Advanced settings for performance optimization."""
    print("=" * 60)
    print("Example 3: Advanced Settings")
    print("=" * 60)
    
    # Load with GPU and quantization for faster inference
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        device="cuda",                # Use GPU (falls back to CPU if not available)
        torch_dtype="float16",        # Use half precision
        load_in_8bit=True,            # 8-bit quantization (saves VRAM)
        use_flash_attention=True,     # Enable Flash Attention 2
        cache_dir="./model_cache",    # Custom cache directory
    ).load()
    
    print("Model loaded with advanced settings!")
    print(f"Device: {model.device}")
    print(f"Model Info: {model.get_model_info()}")
    print()


def example_streaming():
    """Streaming example for real-time responses."""
    print("=" * 60)
    print("Example 4: Streaming Responses")
    print("=" * 60)
    
    model = nn.HorusModel().load()
    
    messages = [
        {"role": "user", "content": "Count from 1 to 10"}
    ]
    
    print("Response: ", end="", flush=True)
    for token in model.stream_chat(messages):
        print(token, end="", flush=True)
    print()
    print()


def example_custom_chat_template():
    """Using custom chat templates."""
    print("=" * 60)
    print("Example 5: Custom Chat Template")
    print("=" * 60)
    
    model = nn.HorusModel().load()
    
    # List available templates
    print("Available templates:")
    for name, template in model.CHAT_TEMPLATES.items():
        print(f"  - {name}")
    
    # Change template
    model.set_chat_template("instruct")
    
    response = model.complete("Explain neural networks in simple terms")
    print(f"Response: {response.content}")
    print()


def example_text_completion():
    """Text completion example."""
    print("=" * 60)
    print("Example 6: Text Completion")
    print("=" * 60)
    
    model = nn.HorusModel().load()
    
    # Complete a prompt
    prompt = "The future of AI is"
    response = model.complete(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95
    )
    
    print(f"Prompt: {prompt}")
    print(f"Completion: {response.content}")
    print()


def example_model_info():
    """Get model information."""
    print("=" * 60)
    print("Example 7: Model Information")
    print("=" * 60)
    
    # List known Horus models
    print("Known Horus Models:")
    models = nn.HorusModel.list_available_models()
    for model_id, info in models.items():
        print(f"  - {model_id}")
        print(f"    Name: {info['name']}")
        print(f"    Size: {info['size']}")
        print(f"    Context: {info['context_window']} tokens")
    
    # Load and get info
    model = nn.HorusModel().load()
    info = model.get_model_info()
    print(f"\nLoaded Model Info:")
    print(f"  ID: {info['id']}")
    print(f"  Device: {info['device']}")
    print(f"  Loaded: {info['loaded']}")
    print()


def example_with_neuralnode():
    """Using Horus with NeuralNode main interface."""
    print("=" * 60)
    print("Example 8: Using with NeuralNode Interface")
    print("=" * 60)
    
    # Use Horus through NeuralNode main class
    ai = nn.NeuralNode(provider="horus", model="tokenaii/horus")
    
    # Simple chat
    response = ai.chat("What is machine learning?")
    print(f"Response: {response}")
    
    # Create an agent with Horus
    agent = ai.agent(skills=["web_search", "datetime"])
    print("Agent created with Horus model!")
    print()


def example_memory_efficient():
    """Memory-efficient loading for low VRAM systems."""
    print("=" * 60)
    print("Example 9: Memory Efficient Loading")
    print("=" * 60)
    
    # For systems with limited VRAM (4GB-6GB)
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        device="cuda",
        load_in_4bit=True,           # 4-bit quantization (saves most memory)
        torch_dtype="float16",
    ).load()
    
    print("Model loaded with 4-bit quantization!")
    print("This uses ~2-3GB VRAM instead of ~8GB")
    print()


if __name__ == "__main__":
    print("\n")
    print("█" * 60)
    print("██" + " " * 56 + "██")
    print("██" + "  Horus Model - NeuralNode Examples".center(56) + "██")
    print("██" + " " * 56 + "██")
    print("█" * 60)
    print()
    
    # Run examples
    try:
        example_basic_usage()
        example_with_system_prompt()
        example_advanced_settings()
        example_streaming()
        example_custom_chat_template()
        example_text_completion()
        example_model_info()
        example_with_neuralnode()
        example_memory_efficient()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install transformers torch")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
