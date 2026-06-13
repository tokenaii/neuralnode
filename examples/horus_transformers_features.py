"""
Horus Model - Complete Transformers Features Example
=====================================================

This example demonstrates ALL transformers configuration options available
in NeuralNode's HorusProvider - matching the full capability of using
transformers directly.

Installation:
    pip install neuralnode[horus]
    
Or for full local model support:
    pip install neuralnode[local-all]
"""

import neuralnode as nn


def example_1_basic_usage():
    """Basic usage - simplest form."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    model = nn.HorusModel("tokenaii/horus").load()
    
    response = model.chat([{
        "role": "user",
        "content": "What is artificial intelligence?"
    }])
    
    print(f"Response: {response.content[:200]}...")
    print()


def example_2_device_options():
    """Device configuration options."""
    print("=" * 70)
    print("Example 2: Device Options")
    print("=" * 70)
    
    # Auto device selection (GPU if available, else CPU)
    model_auto = nn.HorusModel("tokenaii/horus").load()
    
    # Force CPU
    model_cpu = nn.HorusModel(
        "tokenaii/horus",
        device="cpu"
    ).load()
    
    # Force specific GPU
    model_gpu0 = nn.HorusModel(
        "tokenaii/horus",
        device="cuda:0"
    ).load()
    
    print("[OK] Device options demonstrated")
    print()


def example_3_precision_dtypes():
    """Different precision/data types."""
    print("=" * 70)
    print("Example 3: Precision / Data Types")
    print("=" * 70)
    
    # Float16 (half precision) - fastest on modern GPUs
    model_fp16 = nn.HorusModel(
        "tokenaii/horus",
        torch_dtype="float16"  # or torch.float16
    ).load()
    
    # BFloat16 (better for training, supported on Ampere+)
    model_bf16 = nn.HorusModel(
        "tokenaii/horus",
        torch_dtype="bfloat16"
    ).load()
    
    # Float32 (full precision, most accurate)
    model_fp32 = nn.HorusModel(
        "tokenaii/horus",
        torch_dtype="float32"
    ).load()
    
    # Auto (let transformers decide)
    model_auto = nn.HorusModel(
        "tokenaii/horus",
        torch_dtype="auto"
    ).load()
    
    print("[OK] Precision options demonstrated")
    print()


def example_4_quantization():
    """Quantization for memory efficiency."""
    print("=" * 70)
    print("Example 4: Quantization (Memory Efficient Loading)")
    print("=" * 70)
    
    # 8-bit quantization (~50% memory savings)
    model_8bit = nn.HorusModel(
        "tokenaii/horus",
        load_in_8bit=True,
        device="cuda"
    ).load()
    
    # 4-bit quantization (~75% memory savings)
    model_4bit = nn.HorusModel(
        "tokenaii/horus",
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",  # "nf4" or "fp4"
        bnb_4bit_use_double_quant=True,  # Nested quantization
    ).load()
    
    print("[OK] Quantization options demonstrated")
    print("   8-bit: ~50% memory savings")
    print("   4-bit: ~75% memory savings")
    print()


def example_5_device_mapping():
    """Multi-GPU and offloading configurations."""
    print("=" * 70)
    print("Example 5: Device Mapping (Multi-GPU / Offloading)")
    print("=" * 70)
    
    # Auto device mapping (transformers decides optimal distribution)
    model_auto_map = nn.HorusModel(
        "tokenaii/horus",
        device_map="auto",
        load_in_8bit=True
    ).load()
    
    # Balanced distribution across GPUs
    model_balanced = nn.HorusModel(
        "tokenaii/horus",
        device_map="balanced",
        max_memory={0: "8GiB", 1: "8GiB"}  # Per GPU limit
    ).load()
    
    # Custom device mapping (specific layers to specific devices)
    model_custom = nn.HorusModel(
        "tokenaii/horus",
        device_map={
            "transformer.word_embeddings": 0,
            "transformer.layers.0": 0,
            "transformer.layers.1": 1,
            "transformer.layers.2": "cpu",  # Offload to CPU
        }
    ).load()
    
    # CPU offloading for large models
    model_offload = nn.HorusModel(
        "tokenaii/horus",
        device_map="auto",
        offload_folder="offload_weights",  # Disk offloading
        offload_state_dict=True,
        max_memory={0: "6GiB", "cpu": "20GiB"}
    ).load()
    
    print("[OK] Device mapping options demonstrated")
    print()


def example_6_attention_options():
    """Attention implementation options."""
    print("=" * 70)
    print("Example 6: Attention Implementation")
    print("=" * 70)
    
    # Flash Attention 2 (fastest, requires flash-attn package)
    model_flash = nn.HorusModel(
        "tokenaii/horus",
        use_flash_attention=True,  # Uses "flash_attention_2"
        # attn_implementation="flash_attention_2"  # Same effect
    ).load()
    
    # Scaled Dot Product Attention (PyTorch native, good default)
    model_sdpa = nn.HorusModel(
        "tokenaii/horus",
        attn_implementation="sdpa"  # "sdpa", "eager", "flash_attention_2"
    ).load()
    
    # Eager attention (standard, most compatible)
    model_eager = nn.HorusModel(
        "tokenaii/horus",
        attn_implementation="eager"
    ).load()
    
    print("[OK] Attention options demonstrated")
    print("   flash_attention_2: Fastest if available")
    print("   sdpa: PyTorch native, good default")
    print("   eager: Most compatible")
    print()


def example_7_advanced_loading():
    """Advanced model loading options."""
    print("=" * 70)
    print("Example 7: Advanced Loading Options")
    print("=" * 70)
    
    # Specific model variant
    model_variant = nn.HorusModel(
        "tokenaii/horus",
        variant="fp16",  # "fp16", "bf16", etc.
    ).load()
    
    # Specific revision/branch
    model_revision = nn.HorusModel(
        "tokenaii/horus",
        revision="main",  # or specific commit hash
        subfolder="model_files",  # If model is in subfolder
    ).load()
    
    # Use safetensors (recommended, faster loading)
    model_safetensors = nn.HorusModel(
        "tokenaii/horus",
        use_safetensors=True,  # Default True
    ).load()
    
    # Low CPU memory loading (for large models)
    model_low_mem = nn.HorusModel(
        "tokenaii/horus",
        low_cpu_mem_usage=True,  # Default True
        # Prevents loading full model into RAM before moving to GPU
    ).load()
    
    # Custom cache directory
    model_cache = nn.HorusModel(
        "tokenaii/horus",
        cache_dir="/path/to/custom/cache",
    ).load()
    
    # Local files only (offline mode)
    model_offline = nn.HorusModel(
        "tokenaii/horus",
        local_files_only=True,  # Use only cached files
    ).load()
    
    print("[OK] Advanced loading options demonstrated")
    print()


def example_8_generation_params():
    """Generation parameters - equivalent to transformers .generate()."""
    print("=" * 70)
    print("Example 8: Generation Parameters")
    print("=" * 70)
    
    # Model with default generation settings
    model = nn.HorusModel(
        "tokenaii/horus",
        # Default generation parameters
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=1,
        early_stopping=False,
    ).load()
    
    # Override per-call
    response = model.chat(
        [{"role": "user", "content": "Write a story"}],
        max_new_tokens=1024,  # Longer for this call
        temperature=0.9,       # More creative
        top_p=0.95,
        repetition_penalty=1.2,  # Less repetition
    )
    
    # Beam search
    response = model.chat(
        [{"role": "user", "content": "Best solution?"}],
        num_beams=4,           # Beam search with 4 beams
        early_stopping=True,
        max_new_tokens=256,
    )
    
    # Greedy decoding (no sampling)
    response = model.chat(
        [{"role": "user", "content": "Fact-based question"}],
        do_sample=False,     # Greedy, deterministic
        temperature=0.0,
    )
    
    print("[OK] Generation parameters demonstrated")
    print()


def example_9_network_options():
    """Network and download options."""
    print("=" * 70)
    print("Example 9: Network & Download Options")
    print("=" * 70)

    # With proxy
    model_proxy = nn.HorusModel(
        "tokenaii/horus",
        proxies={
            "http": "http://proxy.example.com:8080",
            "https": "https://proxy.example.com:8080",
        }
    ).load()
    
    # Force re-download
    model_force = nn.HorusModel(
        "tokenaii/horus",
        force_download=True,  # Ignore cache
    ).load()
    
    # Resume interrupted download
    model_resume = nn.HorusModel(
        "tokenaii/horus",
        resume_download=True,
    ).load()
    
    print("[OK] Network and download options demonstrated")
    print()


def example_10_model_info():
    """Getting model information."""
    print("=" * 70)
    print("Example 10: Model Information")
    print("=" * 70)
    
    model = nn.HorusModel("tokenaii/horus", load_in_4bit=True).load()
    
    # Get comprehensive model info
    info = model.get_model_info()
    
    print("Model Information:")
    print(f"  ID: {info['id']}")
    print(f"  Name: {info.get('name', 'N/A')}")
    print(f"  Size: {info.get('size', 'N/A')}")
    print(f"  Device: {info['device']}")
    print(f"  Device Map: {info.get('device_map', 'N/A')}")
    print(f"  Dtype: {info['torch_dtype']}")
    print(f"  Quantization: {info['quantization']}")
    print(f"  Flash Attention: {info['flash_attention']}")
    print(f"  Attention Impl: {info.get('attn_implementation', 'N/A')}")
    print(f"  Safetensors: {info.get('use_safetensors', 'N/A')}")
    print(f"  Low CPU Mem: {info.get('low_cpu_mem_usage', 'N/A')}")
    print(f"  Generation Config: {info.get('generation_config', {})}")
    print()


def example_11_list_models():
    """List available Horus models."""
    print("=" * 70)
    print("Example 11: List Available Models")
    print("=" * 70)
    
    models = nn.HorusModel.list_available_models()
    
    print("Available Horus Models:")
    for model_id, info in models.items():
        print(f"  - {model_id}")
        print(f"    Name: {info['name']}")
        print(f"    Size: {info['size']}")
        print(f"    Context: {info['context_window']} tokens")
    print()


def example_12_chat_templates():
    """Using different chat templates."""
    print("=" * 70)
    print("Example 12: Chat Templates")
    print("=" * 70)
    
    model = nn.HorusModel("tokenaii/horus").load()
    
    # List available templates
    print("Available templates:", list(model.CHAT_TEMPLATES.keys()))
    
    # Use default template
    model.set_chat_template("default")
    response1 = model.complete("Explain quantum computing")
    
    # Use instruct template
    model.set_chat_template("instruct")
    response2 = model.complete("Explain quantum computing")
    
    # Use chat template
    model.set_chat_template("chat")
    response3 = model.complete("Explain quantum computing")
    
    print("[OK] Chat templates demonstrated")
    print()


def example_13_streaming():
    """Streaming responses."""
    print("=" * 70)
    print("Example 13: Streaming Responses")
    print("=" * 70)
    
    model = nn.HorusModel("tokenaii/horus").load()
    
    print("Streaming response: ", end="", flush=True)
    for token in model.stream_chat([{
        "role": "user",
        "content": "Count from 1 to 5"
    }]):
        print(token, end="", flush=True)
    print()
    print()


def example_14_complete_setup():
    """Complete production-ready setup."""
    print("=" * 70)
    print("Example 14: Complete Production Setup")
    print("=" * 70)
    
    # Full configuration for production use
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        
        # Device & Precision
        device="cuda",
        torch_dtype="bfloat16",
        
        # Memory Optimization
        load_in_4bit=False,  # Set True for low VRAM
        load_in_8bit=False,  # Set True for medium VRAM
        
        # Multi-GPU / Offloading
        device_map="auto",
        max_memory={0: "20GiB", "cpu": "30GiB"},
        
        # Attention
        use_flash_attention=True,
        # attn_implementation="flash_attention_2",
        
        # Loading Options
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
        
        # Caching
        cache_dir="./model_cache",
        
        # Generation Defaults
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
    ).load()
    
    print("[OK] Production setup complete!")
    print(f"Model info: {model.get_model_info()}")
    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("  Horus Model - Complete Transformers Features".center(66))
    print("=" * 70)
    print()
    
    print("This example demonstrates ALL transformers features available in")
    print("NeuralNode's HorusProvider. Each example shows different configuration")
    print("options you can use to optimize model loading and generation.")
    print()
    
    try:
        # Run examples (commented to avoid actually loading models)
        # Uncomment the examples you want to run
        
        example_1_basic_usage()
        # example_2_device_options()
        # example_3_precision_dtypes()
        # example_4_quantization()
        # example_5_device_mapping()
        # example_6_attention_options()
        # example_7_advanced_loading()
        # example_8_generation_params()
        # example_9_network_options()
        # example_10_model_info()
        example_11_list_models()
        # example_12_chat_templates()
        # example_13_streaming()
        # example_14_complete_setup()
        
        print("=" * 70)
        print("[DONE] All examples completed!")
        print("=" * 70)
        print()
        print("To use Horus models, install with:")
        print("  pip install neuralnode[horus]")
        print()
        print("For all local model features:")
        print("  pip install neuralnode[local-all]")
        
    except ImportError as e:
        print(f"\n[X] Error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install neuralnode[horus]")
        print("  or: pip install transformers torch accelerate bitsandbytes")
        
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
