"""
Horus Model - Simple Download & Usage Guide
============================================

This example shows the EASIEST way to download and use Horus models from HuggingFace.

Question: Do I need an access token?
Answer: It depends on the model repository visibility:

1. PUBLIC model (tokenaii/horus) - NO token needed
   The model is publicly available, anyone can download it.

2. PRIVATE/GATED model - YES, token needed
   If the repo is private or requires acceptance, you need a token.

How to get a HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (read access is enough)
3. Use it in the code or set as environment variable
"""

import neuralnode as nn


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Simplest Way - Just Model ID (Public Model)
# ═══════════════════════════════════════════════════════════════════════════════
def example_simple_download():
    """
    For PUBLIC models like tokenaii/horus - NO token needed!
    The model downloads automatically on first use.
    """
    print("=" * 70)
    print("EXAMPLE 1: Simple Download (No Token Needed)")
    print("=" * 70)
    
    # Just provide the model ID - that's it!
    model = nn.HorusModel(model_id="tokenaii/horus").load()
    
    # The model downloads automatically from HuggingFace on first run
    # It will be cached for future uses
    
    print("[OK] Model loaded successfully!")
    print(f"Model: {model.model_id}")
    print(f"Device: {model.device}")
    
    # Use it immediately
    response = model.chat([{"role": "user", "content": "Hello!"}])
    print(f"\nResponse: {response.content}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: With HuggingFace Token (Private/Gated Models)
# ═══════════════════════════════════════════════════════════════════════════════
def example_with_token():
    """
    For PRIVATE or GATED models - Token required!
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Download with Access Token (Private Model)")
    print("=" * 70)
    
    # Method 1: Pass token directly
    model = nn.HorusModel(
        model_id="tokenaii/horus",  # or your private model
        token="hf_xxxxxxxxxxxxxxxxxxxx"  # Your HuggingFace token
    ).load()
    
    # Method 2: Set environment variable (recommended)
    import os
    os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"
    # Then load without passing token
    model = nn.HorusModel(model_id="tokenaii/horus").load()
    
    print("[OK] Model loaded with authentication!")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: One-Liner Download & Use
# ═══════════════════════════════════════════════════════════════════════════════
def example_one_liner():
    """
    Ultra-simple one-liner for quick usage.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: One-Liner (Chain Loading)")
    print("=" * 70)
    
    # Chain the load() call
    response = nn.HorusModel("tokenaii/horus").load().chat(
        [{"role": "user", "content": "What is AI?"}]
    )
    
    print(f"Response: {response.content}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Download with Custom Cache Directory
# ═══════════════════════════════════════════════════════════════════════════════
def example_custom_cache():
    """
    Download to a specific directory (useful for large models).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Cache Directory")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        cache_dir="D:/Models/Horus"  # Where to save the model
    ).load()
    
    print(f"[OK] Model cached at: D:/Models/Horus")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Check if Token is Needed
# ═══════════════════════════════════════════════════════════════════════════════
def example_check_token_needed():
    """
    Check if the model requires authentication.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Check if Token is Required")
    print("=" * 70)
    
    try:
        # Try without token first
        model = nn.HorusModel("tokenaii/horus").load()
        print("[OK] Model is PUBLIC - no token needed!")
    except Exception as e:
        if "401" in str(e) or "token" in str(e).lower():
            print("[X] Model requires authentication!")
            print("   You need to:")
            print("   1. Get token from https://huggingface.co/settings/tokens")
            print("   2. Pass token='your_token' parameter")
        else:
            print(f"[X] Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Memory-Efficient Download (Low VRAM)
# ═══════════════════════════════════════════════════════════════════════════════
def example_memory_efficient():
    """
    Download and load with minimal memory (good for 4GB-6GB GPUs).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Memory-Efficient Loading")
    print("=" * 70)
    
    model = nn.HorusModel(
        model_id="tokenaii/horus",
        device="cuda",           # Use GPU
        load_in_4bit=True,       # 4-bit quantization (saves ~75% memory)
        torch_dtype="float16",   # Half precision
    ).load()
    
    print("[OK] Model loaded with 4-bit quantization (~2-3GB VRAM)")


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Reference Card
# ═══════════════════════════════════════════════════════════════════════════════
"""
┌─────────────────────────────────────────────────────────────────────┐
│                    HORUS MODEL - QUICK REFERENCE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [PUBLIC] PUBLIC MODEL (No token needed):                                 │
│     model = nn.HorusModel("tokenaii/horus").load()                  │
│                                                                     │
│  [PRIVATE] PRIVATE MODEL (Token required):                                  │
│     model = nn.HorusModel(                                          │
│         "your-username/private-model",                              │
│         token="hf_xxxxxxxxxxxxxxxxxxxx"                               │
│     ).load()                                                        │
│                                                                     │
│  [CACHE] With Custom Cache:                                               │
│     model = nn.HorusModel(                                          │
│         "tokenaii/horus",                                            │
│         cache_dir="D:/MyModels"                                     │
│     ).load()                                                        │
│                                                                     │
│  [GPU] GPU with Low Memory (4-bit):                                     │
│     model = nn.HorusModel(                                          │
│         "tokenaii/horus",                                            │
│         device="cuda",                                               │
│         load_in_4bit=True                                           │
│     ).load()                                                        │
│                                                                     │
│  [ENV] Environment Variable (for private repos):                       │
│     import os                                                       │
│     os.environ["HF_TOKEN"] = "your_token_here"                        │
│     model = nn.HorusModel("tokenaii/horus").load()                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("  Horus Model - Download & Usage Guide".center(66))
    print("=" * 70)
    print()
    
    # Print quick reference
    print(__doc__)
    print()
    
    # Run examples
    try:
        example_simple_download()
        # example_with_token()  # Uncomment if you have a private model
        example_one_liner()
        example_custom_cache()
        example_check_token_needed()
        example_memory_efficient()
        
        print("\n" + "=" * 70)
        print("[DONE] All examples completed!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n[X] Error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install transformers torch")
        
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()
