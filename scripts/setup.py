#!/usr/bin/env python3
"""
NeuralNode Setup Script - One-click installation
Installs all dependencies including dev tools
"""

import subprocess
import sys
import os


def install_package(package_spec):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        return True
    except subprocess.CalledProcessError:
        return False


def setup_neuralnode():
    """Complete NeuralNode setup."""
    print("🚀 Setting up NeuralNode...")
    print("=" * 60)
    
    # Upgrade pip first
    print("\n1. Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Core dependencies
    print("\n2. Installing core dependencies...")
    core_deps = [
        "requests>=2.28.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
    ]
    
    for dep in core_deps:
        print(f"   Installing {dep}...")
        install_package(dep)
    
    # Optional but recommended dependencies
    print("\n3. Installing recommended dependencies...")
    recommended = [
        "faiss-cpu>=1.7.0",  # For vector search
        "chromadb>=0.4.0",   # For vector database
        "sentence-transformers>=2.2.0",  # For embeddings
        "duckduckgo-search>=3.9.0",  # Proper search API
        "pypdf>=3.0.0",      # For PDF processing
        "pillow>=9.0.0",     # For image processing
        "cryptography>=41.0.0",  # For encryption
        "gradio>=3.50.0",    # For GUI
        "torch>=2.0.0",      # For ML
        "transformers>=4.30.0",  # For models
    ]
    
    for dep in recommended:
        print(f"   Installing {dep}...")
        install_package(dep)
    
    # Dev dependencies
    print("\n4. Installing development dependencies...")
    dev_deps = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "ruff>=0.1.0",
        "black>=23.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ]
    
    for dep in dev_deps:
        print(f"   Installing {dep}...")
        install_package(dep)
    
    # Install NeuralNode in editable mode
    print("\n5. Installing NeuralNode in development mode...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", project_root])
        print("   ✅ NeuralNode installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install NeuralNode: {e}")
        return False
    
    # Create necessary directories
    print("\n6. Creating project directories...")
    dirs = [
        os.path.expanduser("~/.neuralnode/cache"),
        os.path.expanduser("~/.neuralnode/models"),
        os.path.expanduser("~/.neuralnode/memory"),
        os.path.expanduser("~/.neuralnode/logs"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   Created: {d}")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! You can now use NeuralNode.")
    print("\nQuick start:")
    print("  import neuralnode as nn")
    print("  ai = nn.NeuralNode(provider='groq', model='llama-3.1-8b-instant')")
    print("  print(ai.chat('Hello!'))")
    print("\nRun tests:")
    print("  pytest tests/ -v")
    print("\nRun linter:")
    print("  ruff check src/")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        setup_neuralnode()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed: {e}")
        sys.exit(1)
