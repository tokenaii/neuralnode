"""Example: Local model management."""

import neuralnode as nn
from neuralnode.providers.local import LocalModelManager


def main():
    """Local model management example."""
    print("NeuralNode Local Model Manager")
    print("=" * 50)
    
    # Create manager
    manager = LocalModelManager()
    
    # List available models
    print("\nAvailable models for download:")
    print("-" * 50)
    
    models = manager.list_available()
    for model in models:
        status = "[INSTALLED]" if manager.is_installed(model.id) else ""
        print(f"  {model.id:<20} {model.name:<25} {model.size_mb:>6}MB {status}")
    
    # List installed models
    print("\n\nInstalled models:")
    print("-" * 50)
    
    installed = manager.list_installed()
    if installed:
        for model in installed:
            print(f"  {model['name']:<30} {model['size_mb']:>8.1f} MB")
            print(f"    Path: {model['path']}")
    else:
        print("  No models installed yet")
    
    # Interactive menu
    print("\n" + "=" * 50)
    print("Options:")
    print("  1. Install a model")
    print("  2. Recommend model for my hardware")
    print("  3. Exit")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        model_id = input("Enter model ID to install: ").strip()
        
        if model_id:
            print(f"\nDownloading {model_id}...")
            print("(This may take several minutes depending on model size)")
            
            try:
                def show_progress(percent, downloaded, total):
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(f"  Progress: {percent:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end='\r')
                
                path = manager.install(model_id, progress_callback=show_progress)
                print(f"\n\nModel installed to: {path}")
            except Exception as e:
                print(f"\nError: {e}")
    
    elif choice == "2":
        print("\nAnalyzing hardware...")
        
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
            print(f"  RAM: {ram_gb:.1f} GB")
            
            # Try to get GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"  VRAM: {vram_gb:.1f} GB")
                else:
                    vram_gb = None
                    print("  GPU: Not available")
            except ImportError:
                vram_gb = None
                print("  GPU: PyTorch not installed")
            
            # Get recommendation
            recommended = manager.recommend_model(
                ram_gb=ram_gb,
                vram_gb=vram_gb,
                priority="balanced"
            )
            
            if recommended:
                print(f"\nRecommended model: {recommended.name}")
                print(f"  Description: {recommended.description}")
                print(f"  Size: {recommended.size_mb} MB")
                print(f"  Tags: {', '.join(recommended.tags)}")
            else:
                print("\nNo models match your hardware")
        
        except ImportError:
            print("  psutil not installed, using defaults")
            recommended = manager.recommend_model(priority="balanced")
            if recommended:
                print(f"\nRecommended model: {recommended.name}")


if __name__ == "__main__":
    main()
