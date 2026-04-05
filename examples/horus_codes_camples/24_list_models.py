"""
List all available Horus models
"""

import neuralnode as nn

# Get all available models
models = nn.HorusModel.list_available_models()

print("=" * 70)
print("Available Horus Models")
print("=" * 70)

for model_id, info in models.items():
    print(f"\nModel ID: {model_id}")
    print(f"  Name: {info['name']}")
    print(f"  Size: {info['size']}")
    print(f"  Type: {info.get('type', 'safetensors')}")
    if 'quantization' in info:
        print(f"  Quantization: {info['quantization']}")
    print(f"  Context Window: {info['context_window']} tokens")

print("\n" + "=" * 70)
print("Usage:")
print("  model = nn.HorusModel('model_id').load()")
print("=" * 70)
