"""
Get model information
"""

import neuralnode as nn

model = nn.HorusModel(
    "horus",
    load_in_4bit=True
).load()

# Get model info
info = model.get_model_info()

print(f"Model ID: {info['id']}")
print(f"Device: {info['device']}")
print(f"Dtype: {info['torch_dtype']}")
print(f"Quantization: {info['quantization']}")
print(f"Context Window: {info.get('context_window')}")

# List available models
models = nn.HorusModel.list_available_models()
for model_id in models:
    print(f"  - {model_id}")
