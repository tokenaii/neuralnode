"""
8-bit quantization for medium VRAM GPUs
"""

import neuralnode as nn

# Load with 8-bit quantization (~50% memory savings)
model = nn.HorusModel(
    model_id="tokenaii/horus",
    device="cuda",
    load_in_8bit=True,
).load()

print("Model loaded with 8-bit quantization")

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
