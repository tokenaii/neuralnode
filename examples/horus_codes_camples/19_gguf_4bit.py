"""
Use GGUF 4-bit quantized model (lowest memory)
Best for: Low VRAM GPUs (4-6GB), fastest inference
"""

import neuralnode as nn

# Load GGUF 4-bit model (~2.78GB file, ~2GB VRAM)
model = nn.HorusModel(
    "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
