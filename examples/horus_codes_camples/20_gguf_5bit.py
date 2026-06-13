"""
Use GGUF 5-bit quantized model
Best for: Balance between quality and memory (~2.5GB VRAM)
"""

import neuralnode as nn

model = nn.HorusModel(
    "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q5_K_M.gguf"
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
