"""
Use GGUF 16-bit model (full precision)
File size: 9.83 GB
Best for: Maximum quality, same as original model
"""

import neuralnode as nn

model = nn.HorusModel(
    "Horus-1.0-4B-F16.gguf"
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
