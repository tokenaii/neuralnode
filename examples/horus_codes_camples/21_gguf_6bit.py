"""
Use GGUF 6-bit quantized model
Best for: Better quality than 4/5-bit with reasonable memory
"""

import neuralnode as nn

model = nn.HorusModel(
    "Horus-1.0-4B-Q6_K.gguf"
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
