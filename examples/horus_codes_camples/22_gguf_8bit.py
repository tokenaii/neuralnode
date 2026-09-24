"""
Use GGUF 8-bit quantized model
Best for: Good quality with ~50% memory savings
"""

import neuralnode as nn

model = nn.HorusModel(
    "Horus-1.0-4B-Q8_0.gguf"
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
