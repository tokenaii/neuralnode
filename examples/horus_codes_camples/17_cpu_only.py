"""
CPU only mode (no GPU)
"""

import neuralnode as nn

# Force CPU usage
model = nn.HorusModel(
    "horus",
    device="cpu",
    torch_dtype="float32",
    low_cpu_mem_usage=True,
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
