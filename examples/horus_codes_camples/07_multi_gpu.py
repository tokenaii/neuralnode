"""
Multi-GPU and offloading configuration
"""

import neuralnode as nn

# Auto device mapping for multi-GPU
model = nn.HorusModel(
    model_id="tokenaii/horus",
    device_map="auto",
    max_memory={0: "20GiB", 1: "20GiB", "cpu": "50GiB"},
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
