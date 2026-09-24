"""
CPU offloading for large models
"""

import neuralnode as nn

# Offload to disk when GPU memory is full
model = nn.HorusModel(
    "horus",
    device_map="auto",
    offload_folder="./offload_weights",
    offload_state_dict=True,
    max_memory={0: "6GiB", "cpu": "20GiB"},
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
