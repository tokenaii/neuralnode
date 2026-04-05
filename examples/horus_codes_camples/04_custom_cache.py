"""
Custom cache directory for model download
"""

import neuralnode as nn

# Download to specific directory
model = nn.HorusModel(
    model_id="tokenaii/horus",
    cache_dir="D:/Models/Horus"
).load()

print(f"Model cached at: D:/Models/Horus")
response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
