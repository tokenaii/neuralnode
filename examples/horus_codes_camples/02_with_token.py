"""
Download with HuggingFace token for private models
"""

import neuralnode as nn

# Method 1: Pass token directly
model = nn.HorusModel(
    model_id="your-model",
    token="hf_xxxxxxxxxxxxxxxxxxxx"
).load()

# Method 2: Use environment variable
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"
model = nn.HorusModel("your-model").load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
