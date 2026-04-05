"""
Flash Attention 2 for faster inference
"""

import neuralnode as nn

# Enable Flash Attention 2
model = nn.HorusModel(
    model_id="tokenaii/horus",
    device="cuda",
    use_flash_attention=True,
    torch_dtype="float16",
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
