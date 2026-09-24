"""
Force re-download model
"""

import neuralnode as nn

# Force download even if cached
model = nn.HorusModel(
    "horus",
    force_download=True,
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
