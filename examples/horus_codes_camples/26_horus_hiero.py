"""
Load and use Horus Hiero Mini 4B GGUF.
"""

import neuralnode as nn


MODEL_ID = "Horus-Hiero-Mini-4B-Q4_K_M.gguf"

model = nn.HorusModel(MODEL_ID, device="cpu").load()

response = model.chat([
    {"role": "user", "content": "Explain what Horus Hiero is in one paragraph."}
])

print(response.content)
