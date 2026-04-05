"""
One-liner: Download and use Horus model in a single line
Choose your model version before running:
  - tokenaii/horus/Horus-1.0-4B-Q4_K_M.gguf (2GB VRAM)
  - tokenaii/horus/Horus-1.0-4B-Q8_0.gguf (4GB VRAM)
  - tokenaii/horus/Horus-1.0-4B (8GB+ VRAM)
"""

import neuralnode as nn

# One-liner: create model, load it, and chat in a single chain
response = nn.HorusModel("tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf").load().chat(
    [{"role": "user", "content": "What is AI?"}]
)

# Print the response content
print(response.content)
