"""
Basic usage - Choose a Horus model version
Available versions with different compression levels:
  - tokenaii/horus/Horus-1.0-4B (Full - 8GB+ VRAM)
  - tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf (4-bit - 2GB VRAM)
  - tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q5_K_M.gguf (5-bit - 2.5GB VRAM)
  - tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q6_K.gguf (6-bit - 3GB VRAM)
  - tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q8_0.gguf (8-bit - 4GB VRAM)
"""

import neuralnode as nn

# Choose your model version (replace with your preferred version)
MODEL_ID = "tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"  # 4-bit for low VRAM

# Download and load
model = nn.HorusModel(MODEL_ID).load()

# Use immediately
response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
