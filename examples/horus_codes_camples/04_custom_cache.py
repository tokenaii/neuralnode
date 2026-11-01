"""Custom cache directory for model download."""

from pathlib import Path

import neuralnode as nn

cache_dir = Path.home() / ".cache" / "neuralnode" / "models"

# Download to specific directory
model = nn.HorusModel(
    model_id="horus",
    cache_dir=str(cache_dir),
).load()

print(f"Model cached at: {cache_dir}")
response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
