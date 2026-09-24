"""
Offline mode (local files only)
"""

import neuralnode as nn

# Use only cached files, no internet
model = nn.HorusModel(
    "horus",
    local_files_only=True,
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
