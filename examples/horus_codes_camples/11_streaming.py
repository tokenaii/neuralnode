"""
Streaming responses (real-time tokens)
"""

import neuralnode as nn

model = nn.HorusModel("tokenaii/horus").load()

# Stream tokens as they are generated
print("Response: ", end="", flush=True)
for token in model.stream_chat([{"role": "user", "content": "Count from 1 to 5"}]):
    print(token, end="", flush=True)
print()
