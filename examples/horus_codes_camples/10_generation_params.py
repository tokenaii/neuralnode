"""
Generation parameters control
"""

import neuralnode as nn

model = nn.HorusModel("horus").load()

# Default generation
response1 = model.chat([{"role": "user", "content": "Tell me a story"}])

# Creative generation (high temperature)
response2 = model.chat(
    [{"role": "user", "content": "Tell me a story"}],
    temperature=0.9,
    top_p=0.95,
    max_new_tokens=1024,
)

# Focused generation (low temperature)
response3 = model.chat(
    [{"role": "user", "content": "What is 2+2?"}],
    temperature=0.1,
    do_sample=False,
)

# Beam search
response4 = model.chat(
    [{"role": "user", "content": "Best solution?"}],
    num_beams=4,
    early_stopping=True,
)

print(response1["content"])
