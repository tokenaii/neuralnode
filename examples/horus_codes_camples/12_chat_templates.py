"""
Chat templates (different formats)
"""

import neuralnode as nn

model = nn.HorusModel("tokenaii/horus").load()

# Use default template
model.set_chat_template("default")
response1 = model.complete("Explain AI")

# Use instruct template
model.set_chat_template("instruct")
response2 = model.complete("Explain AI")

# Use chat template
model.set_chat_template("chat")
response3 = model.complete("Explain AI")

print("Available templates:", list(model.CHAT_TEMPLATES.keys()))
print(response1)
