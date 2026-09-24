"""
4-bit quantization for low VRAM GPUs
"""

import neuralnode as nn

# Load with 4-bit quantization (~75% memory savings)
model = nn.HorusModel(
    model_id="horus",
    device="cuda",
    load_in_4bit=True,
    torch_dtype="float16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
).load()

print("Model loaded with 4-bit quantization")

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
