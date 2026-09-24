"""
Different data types (precision)
"""

import neuralnode as nn

# Float16 (half precision)
model_fp16 = nn.HorusModel(
    "horus",
    torch_dtype="float16"
).load()

# BFloat16 (better for Ampere GPUs)
model_bf16 = nn.HorusModel(
    "horus",
    torch_dtype="bfloat16"
).load()

# Float32 (full precision)
model_fp32 = nn.HorusModel(
    "horus",
    torch_dtype="float32"
).load()

response = model_fp16.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
