"""
Production-ready complete setup
"""

import neuralnode as nn

# Full configuration for production
model = nn.HorusModel(
    model_id="tokenaii/horus",
    
    # Device & Precision
    device="cuda",
    torch_dtype="bfloat16",
    
    # Memory Optimization
    load_in_4bit=False,
    load_in_8bit=False,
    
    # Multi-GPU / Offloading
    device_map="auto",
    max_memory={0: "20GiB", "cpu": "30GiB"},
    
    # Attention
    use_flash_attention=True,
    
    # Loading Options
    low_cpu_mem_usage=True,
    use_safetensors=True,
    trust_remote_code=True,
    
    # Caching
    cache_dir="./model_cache",
    
    # Generation Defaults
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
).load()

response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
