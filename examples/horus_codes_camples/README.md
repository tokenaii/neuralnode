# Horus Models - Code Samples

This folder contains code examples for downloading and using Horus models.

## Available Models

| Model ID | File Size | Type | Best For |
|----------|-----------|------|----------|
| `Horus-1.0-4B` | ~10 GB | Safetensors | Maximum accuracy, GPU with 8GB+ VRAM |
| `Horus-1.0-4B-Q4_K_M.gguf` | 2.78 GB | GGUF 4-bit | Low VRAM (4-6GB), fastest inference |
| `Horus-1.0-4B-Q5_K_M.gguf` | 3.23 GB | GGUF 5-bit | Balance between quality and memory |
| `Horus-1.0-4B-Q6_K.gguf` | 3.71 GB | GGUF 6-bit | Better quality with reasonable memory |
| `Horus-1.0-4B-Q8_0.gguf` | 4.8 GB | GGUF 8-bit | Good quality with ~50% memory savings |
| `Horus-1.0-4B-F16.gguf` | 9.83 GB | GGUF 16-bit | Maximum quality, same as original |
| `Horus-Lens-1.0` | Unknown | Text-to-image | Image generation from prompts |
| `Horus-Hiero-9B-Q4_K_M.gguf` | 5.63 GB | GGUF 4-bit | Hieroglyphs and multilingual tasks |
| `Horus-Hiero-Mini-4B-Q4_K_M.gguf` | 2.71 GB | GGUF 4-bit | Smaller Hiero model |

## Installation

```bash
pip install neuralnode[horus]
```

## Quick Start

```python
import neuralnode as nn

model = nn.HorusModel("Horus-1.0-4B-Q4_K_M.gguf").load()
response = model.chat([{"role": "user", "content": "Hello!"}])
print(response.content)
```

## Example Files

### Basic Usage
- `01_basic_usage.py` - Simple download and use
- `02_horus_lens.py` - Generate an image with Horus Lens
- `03_one_liner.py` - One-liner usage
- `04_custom_cache.py` - Custom download location

### Memory Options
- `05_4bit_quantization.py` - 4-bit bitsandbytes (transformers)
- `06_8bit_quantization.py` - 8-bit bitsandbytes (transformers)
- `19_gguf_4bit.py` - GGUF 4-bit (llama.cpp)
- `20_gguf_5bit.py` - GGUF 5-bit (llama.cpp)
- `21_gguf_6bit.py` - GGUF 6-bit (llama.cpp)
- `22_gguf_8bit.py` - GGUF 8-bit (llama.cpp)
- `23_gguf_16bit.py` - GGUF 16-bit (llama.cpp)

### Advanced Features
- `07_multi_gpu.py` - Multi-GPU setup
- `08_flash_attention.py` - Flash Attention 2
- `09_data_types.py` - Float16, BFloat16, Float32
- `10_generation_params.py` - Temperature, top_p, beams
- `11_streaming.py` - Real-time token streaming
- `12_chat_templates.py` - Different prompt formats

### Network & Cache
- `13_offline_mode.py` - Use cached files only
- `14_force_download.py` - Re-download model

### Info & Debugging
- `15_model_info.py` - Get model information
- `16_cpu_offloading.py` - Disk offloading
- `17_cpu_only.py` - CPU inference
- `18_production_setup.py` - Full production config
- `24_list_models.py` - List all available models
- `26_horus_hiero.py` - Horus Hiero Mini GGUF usage
