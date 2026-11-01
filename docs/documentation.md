# NeuralNode Documentation

This document reflects the current public API and shipped functionality in `neuralnode` v2.2.9.

## Overview

NeuralNode provides:
- a unified provider layer for supported cloud and local LLMs
- agents and reasoning helpers
- memory and RAG
- Horus local model support
- Replica TTS and speech recognition
- Telegram integration for text, voice, and documents
- optional TurboQuant KV-cache compression on supported Transformers flows

## Installation

Base:

```bash
pip install neuralnode
```

Recommended feature extras:

```bash
pip install "neuralnode[all]"
pip install "neuralnode[horus]"
pip install "neuralnode[telegram]"
pip install "neuralnode[replica]"
pip install "neuralnode[speech]"
pip install "neuralnode[turboquant]"
```

Important runtime dependencies:
- `llama-cpp-python` for GGUF local models and GGUF Horus
- `huggingface_hub` for Horus downloads from Hugging Face
- `edge-tts` for Replica
- `SpeechRecognition` and `pydub` for voice transcription
- `pypdf` and `python-docx` for document ingestion

## Public Provider Registry

The public registry only exposes implemented providers.

Cloud chat providers:
- `anthropic`
- `google`
- `cohere`
- `mistral`
- `groq`
- `deepseek`
- `perplexity`
- `ai21`
- `together`
- `fireworks`
- `bedrock`
- `vertexai`

Local providers:
- `ollama`
- `llamacpp`
- `transformers`
- `llamafile`
- `koboldcpp`
- `textgenwebui`
- `exllama`
- `autogptq`
- `autoawq`
- `vllm`
- `tgi`
- `deepspeed`
- `rayserve`
- `mlflow`
- `bentoml`
- `triton`
- `mlx`
- `horus`

## NeuralNode API

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="anthropic",
    model="claude-3-haiku-20240307",
    api_key="YOUR_API_KEY",
)

print(ai.chat("Hello"))
```

## Horus

Horus is exposed through:
- `nn.HorusModel`
- `nn.HorusProvider`
- `nn.load_horus`

Current Horus guarantees:
- all Horus models use context window `8192`
- chat template is unified as `horus_unified`
- Horus can use Replica TTS
- Horus can use TurboQuant on Transformers-based generation paths
- Horus downloads are public-only

Example:

```python
import neuralnode as nn

model = nn.HorusModel(
    model_id="Horus-1.0-4B",
    turboquant=True,
    turboquant_bits=4,
    enable_tts=True,
    tts_voice_id="replic-aria-language{en-us}",
).load()

response = model.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize Horus in one paragraph."},
])

print(response.content)
```

Available Horus model IDs:
- `horus`
- `Horus-1.0-4B`
- `Horus-1.0-4B-Q4_K_M.gguf`
- `Horus-1.0-4B-Q5_K_M.gguf`
- `Horus-1.0-4B-Q6_K.gguf`
- `Horus-1.0-4B-Q8_0.gguf`
- `Horus-1.0-4B-F16.gguf`
- `Horus-Lens-1.0`
- `Horus-Hiero-9B`
- `Horus-Hiero-9B-Q4_K_M.gguf`
- `Horus-Hiero-9B-Q6_K.gguf`
- `Horus-Hiero-Mini-4B`
- `Horus-Hiero-Mini-4B-Q4_K_M.gguf`
- `Horus-Hiero-Mini-4B-Q6_K.gguf`

Horus Lens text-to-image example:

```python
import neuralnode as nn

model = nn.HorusLensModel("Horus-Lens-1.0").load()
model.generate_image(
    "A detailed cinematic image of an ancient Egyptian AI lab, golden light",
    output_path="outputs/horus_lens.png",
    seed=42,
)
```

## Replica

Replica is the dedicated `edge_tts` wrapper for NeuralNode.

```python
import neuralnode as nn

for voice in nn.replica_voice_list():
    print(voice["id"], "->", voice["edge_voice_id"])
```

The public IDs are custom Replica voice IDs, not raw `edge_tts` IDs.

Voice mapping reference:
- [replica_voice_ids.md](./replica_voice_ids.md)
- [replica_voice_ids.csv](./replica_voice_ids.csv)

## Telegram

Telegram integration is available through:
- `nn.TelegramBot`
- `nn.TelegramBotConfig`
- `nn.TelegramBotBuilder`
- `nn.create_telegram_bot`

Current Telegram support:
- text chat
- session persistence
- voice message download and transcription
- optional Replica voice replies
- document download and analysis

Example:

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="horus", model="Horus-1.0-4B")
agent = ai.agent(agent_type="simple", thinking=False)

bot = nn.TelegramBot(
    token="YOUR_BOTFATHER_TOKEN",
    agent=agent,
    config=nn.TelegramBotConfig(
        token="YOUR_BOTFATHER_TOKEN",
        enable_voice=True,
        enable_documents=True,
        reply_mode="both",
        voice_reply_voice_id="replic-salma-language{ar-eg}",
    ),
)

bot.start()
```

## RAG

RAG automatically handles documents and embedding fallbacks.

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="ollama", model="llama3.2")
rag = ai.rag(store="memory")
rag.add_documents(["guide.pdf", "notes.txt"])

print(rag.query("What are the main points?"))
```

Embedding resolution order:
1. current LLM if it supports embeddings
2. sentence-transformers provider
3. lexical fallback embeddings

## TurboQuant

TurboQuant is available through:
- `nn.TurboQuant`
- `nn.TurboQuantConfig`
- `nn.TurboQuantEngine`

It is integrated into:
- `HorusModel(..., turboquant=True, turboquant_bits=4)`
- `TransformersProvider(..., turboquant=True, turboquant_bits=4)`

Example:

```python
import neuralnode as nn

llm = nn.create_provider(
    "transformers",
    model="Qwen/Qwen2.5-3B-Instruct",
    turboquant=True,
    turboquant_bits=4,
)
```

If the runtime cannot create a TurboQuant cache, NeuralNode falls back to standard generation automatically.

## Notes

- OpenAI-compatible provider support is available when the optional OpenAI SDK
  is installed and the provider credentials/configuration are supplied.
- Some advanced historical documentation from older versions has been removed because it did not match the shipped implementation.
- `get_available_providers()` now reflects only the providers that are actually implemented in the current package surface.
