# NeuralNode v2.2.0

NeuralNode is a Python AI framework focused on:
- real cloud providers that are implemented
- local model execution with Transformers, Ollama, llama.cpp, and Horus
- agents, memory, and RAG
- Replica TTS and speech recognition
- Telegram integration

## Install

```bash
pip install neuralnode
pip install "neuralnode[all]"
```

Useful extras:

```bash
pip install "neuralnode[horus]"
pip install "neuralnode[telegram]"
pip install "neuralnode[replica]"
pip install "neuralnode[speech]"
pip install "neuralnode[turboquant]"
```

## Quick Start

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="groq",
    model="llama-3.1-70b-versatile",
    api_key="YOUR_GROQ_API_KEY",
)

print(ai.chat("Hello from NeuralNode"))
```

## Horus

All Horus models use:
- unified chat template: `horus_unified`
- unified context window: `8192`

Horus downloads are public-only. Private or gated Hugging Face repositories are not supported.

```python
import neuralnode as nn

model = nn.HorusModel(
    model_id="tokenaii/horus/Horus-1.0-4B",
    turboquant=True,
    turboquant_bits=4,
).load()

response = model.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain Horus briefly."},
])

print(response.content)
```

### Available Horus model IDs

```python
from neuralnode import HorusModel

print(HorusModel.list_available_models())
```

Supported IDs currently include:
- `tokenaii/horus`
- `tokenaii/horus/Horus-1.0-4B`
- `tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf`
- `tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q5_K_M.gguf`
- `tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q6_K.gguf`
- `tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-Q8_0.gguf`
- `tokenaii/Horus-1.0-4B-GGUF/Horus-1.0-4B-F16.gguf`
- `tokenaii/Horus-Lens-1.0`

### Horus Lens text-to-image

```python
import neuralnode as nn

model = nn.HorusLensModel("tokenaii/Horus-Lens-1.0").load()
model.generate_image(
    "A detailed cinematic image of an ancient Egyptian AI lab, golden light",
    output_path="outputs/horus_lens.png",
    seed=42,
)
```

## Replica TTS

Replica exposes 20 curated `edge_tts` voices through custom voice IDs.

```python
import neuralnode as nn

print(nn.replica_voice_list())

tts = nn.ReplicaTTS(voice_id="replic-salma-language{ar-eg}")
tts.save_to_file("مرحبا من نيورال نود", "reply.mp3")
```

Voice mapping docs:
- [docs/replica_voice_ids.md](docs/replica_voice_ids.md)
- [docs/replica_voice_ids.csv](docs/replica_voice_ids.csv)

## Horus + Replica

```python
import neuralnode as nn

model = nn.HorusModel(
    model_id="tokenaii/horus/Horus-1.0-4B",
    enable_tts=True,
    tts_voice_id="replic-salma-language{ar-eg}",
).load()

result = model.chat_and_speak(
    [{"role": "user", "content": "قل لي جملة ترحيب قصيرة"}],
    output_file="horus_reply.mp3",
)

print(result["response"].content)
print(result["audio_path"])
```

## Telegram

Use a BotFather token to connect an agent to Telegram.

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="horus", model="tokenaii/horus/Horus-1.0-4B")
agent = ai.agent(agent_type="simple", thinking=False)

bot = nn.TelegramBot(
    token="YOUR_BOTFATHER_TOKEN",
    agent=agent,
    config=nn.TelegramBotConfig(
        token="YOUR_BOTFATHER_TOKEN",
        enable_voice=True,
        enable_documents=True,
        reply_mode="both",  # text | voice | both
        voice_reply_voice_id="replic-aria-language{en-us}",
    ),
)

bot.start()
```

Telegram now supports:
- text chat
- voice transcription
- optional Replica voice replies
- document download and analysis

## RAG

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="ollama", model="llama3.2")
rag = ai.rag(store="memory")
rag.add_documents(["notes.txt", "report.pdf"])

print(rag.query("Summarize the key findings"))
```

If the current LLM does not support embeddings, RAG automatically tries:
1. a dedicated embedding provider
2. sentence-transformers
3. lexical fallback embeddings

## Supported Provider Surface

Only implemented providers are exposed through the public provider registry.

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

## TurboQuant

TurboQuant is integrated into:
- `HorusModel(..., turboquant=True, turboquant_bits=4)`
- `create_provider("transformers", turboquant=True, turboquant_bits=4, ...)`

If the `turboquant` package is not installed or the backend cannot use it, NeuralNode falls back to normal generation automatically.

## Notes

- OpenAI is intentionally blocked in NeuralNode.
- GGUF Horus models require `llama-cpp-python`.
- Horus downloads from Hugging Face require `huggingface_hub`.

## OpenClaw Integration

NeuralNode 2.2.0 introduces full end-to-end integration with OpenClaw. You can easily set up, start, and control OpenClaw directly through the framework.

The integration operates by running a local FastAPI server in the background that exposes an OpenAI-compatible API to the OpenClaw Node gateway. This server proxies LLM requests directly to NeuralNode providers, allowing seamless interaction.

It also fully supports the WhatsApp API, allowing users to activate it and bind it to their phone numbers entirely via code and OpenClaw.

## Links

- **TokenAI**: [https://tokenai.cloud/](https://tokenai.cloud/)
- **NeuralNode**: [https://neuralnode.tokenai.cloud/](https://neuralnode.tokenai.cloud/)
