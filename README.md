<p align="center">
    <img src="https://raw.githubusercontent.com/tokenaillc/neuralnode/v2.2.8/media/neuralnode.png" alt="NeuralNode — Connect models. Build AI." width="720">
</p>

<p align="center">
  <a href="https://pypi.org/project/neuralnode/"><img src="https://img.shields.io/pypi/v/neuralnode.svg?style=flat-square" alt="PyPI version"></a>
  <a href="https://pypi.org/project/neuralnode/"><img src="https://img.shields.io/pypi/pyversions/neuralnode.svg?style=flat-square" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Source--Available-172033.svg?style=flat-square" alt="Source-Available License"></a>
  <a href="https://github.com/tokenaillc/neuralnode/releases"><img src="https://img.shields.io/github/v/release/tokenaillc/neuralnode?display_name=tag&sort=semver&style=flat-square" alt="Latest release"></a>
</p>

<p align="center"><strong>Connect models. Build AI.</strong></p>

<p align="center">NeuralNode <code>v2.2.8</code></p>

NeuralNode is a Python framework for connecting cloud and local AI models to
one consistent interface. Build applications with model providers, agents,
tools, memory, RAG, embeddings, vector stores, and integrations without
rewriting your application around a single provider.

## What's new in 2.2.8

NeuralNode 2.2.8 expands the framework's local-model and multimodal building
blocks while making the public release safer to build and publish.

- Added Hugging Face authentication support for public, private, and gated
  model repositories through the `HF_TOKEN` environment variable or an
  explicit token argument.
- Added local Hugging Face speech adapters for speech-to-text and text-to-speech,
  including lazy loading, automatic CPU/CUDA selection, configurable model IDs,
  language handling, and WAV output.
- Added a sequential `VoiceAgent` workflow that connects audio transcription,
  an agent turn, and speech synthesis for file-based and push-to-talk use cases.
- Added local vision-language input and Diffusers-compatible text-to-image
  generation through dedicated Hugging Face model loaders.
- Improved Horus model loading, provider identity, deterministic generation,
  optional runtime imports, and model repository download behavior.
- Improved provider-backed agent tool calling, including textual actions and
  tool-result handling.
- Added and aligned OpenClaw integration behavior, including safer local API
  access and test coverage that does not require optional HTTP test packages.
- Hardened the release pipeline with source and wheel build validation,
  isolated build output, and trusted publishing to PyPI through GitHub Actions.

For the complete release record, see [CHANGELOG.md](CHANGELOG.md).

## Why NeuralNode?

- **One interface for many models** — connect cloud APIs, local runtimes, and
  custom providers through a unified interface.
- **Production building blocks** — agents, tool calling, memory, RAG,
  embeddings, vector stores, and API gateways.
- **Local and hosted execution** — use Transformers, Ollama, llama.cpp, vLLM,
  Horus, and other supported runtimes.
- **Composable integrations** — connect applications to Telegram, Slack,
  Discord, OpenClaw, speech, vision, and text-to-speech workflows.

## Quickstart

```bash
pip install neuralnode
```

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="groq",
    model="llama-3.1-70b-versatile",
    api_key="YOUR_GROQ_API_KEY",
)

print(ai.chat("Hello from NeuralNode"))
```

## Install optional integrations

```bash
pip install "neuralnode[all]"
pip install "neuralnode[horus]"
pip install "neuralnode[image]"
pip install "neuralnode[vision]"
pip install "neuralnode[telegram]"
pip install "neuralnode[replica]"
pip install "neuralnode[speech]"
pip install "neuralnode[turboquant]"
```

## Core capabilities

- Chat and text-generation providers
- Agents and structured tool calling
- Conversation and persistent memory
- Retrieval-Augmented Generation (RAG)
- Embeddings and vector stores
- Vision and speech-to-text
- Text-to-speech and Replica integrations
- OpenAI-compatible local and hosted gateways
- Provider registry and custom provider extensions

### Hugging Face text-to-image

Install the optional image dependencies and load a Diffusers-compatible
text-to-image repository. `AutoPipelineForText2Image` supports compatible
Diffusers model repositories; it does not make unrelated Hugging Face models
image generators.

```bash
pip install "neuralnode[image]"
```

For local image input with a Transformers vision-language model, install the
`vision` extra (`pip install "neuralnode[vision]"`) and use the existing
`chat_with_image(...)` interface with a local image file.

```python
from neuralnode import load_hf_image_model

image_model = load_hf_image_model("stabilityai/stable-diffusion-xl-base-1.0")
image = image_model.generate_image(
    "A small robot reading under a tree, watercolor illustration",
    output_path="generated/robot.png",
    num_inference_steps=25,
    seed=42,
)
```

### One-turn voice agent

Connect speech recognition, an agent, and speech synthesis in one sequential
turn. This is suitable for file-based or push-to-talk workflows; it does not
provide low-latency, full-duplex streaming audio.

```python
from neuralnode.speech import HTTPSTT
from neuralnode.tts import HTTPTTS
from neuralnode.voice import VoiceAgent

voice_agent = VoiceAgent(
    agent=ai.agent(tools=[...]),
    speech=HTTPSTT("http://localhost:5010"),
    tts=HTTPTTS("http://localhost:5012", voice="sara", language="en"),
)
turn = voice_agent.process_audio_file("question.wav", output_audio="answer.mp3")
print(turn.transcript)
print(turn.response)
```

For local Hugging Face speech models, install `pip install "neuralnode[hf-audio]"`.
The default models are `openai/whisper-small` for transcription and
`facebook/mms-tts-eng` for English speech generation. Pass another compatible
public or accessible Hugging Face model ID to either adapter.

```python
from neuralnode.speech import HuggingFaceSTT
from neuralnode.tts import HuggingFaceTTS

stt = HuggingFaceSTT("openai/whisper-small")
text = stt.transcribe("question.wav", language="en-US")

tts = HuggingFaceTTS("facebook/mms-tts-eng")
tts.speak(text, output="recognized-speech.wav")
```

These adapters load models lazily and use CPU unless CUDA is available. Public
models do not need a token; gated models require `HF_TOKEN` set in the process
environment. `HuggingFaceTTS.speak()` writes mono PCM WAV output.

## Documentation

- [Model API deployment](docs/model_api.md)
- [Examples](examples/)
- [Contributing](CONTRIBUTING.md)
- [License and ownership](LICENSE)

## Horus

Locally loaded Horus models use:
- unified chat template: `horus_unified`
- unified context window: `8192`

Horus downloads are public-only. Private or gated Hugging Face repositories are not supported.

```python
import neuralnode as nn

model = nn.HorusModel(
    model_id="Horus-1.0-4B",
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
- `Horus-Cyber-Nano-1.0`
- `Horus-Cyber-Nano-1.0-Q4_K_M.gguf`
- `Horus-Cyber-Nano-1.0-Q5_K_M.gguf`
- `Horus-Cyber-Nano-1.0-Q6_K.gguf`
- `Horus-Cyber-Nano-1.0-Q8_0.gguf`
- `Horus-Cyber-Nano-1.0-F16.gguf`
- `Horus-Taleeq-0.2B-Base`

### Horus Hiero

```python
import neuralnode as nn

model = nn.HorusModel(
    "Horus-Hiero-Mini-4B-Q4_K_M.gguf",
    device="cpu",
).load()

response = model.chat([
    {"role": "user", "content": "Translate this hieroglyphic text into English."}
])

print(response.content)
```

### Horus Lens text-to-image

```python
import neuralnode as nn

model = nn.HorusLensModel("Horus-Lens-1.0").load()
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
    model_id="Horus-1.0-4B",
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

ai = nn.NeuralNode(provider="horus", model="Horus-1.0-4B")
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

- OpenAI-compatible providers are fully supported (OpenAI, OpenRouter, and AgentRouter).
- GGUF Horus models require `llama-cpp-python`.
- Horus downloads from Hugging Face require `huggingface_hub`.

## OpenClaw Integration

NeuralNode can expose a local OpenAI-compatible API for an OpenClaw Gateway to use. The
`OpenClawServer` serves `/v1/models` and `/v1/chat/completions` without authentication
on loopback by default. Binding to a non-loopback host requires an API key, and clients
must send it as a Bearer token. Browser access is disabled by default; pass explicit
`cors_origins` only when browser clients need access. The `OpenClawManager` generates a
per-instance token automatically and configures its local server to require it.

`OpenClawClient` calls the Gateway's OpenAI-compatible chat endpoint. Enable
`gateway.http.endpoints.chatCompletions` in the OpenClaw configuration before using
`send_task`; OpenClaw disables this endpoint by default. A stable `session_id` is sent
as the OpenAI `user` value, so the Gateway creates or resumes that conversation when
the first message is sent. OpenClaw session status/control uses its WebSocket RPC API
and is not implemented by this HTTP client. Messaging channels such as WhatsApp must
be configured and authorized in OpenClaw itself.

## Links

- **TokenAI**: [https://tokenai.llc/](https://tokenai.llc/)
- **NeuralNode**: [https://tokenai.llc/](https://tokenai.llc/)

## License and ownership

NeuralNode is distributed under the **TokenAI Source-Available License
Version 1.0**. This is a source-available license, not an OSI-approved Open
Source license.

The NeuralNode Framework and the original NeuralNode source code are owned by
**TokenAI**, subject to the separate licenses of third-party libraries, models,
model weights, APIs, datasets, and other external materials. This repository's
license does not transfer ownership of NeuralNode, TokenAI, or their trademarks.

If you use, fork, embed, modify, or build a product on NeuralNode, you must
retain the license and attribution notices and clearly state:

> This product uses NeuralNode, a framework owned by TokenAI.

See [LICENSE](LICENSE), [NOTICE](NOTICE), [TRADEMARKS.md](TRADEMARKS.md), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the applicable terms.

## Contributing

Contributions, bug fixes, tests, and documentation improvements are welcome
through GitHub issues, forks, and pull requests. Every pull request is reviewed
by TokenAI and may be accepted, changed, or rejected. A pull request becomes
part of NeuralNode only after TokenAI merges it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution terms and workflow.

## Hugging Face Open-Weights Support

Any open-weight model on **Hugging Face** can be downloaded and used with NeuralNode (not only TokenAI models). Use the built-in HF helpers:

```python
import neuralnode as nn

# 1) Discover files in a repo
print(nn.list_repo_files("org/model"))  # e.g. ["config.json", "model.safetensors", ...]

# 2) Download the repo (or a specific file)
path = nn.download_repo("org/model")                       # safetensors/auto
path = nn.download_repo("org/model", file="model.gguf")    # GGUF

# 3) Load it into NeuralNode
llm = nn.load_hf_model("org/model", device="auto")
print(llm.chat([{"role": "user", "content": "Hello with any HF model"}]))
```

Supported formats: **GGUF/GGML, SafeTensors, PyTorch (.bin/.pt), ONNX, AWQ, GPTQ**.
Private or gated repos **are** supported by the HF helpers (with `HF_TOKEN` env var).
Python example:

