# NeuralNode

**NeuralNode 2.2.9** is a Python framework for connecting cloud and local AI models through one provider interface. It includes chat, streaming, vision, embeddings, agents, memory, RAG, speech, Hugging Face models, Horus models, Ollama, OpenClaw, and optional integrations.

NeuralNode is developed and fully owned by **TokenAI**. It is distributed under the TokenAI Source-Available License described at the end of this file.

## Install

```bash
pip install neuralnode
```

Install only the feature set you need:

```bash
pip install "neuralnode[ollama]"      # Ollama client support
pip install "neuralnode[local]"       # local Transformers/GGUF models
pip install "neuralnode[horus]"       # Horus local models
pip install "neuralnode[replica]"     # Replica TTS adapter
pip install "neuralnode[telegram]"    # Telegram integration
pip install "neuralnode[openclaw]"    # OpenClaw gateway integration
pip install "neuralnode[all]"         # all published optional features
```

The base package stays lightweight. Large ML runtimes, vector stores, speech libraries, and provider SDKs are optional.

## Quick start

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="groq", model="llama-3.1-8b-instant")
print(ai.chat("Explain embeddings in one paragraph."))
```

Set the provider API key before running:

```powershell
$env:GROQ_API_KEY="your-key"
```

## Do this → run this

### See supported providers

```python
import neuralnode as nn

print(nn.get_available_providers())
print(nn.get_provider_info("ollama"))
```

The registry is the source of truth for the installed package. Optional provider SDKs may still be required when creating a provider.

### List the built-in model catalog

```python
import neuralnode as nn

for model_id, info in nn.list_models().items():
    print(model_id, info)
```

### Chat with a local Ollama model

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen3
```

Then:

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="ollama", model="qwen3")
print(ai.chat("Give me three names for a robotics project."))
```

For a remote/containerized Ollama server:

```python
ai = nn.NeuralNode(
    provider="ollama",
    model="qwen3",
    base_url="http://127.0.0.1:11434",
)
```

### Enable or disable local-model reasoning

For Ollama, `thinking` maps to its native `think` field. `None` keeps the model default, `True` enables reasoning, and `False` disables it:

```python
import neuralnode as nn

ai = nn.NeuralNode(provider="ollama", model="qwen3")
print(ai.chat("What is 2 + 2?", thinking=False))
print(ai.chat("Compare two sorting algorithms.", thinking=True))
```

To access the normalized reasoning trace itself:

```python
from neuralnode.providers.local_providers import OllamaProvider

ollama = OllamaProvider(model="qwen3")
response = ollama.chat(
    [{"role": "user", "content": "Explain binary search."}],
    thinking=True,
)
print(response["content"])
print(response["thinking"])
```

Reasoning is model-dependent; a model without native reasoning may return an empty `thinking` field.

### Stream a response

```python
for chunk in ai.chat("Write a short poem about Cairo.", stream=True):
    print(chunk, end="", flush=True)
```

### Send an image to a vision-capable model

```python
ai = nn.NeuralNode(provider="ollama", model="llava")
print(ai.chat_with_image("Describe this image.", image="photo.jpg"))
```

### Create an agent with tools and reasoning

```python
ai = nn.NeuralNode(provider="ollama", model="qwen3")
agent = ai.agent(
    skills=["datetime", "calculator"],
    thinking=True,
    max_iterations=10,
)
print(agent.run("What time is it, and what is 37 * 19?"))
```

Supported skill names include `web_search`, `datetime`, `system_control`, `file`, `python`, `calculator`, and `all`. Use `thinking=False` to execute without the reasoning wrapper.

### Add conversation memory

```python
ai = nn.NeuralNode(provider="ollama", model="qwen3").with_memory()
ai.chat("My project is called Atlas.")
print(ai.chat_with_memory("What is my project called?"))
```

### Generate embeddings

```python
ai = nn.NeuralNode(provider="ollama", model="nomic-embed-text")
vectors = ai.embed(["NeuralNode connects models.", "Ollama runs locally."])
print(len(vectors), len(vectors[0]))
```

### Build a small RAG system

```python
ai = nn.NeuralNode(provider="ollama", model="qwen3")
rag = ai.rag(store="memory", top_k=3)
rag.add_documents(["NeuralNode is owned by TokenAI.", "Ollama is a local runtime."])
print(rag.query("Who owns NeuralNode?"))
```

For persistent Chroma storage:

```bash
pip install "neuralnode[chroma]"
```

```python
rag = ai.rag(store="chroma", persist_dir="./neuralnode-db")
```

### Download or load a Hugging Face model

```python
from neuralnode import download_repo, load_hf_model

path = download_repo("sentence-transformers/all-MiniLM-L6-v2")
model = load_hf_model("sentence-transformers/all-MiniLM-L6-v2")
print(path, model)
```

Private or gated repositories require `HF_TOKEN` or an explicit token. Heavy dependencies are installed with the `local` extra.

### Use Horus models

```bash
pip install "neuralnode[horus]"
```

```python
from neuralnode import HorusModel

model = HorusModel(model_id="TokenAI/your-horus-model")
print(model.chat("Hello from a local Horus model."))
```

### Generate speech with Replica

```bash
pip install "neuralnode[replica]"
```

```python
from neuralnode import ReplicaTTS

tts = ReplicaTTS(voice_id="default")
tts.speak("Welcome to NeuralNode.", output_file="welcome.mp3")
```

### Run Telegram

```bash
pip install "neuralnode[telegram]"
```

```python
from neuralnode import NeuralNode
from neuralnode.integrations.telegram import TelegramBot

ai = NeuralNode(provider="ollama", model="qwen3")
bot = TelegramBot(token="your-token", agent=ai.agent())
bot.run()
```

### Connect NeuralNode to OpenClaw

```bash
pip install "neuralnode[openclaw]"
```

```python
from neuralnode import NeuralNode
from neuralnode.integrations.openclaw import OpenClawServer

ai = NeuralNode(provider="ollama", model="qwen3")
server = OpenClawServer(node=ai, host="127.0.0.1", port=8001, api_key="change-me")
server.run()
```

Keep the gateway on loopback unless a deliberate network and authentication configuration protects it. Do not expose a development gateway publicly without authentication and a reverse proxy policy.

## Provider surface

The registry includes cloud adapters such as OpenAI-compatible, Anthropic, Google, Groq, Cohere, Mistral, DeepSeek, Perplexity, Together, Fireworks, Bedrock, and Vertex AI, plus local adapters including Ollama, llama.cpp, Transformers, vLLM, TGI, MLX, Horus, and other compatible runtimes.

Print the exact installed registry:

```python
import neuralnode as nn
print("\\n".join(nn.get_available_providers()))
```

Provider SDKs and credentials are optional. A provider in the registry does not mean its external service or model is included in this package.

## What's new in 2.2.9

- Removed the deprecated WhatsApp/Twilio integration and legacy gateway.
- Improved OpenClaw configuration, loopback binding, token authentication, OpenAI-compatible chat-completions support, and streaming fallback.
- Upgraded Ollama to native `/api/chat` with role-preserving messages, streaming, structured output, tool calls, and native reasoning control through `thinking=True` or `thinking=False`.
- Updated Ollama embeddings to batch `/api/embed` with a legacy compatibility fallback.
- Added and validated the NeuralNode → OpenClaw → local model path on AWS.
- Retained Hugging Face speech, vision, Horus, Replica, agents, memory, RAG, vector-store, and optional TurboQuant workflows.
- Added reproducible wheel/source builds, CI validation for Python 3.9–3.12, and trusted PyPI publishing through GitHub Actions.

See [CHANGELOG.md](CHANGELOG.md) for the complete release history.

## Coming soon to NeuralNode

- `replica-preview-0.6B` — TTS model.
- `horus-cyber-nano-1.2` — text model with vision capabilities.

These are announced as coming soon and are not shipped weights in NeuralNode 2.2.9.

## Documentation and project files

- [Detailed documentation](docs/documentation.md)
- [Changelog](CHANGELOG.md)
- [License](LICENSE)
- [Contributing guide](CONTRIBUTING.md)

## Support and security

For bugs, include the Python version, NeuralNode version, provider, model, a minimal reproduction, and the full error message with secrets removed. Never commit API keys, Hugging Face tokens, Telegram tokens, or OpenClaw authentication tokens.

## Ownership and license

NeuralNode, its source code, documentation, branding, release artifacts, and maintained integrations are the **exclusive property of TokenAI**. Nothing in this repository transfers ownership or permits redistribution, rebranding, sale, sublicensing, or offering NeuralNode as a competing hosted service without written authorization from TokenAI.

NeuralNode is released under the **TokenAI Source-Available License v1.0** in [LICENSE](LICENSE). This is a proprietary source-available license, not an OSI-approved open-source license. Third-party providers, SDKs, model weights, and external services remain subject to their own licenses and terms.

