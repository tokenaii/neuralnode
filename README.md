# NeuralNode

A next-generation Python framework for building, running, and managing **Large Language Models (LLMs)** and AI Agents, locally or via cloud providers.

> **The smarter, simpler, and more powerful alternative to LangChain.**

## Features

- **Unified LLM Interface** - One API for all LLMs (local or cloud)
- **Multi-Provider Support** - OpenAI, Anthropic, Google, Ollama, HuggingFace
- **Streaming & Async** - Real-time responses with full async support
- **Function Calling** - Native tool use with structured outputs
- **Local AI Engine** - Auto-download, quantization, hardware optimization
- **AI Agents** - Convert any LLM into an autonomous agent
- **Browser Integration** - Chrome/Edge automation built-in
- **RAG Built-in** - PDFs, documents, vector stores, semantic search
- **Real-Time Context** - Automatic time/date injection
- **Memory Management** - Conversation history, context-aware chaining

## Quick Start

```bash
pip install neuralnode
```

```python
from neuralnode import NeuralNode

# Auto-detect best provider
ai = NeuralNode(provider="auto")

# Simple chat
response = ai.chat("Explain quantum computing")
print(response.text)

# With streaming
for chunk in ai.chat("Tell me a story", stream=True):
    print(chunk.text, end="")

# Async support
response = await ai.achat("Hello!")

# Create an agent
agent = ai.agent()
result = agent.run("Find the latest news about AI")
```

## Installation

```bash
# Base install
pip install neuralnode

# With specific providers
pip install neuralnode[openai,anthropic]

# Full install with all features
pip install neuralnode[all]
```

## Architecture

```
neuralnode/
├── core/           # Base interfaces and types
├── providers/      # LLM provider implementations
├── agents/         # Agent system and orchestration
├── tools/          # Tool integrations (browser, APIs)
├── rag/            # Retrieval-Augmented Generation
├── memory/         # Conversation and context management
├── local/          # Local model runtime and hardware
└── utils/          # Utilities and helpers
```

## License

MIT License - see LICENSE file for details.
