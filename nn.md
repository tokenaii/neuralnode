# NeuralNode AI Framework

**NeuralNode** is a next-generation Python framework for building, running, and managing **Large Language Models (LLMs)** and AI Agents, locally or via cloud providers. It is designed as a **smarter, simpler, and more powerful alternative to LangChain**, solving its major limitations while adding advanced capabilities for real-time AI interactions.

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Advantages over LangChain](#advantages-over-langchain)  
4. [Supported Providers and Models](#supported-providers-and-models)  
5. [LLM Management & Runtime Features](#llm-management--runtime-features)  
6. [AI Agents & Browser Integration](#ai-agents--browser-integration)  
7. [Real-Time Time & Date Access](#real-time-time--date-access)  
8. [Future Enhancements](#future-enhancements)  
9. [Architecture Overview](#architecture-overview)

---

## Overview

NeuralNode aims to simplify AI development by providing a **unified, modular, and highly flexible framework** for:

- Local LLM deployment  
- Cloud-based LLM integration  
- Advanced agent creation  
- Tool integration  
- Real-time interactions  

Unlike existing frameworks, NeuralNode prioritizes **speed, simplicity, and modularity** while maintaining full extensibility for developers and researchers.

---

## Key Features

### 1. Unified LLM Interface
- One API for **all LLMs**, local or cloud
- Supports **chat, completion, embeddings, code generation, vision**
- Automatic provider selection (cloud vs local)

```python
from neuralnode import NeuralNode

ai = NeuralNode(provider="auto")
ai.chat("Explain Neural Networks")
```

### 2. Multi-Provider Support
- OpenAI, Anthropic, Google, Ollama, HuggingFace, and more
- Users can switch providers seamlessly without changing code
- Provider-specific optimizations handled automatically

### 3. Local AI Engine
- Runs any model fully locally (CPU/GPU supported)
- Automatic model download, installation, and quantization
- Detects hardware and recommends the best model

```python
ai.install("qwen2.5")  # Auto downloads and configures model
ai.local_chat("Hello")  # Runs fully offline
```

### 4. Agent Capabilities
- Convert any LLM into a fully autonomous AI Agent
- Supports decision-making, tool usage, memory
- Can interact with web browsers, APIs, local tools

### 5. Tool & Browser Integration
- Connect LLMs to Chrome, Edge, or other browsers
- Automate tasks like web search, scraping, or navigation
- Supports JavaScript execution, form filling, and more

```python
ai.agent("search latest bitcoin price")
```

### 6. Real-Time Context Awareness
- Automatically provides current time and date to the model
- No additional coding required
- Useful for scheduling, reminders, or time-sensitive reasoning

### 7. RAG (Retrieval-Augmented Generation)
- Built-in support for PDFs, documents, and knowledge bases
- Embeddings, vector stores, and semantic search fully integrated

### 8. Advanced Prompt & Memory Management
- Conversation memory for multi-turn chat
- Automatic prompt formatting
- Context-aware chaining of tools and models

### 9. Hardware & Performance Optimization
- Detects CPU/GPU, RAM, and VRAM
- Automatically selects best model configuration (quantization, batch size, precision)
- Benchmarks models for latency and throughput

### 10. Modular & Extensible Architecture
- Easily add new providers, tools, embeddings, or agents
- Lightweight core, avoids LangChain over-engineering

---

## Advantages over LangChain

| LangChain Limitation | NeuralNode Solution |
| :--- | :--- |
| Over-engineered; too many layers for simple tasks | Minimal, unified API: `ai.chat("Hello")` |
| Slow due to deep abstraction layers | Direct API calls, optimized runtime |
| Complex to learn; 50+ concepts | 5 core concepts: Provider, Model, Agent, Tool, Memory |
| Debugging difficult | Clear errors and logging system |
| No real-time system access | Built-in time & date injection |
| Limited local AI support | Full local model support with auto installation & quantization |
| Partial provider support | Supports OpenAI, Anthropic, Google, HuggingFace, Ollama, and any future provider |
| Browser automation requires custom coding | Built-in browser and tool integration |
| Agent conversion is manual | Any LLM can be instantly converted into a fully autonomous local agent |

---

## Supported Providers and Models

- **Cloud Providers:** OpenAI, Anthropic, Google Gemini
- **Local Providers/Models:** Ollama, LLaMA (via llama.cpp), Qwen, HuggingFace Transformers
- **Embeddings:** OpenAI, HuggingFace, local models
- **Vision:** OpenAI Vision, HuggingFace Diffusion, local Vision models
- **Auto Provider Selection:** `provider="auto"` chooses cloud or local based on available hardware and API keys

---

## LLM Management & Runtime Features

- Automatic model download & caching
- Quantization (Q4/Q5/Q8)
- Batch processing & multi-turn chat
- Embeddings, code generation, and summarization
- GPU optimization and memory-efficient execution
- Real-time vector search for RAG applications
- Chain multiple models or tools without custom chaining code

---

## AI Agents & Browser Integration

**Agents can:**
- Open browser tabs in Chrome/Edge
- Execute search queries
- Scrape information
- Fill forms & interact with websites
- Connect to APIs and local tools

**Easy one-line agent creation:**
```python
ai.agent("Book a flight from NYC to London next week")
```
- Supports multi-agent orchestration

---

## Real-Time Time & Date Access

Every LLM invocation automatically receives:
- Current time
- Current date
- Timezone awareness

**Example:**
```python
ai.chat("Schedule a meeting 2 hours from now")
# Model automatically knows the current time & timezone
```

---

## Future Enhancements

- Voice integration (text-to-speech / speech-to-text)
- Multi-modal LLMs with vision & audio
- Built-in model benchmarking and recommendation engine
- Enhanced agent orchestration for large workflows

---

## Architecture Overview

```text
neuralnode/
│
├─ core/
│   └─ ai.py                # Main API interface
│
├─ providers/
│   ├─ openai.py
│   ├─ anthropic.py
│   ├─ google.py
│   ├─ ollama.py
│   └─ huggingface.py
│
├─ local/
│   ├─ hardware.py           # GPU/CPU detection
│   ├─ installer.py          # Auto model download
│   └─ runtime.py            # llama.cpp, Ollama runtime
│
├─ agents/
│   └─ agent.py              # Full AI Agent logic
│
├─ tools/
│   └─ browser_integration.py
│
└─ rag/
    └─ vectorstore.py
```

---

## Summary

NeuralNode is designed to be:
- **Simpler than LangChain**
- **More powerful & modular**
- **Fully local & cloud-ready**
- **Agent-first design**
- **Real-time aware**
- **Provider-agnostic**

It turns any LLM into a fully autonomous AI Agent running locally, with easy integration to browsers, tools, and real-world data sources. NeuralNode solves the main pain points of existing frameworks while staying flexible for research, development, and production use.
