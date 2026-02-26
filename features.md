# NeuralNode Features Documentation

Complete feature documentation for the NeuralNode AI framework.

---

## Table of Contents

1. [Core Features](#core-features)
2. [Provider Support](#provider-support)
3. [Advanced Features](#advanced-features)
4. [Utility Features](#utility-features)
5. [Future Features](#future-features)

---

## Core Features

### 1. Unified LLM Interface

**Description:** One simple API for all LLM providers, local or cloud.

**Usage:**
```python
from neuralnode import NeuralNode
# or
import nn

ai = nn.NeuralNode(provider="auto")
response = ai.chat("Explain quantum computing")
print(response.text)
```

**Key Points:**
- Automatic provider selection with `provider="auto"`
- Unified interface across OpenAI, Anthropic, Google, Ollama, HuggingFace
- Consistent response format regardless of provider

---

### 2. Multi-Provider Support

**Description:** Seamlessly switch between cloud and local providers.

**Supported Providers:**
| Provider | Type | Models |
|----------|------|--------|
| OpenAI | Cloud | GPT-4, GPT-4 Turbo, GPT-3.5 |
| Anthropic | Cloud | Claude 3 (Opus, Sonnet, Haiku) |
| Google | Cloud | Gemini Pro, Gemini Ultra |
| Ollama | Local | Llama3, Mistral, Qwen, Phi3 |
| HuggingFace | Local | Any transformers model |

**Usage:**
```python
# OpenAI
ai = nn.NeuralNode(provider="openai", model="gpt-4")

# Anthropic
ai = nn.NeuralNode(provider="anthropic", model="claude-3-sonnet-20240229")

# Local Ollama
ai = nn.NeuralNode(provider="ollama", model="llama3")

# HuggingFace local
ai = nn.NeuralNode(provider="huggingface", model="HuggingFaceH4/zephyr-7b-beta")
```

---

### 3. Streaming Responses

**Description:** Real-time token streaming for better UX.

**Usage:**
```python
# Synchronous streaming
for chunk in ai.chat("Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)

# Async streaming
async for chunk in await ai.achat("Tell me a story", stream=True):
    print(chunk.text, end="", flush=True)
```

**Features:**
- Works with all providers that support streaming
- Chunk-based processing
- Token usage tracking even in streaming mode

---

### 4. Async/Await Support

**Description:** Full async support for concurrent operations.

**Usage:**
```python
import asyncio
from nn import NeuralNode

ai = nn.NeuralNode(provider="auto")

# Single async request
response = await ai.achat("Hello!")

# Concurrent requests
questions = ["What is AI?", "What is ML?", "What is DL?"]
tasks = [ai.achat(q) for q in questions]
responses = await asyncio.gather(*tasks)

# Async embeddings
embeddings = await ai.aembed(["text1", "text2"])
```

**Use Cases:**
- High-throughput applications
- Batch processing
- Concurrent API calls
- Non-blocking I/O

---

### 5. Conversation Memory

**Description:** Automatic conversation history management.

**Usage:**
```python
from nn import NeuralNode

ai = nn.NeuralNode(provider="auto")

# Automatically remembers conversation
ai.chat("My name is Alice")
ai.chat("What's my name?")  # Remembers "Alice"

# Clear memory
ai.clear_memory()

# Custom memory settings
from nn.memory import ConversationMemory
memory = ConversationMemory(max_messages=50)
ai = nn.NeuralNode(provider="auto", memory=memory)
```

**Memory Types:**
- `ConversationMemory` - Full history with max limit
- `SlidingWindowMemory` - Keeps only last N exchanges

---

### 6. Structured Output (Pydantic)

**Description:** Type-safe LLM responses using Pydantic models.

**Usage:**
```python
from pydantic import BaseModel, Field
from typing import List

class Person(BaseModel):
    name: str
    age: int
    skills: List[str]

person = ai.structured_output(
    "Extract info: John is 28, knows Python and AI",
    output_schema=Person
)
print(person.name)  # "John"
print(person.age)   # 28

# Async version
person = await ai.astructured_output("...", output_schema=Person)
```

**Benefits:**
- Type safety and validation
- IDE autocomplete
- Automatic error handling
- JSON parsing built-in

---

## Provider Support

### 7. Chat Templates

**Description:** Proper prompt formatting for different model families.

**Usage:**
```python
from nn.utils.chat_templates import apply_chat_template, get_stop_sequences
from nn.core.types import Message, MessageRole

messages = [
    Message(role=MessageRole.SYSTEM, content="You are helpful"),
    Message(role=MessageRole.USER, content="Hello")
]

# Format for Llama 3
prompt = apply_chat_template(messages, "llama3")

# Format for Qwen
prompt = apply_chat_template(messages, "qwen")

# Get stop sequences
stop_tokens = get_stop_sequences("llama3")
```

**Supported Templates:**
- Llama 3 (`llama3`, `llama3.1`)
- Qwen (`qwen`, `qwen2`, `qwen2.5`)
- Mistral (`mistral`, `mixtral`)
- Zephyr
- Phi-3 (`phi`, `phi3`)
- Gemma
- ChatML (OpenChat, Starling, Neural-Chat, Dolphin)

---

### 8. Hardware Detection

**Description:** Automatic hardware detection for optimal local model configuration.

**Usage:**
```python
from nn.local import HardwareDetector

hw = HardwareDetector()

# Check capabilities
print(hw.has_cuda)           # True if CUDA available
print(hw.vram_gb)           # GPU VRAM in GB
print(hw.ram_gb)            # System RAM in GB
print(hw.cpu_count)         # CPU core count

# Get recommendations
config = hw.recommend_model_config()
# Returns: device, quantization, batch_size, context_length, n_gpu_layers

# Check if can run model
if hw.can_run_model("70b"):
    print("Can run 70B model")
```

**Features:**
- CUDA detection via PyTorch or NVML
- Automatic quantization recommendations
- GPU VRAM calculation
- Model size compatibility checking

---

## Advanced Features

### 9. AI Agents with Tools

**Description:** Convert any LLM into an autonomous agent with tool use.

**Usage:**
```python
from nn import NeuralNode, Agent, Tool, create_tool

ai = nn.NeuralNode(provider="auto")

# Method 1: Create tool manually
calculator = Tool(
    name="calculator",
    description="Perform calculations",
    parameters={"expression": {"type": "string"}},
    handler=lambda expression: eval(expression)
)

# Method 2: Use decorator
@create_tool(
    name="get_weather",
    description="Get weather for a city",
    parameters={"city": {"type": "string"}}
)
def get_weather(city: str) -> str:
    return f"Weather in {city}: 72°F (mock)"

# Create agent
agent = ai.agent(tools=[calculator, get_weather])

# Run agent
result = agent.run("Calculate 15 * 23 and get weather in Paris")
print(result)
```

**Agent Features:**
- Tool calling with automatic parameter extraction
- Multi-step reasoning
- Error handling
- Max iteration limits
- Verbose mode for debugging

---

### 10. RAG (Retrieval-Augmented Generation)

**Description:** Built-in RAG with document loading and vector search.

**Usage:**
```python
from nn import NeuralNode
from nn.rag import RAG, Document, FileLoader

ai = nn.NeuralNode(provider="auto")

# Create RAG system
def embed_fn(text: str) -> list[float]:
    # Your embedding function
    return ai.embed(text)[0]

rag = RAG(llm=ai, embedding_function=embed_fn)

# Add documents
rag.add_documents([
    Document(content="NeuralNode is a Python framework..."),
    Document(content="It supports multiple LLM providers...")
])

# Query
answer = rag.query("What is NeuralNode?")
print(answer)

# Load files
from nn.rag import CSVLoader, MarkdownLoader, HTMLLoader, DirectoryLoader

# Load CSV
csv_docs = CSVLoader("data.csv", content_columns=["text"]).load()

# Load Markdown (splits by headers)
md_docs = MarkdownLoader("readme.md", split_headers=True).load()

# Load HTML
html_docs = HTMLLoader("page.html").load()

# Load entire directory
dir_docs = DirectoryLoader("./docs", glob_pattern="**/*.md").load()
```

**RAG Components:**
- `SimpleVectorStore` - In-memory vector storage with cosine similarity
- `TextSplitter` - Chunk documents with overlap
- `FileLoader` - Load PDF, web, text
- `CSVLoader` - Load CSV files
- `MarkdownLoader` - Parse markdown with header splitting
- `HTMLLoader` - Extract text from HTML
- `JSONLoader` - Load JSON data
- `DirectoryLoader` - Batch load directories

---

### 11. Chains / Pipelines

**Description:** Composable workflows for complex LLM operations.

**Usage:**
```python
from nn import Chain, Pipeline, PromptTemplate, LLMChain

# Method 1: Manual chain construction
prompt = PromptTemplate.from_template("Translate to French: {text}")
transform = lambda x: f"Result: {x}"

chain = Chain([prompt, transform])
result = chain.run("Hello world")

# Method 2: Using | operator
from nn import TransformChain

chain = (
    PromptTemplate.from_template("Summarize: {text}")
    | TransformChain(lambda x: x.upper())
)

# Method 3: Pipeline builder
ai = nn.NeuralNode(provider="auto")

pipe = Pipeline(llm=ai)
pipe.add_prompt("Analyze sentiment: {text}")
pipe.add_llm(system_prompt="You are a sentiment analyzer")
pipe.add_transform(lambda x: x.text.strip())

result = pipe.run("I love this product!")

# Conditional chains
from nn import ConditionalChain

conditional = ConditionalChain(
    condition=lambda x: "error" in x.lower(),
    if_true=error_handler_chain,
    if_false=success_chain
)

# Parallel execution
from nn import ParallelChain

parallel = ParallelChain([
    translate_to_french,
    translate_to_spanish,
    translate_to_german
])
results = await parallel.arun("Hello")  # All 3 simultaneously
```

**Chain Components:**
- `PromptTemplate` - Variable substitution
- `LLMChain` - LLM calls
- `TransformChain` - Custom transformations
- `ConditionalChain` - Branching logic
- `ParallelChain` - Concurrent execution

---

## Utility Features

### 12. Output Parsers

**Description:** Convert LLM text output to structured data.

**Usage:**
```python
from nn.utils.output_parsers import (
    JSONOutputParser,
    ListOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    BooleanOutputParser,
    MarkdownOutputParser,
    parse_json,
    parse_list,
)

# JSON parsing
parser = JSONOutputParser()
data = parser.parse('{"name": "Alice", "age": 30}')

# List parsing
parser = ListOutputParser()
items = parser.parse("1. Apple\\n2. Banana\\n3. Cherry")
# Returns: ["Apple", "Banana", "Cherry"]

# Pydantic parsing
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(Person)
person = parser.parse("Name: Bob, Age: 25")

# Comma-separated
csv_parser = CommaSeparatedListOutputParser()
items = csv_parser.parse("apple, banana, cherry")

# Boolean parsing
bool_parser = BooleanOutputParser()
result = bool_parser.parse("yes")  # True

# Markdown parsing
md_parser = MarkdownOutputParser()
sections = md_parser.parse("# Title\\nContent\\n## Section\\nMore")
# Returns: {"title": "Content", "section": "More"}

# Convenience functions
data = parse_json('{\"key\": \"value\"}')
items = parse_list("item1\\nitem2")
```

---

### 13. Caching Layer

**Description:** Save costs by caching LLM responses.

**Usage:**
```python
from nn.utils.cache import (
    InMemoryCache,
    DiskCache,
    SemanticCache,
    cached,
    get_default_cache,
)

# In-memory cache
cache = InMemoryCache(max_size=1000)

# Store with TTL
cache.set("key", llm_response, ttl=3600)  # 1 hour

# Retrieve
cached = cache.get("key")
if cached:
    print("Cache hit!")

# Disk cache (persists across sessions)
disk_cache = DiskCache(cache_dir=".nn_cache")
disk_cache.set("key", response)

# Semantic cache (similarity-based)
def embed_fn(text: str) -> list[float]:
    return ai.embed(text)[0]

semantic_cache = SemanticCache(
    embedding_function=embed_fn,
    similarity_threshold=0.95
)

# Cache decorator
@cached(ttl=3600)
def get_summary(text: str):
    return ai.chat(f"Summarize: {text}")

# Use default cache
response = get_summary("Long text...")  # Cached for 1 hour
```

**Cache Types:**
- `InMemoryCache` - Fast, ephemeral
- `DiskCache` - Persistent, survives restarts
- `SemanticCache` - Matches similar queries

---

### 14. Rate Limiting & Retries

**Description:** Production-ready reliability with automatic retries.

**Usage:**
```python
from nn.utils.retry import (
    with_retry,
    with_rate_limit,
    RetryConfig,
    RateLimiter,
    AdaptiveRateLimiter,
)

# Retry decorator
@with_retry(max_attempts=3, min_wait=1, max_wait=10)
def call_api():
    return requests.get("https://api.example.com")

# Rate limit decorator
@with_rate_limit(max_requests=60, time_window=60)
def call_api():
    return requests.get("https://api.example.com")

# Custom retry config
config = RetryConfig(
    max_attempts=5,
    min_wait=2,
    max_wait=60,
    exponential=True,
    retryable_exceptions=(ConnectionError, TimeoutError)
)

# Manual rate limiting
limiter = RateLimiter(max_requests=100, time_window=60)

if limiter.acquire():
    make_request()
else:
    wait_time = limiter.wait_time()
    time.sleep(wait_time)

# Adaptive rate limiting
adaptive = AdaptiveRateLimiter(initial_rate=100)
# Auto-adjusts based on API responses
```

---

### 15. Time Context Injection

**Description:** Automatic real-time awareness for LLMs.

**Usage:**
```python
from nn.utils.time_context import inject_time_context, get_current_time_info

# Automatic (enabled by default)
ai = nn.NeuralNode(provider="auto", auto_time=True)
ai.chat("What time is it?")  # LLM knows current time

# Manual injection
time_context = inject_time_context()
print(time_context)
# Output: Current Date: 2024-01-15, Current Time: 14:30:00, Day: Monday...

# With custom prompt
custom = inject_time_context("You are a scheduling assistant")

# Get time info
time_info = get_current_time_info()
print(time_info["current_date"])
print(time_info["current_time"])
print(time_info["day_of_week"])
```

---

### 16. Tool Definitions

**Description:** Structured tool definitions for function calling.

**Usage:**
```python
from nn.core.types import ToolDefinition

tool = ToolDefinition(
    name="search",
    description="Search the web",
    parameters={
        "query": {
            "type": "string",
            "description": "Search query"
        },
        "num_results": {
            "type": "integer",
            "description": "Number of results"
        }
    },
    required=["query"],
    handler=lambda query, num_results=5: web_search(query, num_results)
)

# Convert to OpenAI format
openai_format = tool.to_openai_format()
```

---

## Future Features

Features planned for future releases:

### v0.2.0 (Short Term)

#### 17. File Watcher
```python
from nn.tools.file_watcher import FileWatcher, watch_directory

# Watch for file changes
watcher = FileWatcher()
watcher.watch("/path/to/directory")

def on_file_change(event):
    print(f"File {event.path} was {event.event_type.value}")

watcher.on_change(on_file_change)
watcher.start()
```

#### 18. Email Tool
```python
from nn.tools.email_tool import GmailClient, EmailClient

# Send email via Gmail
client = GmailClient(
    username="your@gmail.com",
    password="app_password"
)

client.send_email(
    to="recipient@example.com",
    subject="Hello",
    body="Message content"
)

# Receive emails
emails = client.receive_emails(unread_only=True, limit=10)
```

#### 19. Calendar Integration
```python
from nn.tools.calendar_integration import GoogleCalendarClient, OutlookCalendarClient

# Google Calendar
cal = GoogleCalendarClient(credentials={"token": "your_token"})

# List events
events = cal.list_events(start_date=datetime.now(), days=7)

# Create event
event_id = cal.create_event(CalendarEvent(
    title="Meeting",
    start_time=datetime(2024, 1, 15, 10, 0),
    end_time=datetime(2024, 1, 15, 11, 0),
    description="Team meeting"
))
```

#### 20. Universal Tools (Thinking & Verified Search)
```python
from nn.tools.universal_tools import ThinkingTool, VerifiedSearchTool

# Force LLM to think
thinker = ThinkingTool(llm)
result = thinker.think("Complex question")
print(result.thought_process)
print(result.final_answer)

# Search verified sources
searcher = VerifiedSearchTool()
results = searcher.search_verified(
    "query",
    sources=["wikipedia", "arxiv"],
    min_trust_score=0.8
)
```

#### 18. Browser Automation
```python
from nn.tools import BrowserTools

with BrowserTools() as browser:
    browser.navigate("https://example.com")
    text = browser.get_page_text()
    browser.fill_form({"username": "user", "password": "pass"})
```

#### 19. Voice Integration
```python
# Text-to-Speech
ai.speak("Hello, how can I help you?")

# Speech-to-Text
text = ai.listen()
response = ai.chat(text)
```

#### 20. Multi-Modal Support
```python
# Vision
response = ai.chat("Describe this image", image="photo.jpg")

# Audio
response = ai.chat("Transcribe this", audio="recording.mp3")
```

---

### v0.3.0 (Medium Term) - COMPLETED

#### 21. Agent Orchestrator
```python
from nn.agents.orchestrator import AgentOrchestrator, AgentConfig, AgentRole

# Create multi-agent system
orchestrator = AgentOrchestrator()

# Add specialized agents
orchestrator.add_agent(AgentConfig(
    id="researcher",
    name="Web Researcher",
    role=AgentRole.RESEARCHER,
    llm_provider="openai",
    model="gpt-4",
))

orchestrator.add_agent(AgentConfig(
    id="writer",
    name="Content Writer",
    role=AgentRole.WRITER,
    llm_provider="openai",
    model="gpt-4",
))

# Execute workflow
result = orchestrator.run_collaborative_task(
    "Research AI trends and write a report",
    agent_roles=[AgentRole.RESEARCHER, AgentRole.WRITER]
)
```

#### 22. Code Interpreter
```python
from nn.tools.code_interpreter import CodeInterpreter, SandboxedInterpreter

# Execute Python safely
interpreter = CodeInterpreter(timeout=30)

result = interpreter.execute("""
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
print(df.sum())
""")

print(result.output)
print(result.variables)
```

#### 23. Model Benchmarking
```python
from nn.training.benchmark import ModelBenchmark, quick_benchmark

# Compare models
benchmark = ModelBenchmark()

results = benchmark.run_benchmark(
    model_configs=[
        {"name": "gpt-4", "provider": "openai", "model": "gpt-4"},
        {"name": "claude-3", "provider": "anthropic", "model": "claude-3-opus"},
    ],
    tasks=["reasoning", "coding", "creativity"]
)

# Generate report
benchmark.report(results, output_file="benchmark_report.txt")
```

#### 24. Prompt Registry
```python
from nn.prompts.registry import PromptRegistry, get_builtin_prompt

# Register custom prompt
registry = PromptRegistry()
prompt = registry.register(
    name="code-review",
    template="Review this {language} code: {code}",
    description="Code review prompt",
    tags=["coding", "review"]
)

# Use prompt
formatted = prompt.format(language="python", code="def foo(): pass")

# Load built-in
react_prompt = get_builtin_prompt("zero-shot-react")
```

#### 25. Reinforcement Learning
```python
from nn.training.rl_system import RLHFTrainer, SimpleRLAgent, AdaptiveAgent

# Create RL agent
rl_agent = SimpleRLAgent(llm, policy=Policy(strategy="epsilon_greedy", epsilon=0.1))

# Train with feedback
for experience in training_data:
    rl_agent.learn(experience)

# Or use RLHF
trainer = RLHFTrainer(base_llm)
trainer.collect_preference(prompt, response_a, response_b, preferred="A")
trainer.train_with_dpo()
```
```python
from nn.rag import PersistentVectorStore

# ChromaDB, Pinecone, Weaviate support
store = PersistentVectorStore(backend="chroma", path="./chroma_db")
rag = RAG(llm=ai, vector_store=store)
```

#### 22. Agent Orchestration
```python
from nn.agents import AgentOrchestrator

# Multi-agent workflows
orchestrator = AgentOrchestrator()
orchestrator.add_agent("researcher", researcher_agent)
orchestrator.add_agent("writer", writer_agent)
orchestrator.add_agent("reviewer", reviewer_agent)

result = orchestrator.run("Research and write about AI")
```

#### 23. Fine-tuning Support
```python
from nn.finetune import FineTuner

# Prepare dataset
tuner = FineTuner(base_model="gpt-3.5-turbo")
tuner.add_example("prompt", "completion")
tuner.train(output_model="my-custom-model")

ai = nn.NeuralNode(provider="openai", model="my-custom-model")
```

#### 24. Model Benchmarking
```python
from nn.benchmark import ModelBenchmark

benchmark = ModelBenchmark()
results = benchmark.compare(
    models=["gpt-4", "claude-3", "llama3"],
    tasks=["coding", "reasoning", "creativity"]
)
benchmark.report(results)
```

---

### v0.4.0+ (Long Term)

#### 25. Distributed Inference
```python
from nn.distributed import DistributedCluster

# Multi-node inference
cluster = DistributedCluster(nodes=["gpu1", "gpu2", "gpu3"])
ai = nn.NeuralNode(provider="distributed", cluster=cluster)
```

#### 26. Model Quantization Tools
```python
from nn.quantize import quantize_model

# Automatic quantization
quantize_model(
    "llama-3-70b",
    output="llama-3-70b-q4.gguf",
    method="Q4_K_M"
)
```

#### 27. RLHF Training
```python
from nn.rlhf import RLHFTrainer

trainer = RLHFTrainer(base_model="llama3")
trainer.add_preference(good_response, bad_response)
trainer.train()
```

#### 28. Model Compression
```python
from nn.compression import ModelCompressor

compressor = ModelCompressor()
compressor.compress("model.pt", method="pruning", target_size="50%")
```

#### 29. Custom Provider Plugins
```python
from nn.providers import register_provider

@register_provider("my_api")
class MyProvider(BaseProvider):
    def chat(self, messages, **kwargs):
        # Custom implementation
        pass
```

---

## Quick Reference

### Import Shortcuts
```python
# Main package
import nn
ai = nn.NeuralNode()

# Or full name
from neuralnode import NeuralNode
```

### Common Patterns
```python
# Quick chat
import nn
ai = nn.NeuralNode(provider="auto")
response = ai.chat("Hello")

# With streaming
for chunk in ai.chat("Hello", stream=True):
    print(chunk.text, end="")

# Agent
tool = nn.Tool(name="calc", description="Calculate", parameters={"expr": str}, handler=lambda e: eval(e))
agent = ai.agent(tools=[tool])
agent.run("Calculate 2+2")

# RAG
from nn.rag import RAG
rag = RAG(llm=ai)
rag.add_documents(["doc1", "doc2"])
answer = rag.query("Question?")

# Chain
from nn import Chain, PromptTemplate
chain = Chain([
    PromptTemplate.from_template("Summarize: {text}"),
    ai.chat
])
chain.run("Long text...")
```

---

## Contributing

To add a new feature:
1. Implement in appropriate module under `src/neuralnode/`
2. Add exports to `__init__.py`
3. Document in this file
4. Add example to `examples/`
5. Add tests to `tests/`

---

*Last updated: 2024-01-15*
*Version: 0.1.0*
