"""
NeuralNode — Simplest AI Framework for Python
==============================================

Quick Start::

    import neuralnode as nn

    # Cloud API
    ai = nn.NeuralNode(provider="anthropic")
    print(ai.chat("Hello!"))

    # Local model
    ai = nn.NeuralNode(provider="ollama", model="llama3.2")
    print(ai.chat("Tell me a joke"))

    # Agent with tools
    agent = ai.agent(skills=["web_search", "datetime"])
    print(agent.run("What is the latest news about AI?"))

    # Agent with thinking (best for local models)
    agent = ai.agent(thinking=True, tools=[])
    print(agent.run("Explain quantum entanglement simply"))

    # Vision
    ai = nn.NeuralNode(provider="google", model="gemini-1.5-pro")
    print(ai.chat_with_image("What's in this image?", image="photo.jpg"))

    # RAG
    rag = ai.rag()
    rag.add_documents(["report.pdf", "data.txt"])
    print(rag.query("What are the main findings?"))

    # TTS
    from neuralnode.tts import NeuralTTS
    tts = NeuralTTS()
    tts.speak("Hello world!")

    # Speech Recognition
    from neuralnode.speech import NeuralSpeech
    sr = NeuralSpeech()
    text = sr.listen()
"""

__version__ = "2.1.0"
__author__ = "NeuralNode Contributors"

# ── Core types ────────────────────────────────────────────────────────────────
from .core import (
    Message,
    MessageRole,
    ToolCall,
    LLMResponse,
    StreamingChunk,
    ensure_llm_response,
    ensure_streaming_chunks,
    ToolResult,
    ParameterSpec,
    BaseTool,
    BaseProvider,
    BaseAgent,
    Memory,
)

# ── Provider layer ────────────────────────────────────────────────────────────
from .providers import (
    # Registry & factories
    ProviderRegistry,
    create_provider,
    auto_select_provider,
    get_available_providers,
    is_local_provider,
    # Most-used providers (re-exported for convenience)
    OllamaProvider,
    AnthropicProvider,
    GoogleProvider,
    GroqProvider,
    MistralProvider,
    CohereProvider,
    DeepSeekProvider,
    LlamaCppProvider,
    TransformersProvider,
    UniversalLocalProvider,
    # Custom models
    Horus,
    HorusModel,
    HorusProvider,
)

# ── Memory ────────────────────────────────────────────────────────────────────
from .memory import (
    ConversationMemory,
    FileMemory,
    SlidingWindowMemory,
    SummaryMemory,
)

# ── RAG ───────────────────────────────────────────────────────────────────────
from .rag import (
    RAG,
    RAGPipeline,
    create_rag,
    Document,
    Chunk,
    TextSplitter,
    RecursiveCharacterTextSplitter,
    InMemoryVectorStore,
    ChromaDBStore,
    FAISSVectorStore,
    DocumentLoader,
    load_documents,
    split_documents,
)

# ── Agents ────────────────────────────────────────────────────────────────────
from .agents import (
    SimpleAgent,
    ReActAgent,
    ChainOfThoughtAgent,
    ReflectiveAgent,
    ThinkingAgent,
    AgentBuilder,
    create_agent,
    create_react_agent,
    create_thinking_agent,
    AgentExecutorOutput,
)

from .reasoning import (
    UniversalReasoner,
    ReasoningResult,
    ReasoningStrategy,
    reason,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
from .tools import (
    WebSearchTool,
    FileReaderTool,
    FileWriterTool,
    CalculatorTool,
    PythonREPLTool,
    DateTimeTool,
    SystemOperationsTool,
)

# Vision support
try:
    from .vision import VisionTool, build_vision_message, capture_screenshot
    _VISION_AVAILABLE = True
except ImportError:
    _VISION_AVAILABLE = False

# TTS support
try:
    from .tts import NeuralTTS, list_voices as tts_list_voices
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False

# Speech recognition
try:
    from .speech import NeuralSpeech, SUPPORTED_LANGUAGES as SPEECH_LANGUAGES
    _SPEECH_AVAILABLE = True
except ImportError:
    _SPEECH_AVAILABLE = False

# Replica TTS module (standalone)
try:
    from . import replica
    from .replica import ReplicaTTS, list_voices as replica_list_voices, replica_voice_list, speak as replica_speak
    _REPLICA_AVAILABLE = True
except ImportError:
    _REPLICA_AVAILABLE = False

# TurboQuant compression
try:
    from .turboquant import TurboQuant, TurboQuantEngine, TurboQuantConfig
    _TURBOQUANT_AVAILABLE = True
except ImportError:
    _TURBOQUANT_AVAILABLE = False

# Thinking Mode (structured reasoning for any LLM)
try:
    from .thinking import ThinkingMode, ThinkingResponse, enable_thinking
    _THINKING_AVAILABLE = True
except ImportError:
    _THINKING_AVAILABLE = False

# OpenAI blocker
from .core.openai_blocker import check_openai_blocked

# Chains module
try:
    from .chains import (
        BaseChain, ChainOutput, LLMChain, SequentialChain,
        RouterChain, RetrievalQAChain, TransformChain, APIChain,
        create_llm_chain, create_sequential_chain, create_retrieval_qa_chain
    )
    _CHAINS_AVAILABLE = True
except ImportError:
    _CHAINS_AVAILABLE = False

# Integrations
try:
    from .integrations.telegram import (
        TelegramBot,
        TelegramAgent,
        TelegramBotConfig,
        TelegramBotBuilder,
        create_telegram_bot,
    )
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False

try:
    from .integrations.discord import DiscordBot
    _DISCORD_AVAILABLE = True
except ImportError:
    _DISCORD_AVAILABLE = False

try:
    from .integrations.slack import SlackBot
    _SLACK_AVAILABLE = True
except ImportError:
    _SLACK_AVAILABLE = False

# Logging helper
from .utils.logger import get_logger, setup_logging
from .utils.dependencies import ensure_feature_dependencies



# ── Skill → tools mapping ─────────────────────────────────────────────────────

def _resolve_skills(skills):
    """Expand skill names to concrete tool instances."""
    tools = []
    for skill in skills:
        skill = skill.lower()
        if skill in ("web_search", "search", "websearch"):
            tools.append(WebSearchTool())
        elif skill in ("datetime", "date", "time"):
            tools.append(DateTimeTool())
        elif skill in ("system", "system_control", "sysops"):
            tools.append(SystemOperationsTool())
        elif skill in ("file", "files", "file_system"):
            tools.extend([FileReaderTool(), FileWriterTool()])
        elif skill in ("python", "code", "repl"):
            tools.append(PythonREPLTool())
        elif skill in ("calculator", "math"):
            tools.append(CalculatorTool())
        elif skill in ("vision", "image", "vision_tool") and _VISION_AVAILABLE:
            pass  # VisionTool needs an LLM; added separately in agent()
        elif skill == "all":
            tools.extend([
                WebSearchTool(),
                DateTimeTool(),
                SystemOperationsTool(),
                FileReaderTool(),
                FileWriterTool(),
                CalculatorTool(),
            ])
    return tools


# ── Main NeuralNode class ─────────────────────────────────────────────────────

class NeuralNode:
    """The main entry-point for NeuralNode.

    Supports cloud APIs and local models with the same simple interface.

    Args:
        provider: Provider name, e.g. ``"ollama"``, ``"anthropic"``,
            ``"google"``, ``"groq"``, ``"mistral"``, etc.
        model: Model name, e.g. ``"llama3.2"``, ``"claude-3-haiku-20240307"``.
        api_key: API key (most providers also read env variables).
        system_prompt: Default system prompt for all conversations.
        temperature: Generation temperature (0.0–1.0).
        max_tokens: Maximum output tokens.
        model_settings: Extra settings for local models, e.g.
            ``{"context_length": 8192, "gpu_layers": 35, "top_k": 40}``.
        **kwargs: Additional kwargs forwarded to the provider constructor.

    Example::

        # OpenAI is blocked — use any of these instead:
        ai = NeuralNode(provider="anthropic")             # Claude
        ai = NeuralNode(provider="google")                # Gemini
        ai = NeuralNode(provider="groq")                  # Groq
        ai = NeuralNode(provider="ollama", model="llama3.2")  # Local
    """

    def __init__(
        self,
        provider: str = "ollama",
        model: str = None,
        api_key: str = None,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        model_settings: dict = None,
        **kwargs,
    ):
        # Block OpenAI before doing anything else
        check_openai_blocked(
            provider=provider,
            model=model,
            api_key=api_key,
        )

        self.provider_name = provider.lower()
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_settings = model_settings or {}
        self._is_local = is_local_provider(self.provider_name)

        # Build kwargs for provider
        provider_kwargs = dict(kwargs)
        if api_key:
            provider_kwargs["api_key"] = api_key
        if temperature is not None:
            provider_kwargs["temperature"] = temperature
        # Forward model_settings for local providers
        if self.model_settings and self._is_local:
            if "context_length" in self.model_settings:
                provider_kwargs["context_length"] = self.model_settings["context_length"]
            if "gpu_layers" in self.model_settings:
                provider_kwargs["n_gpu_layers"] = self.model_settings["gpu_layers"]
            if "top_k" in self.model_settings:
                provider_kwargs["top_k"] = self.model_settings["top_k"]
            if "top_p" in self.model_settings:
                provider_kwargs["top_p"] = self.model_settings["top_p"]
            if "repeat_penalty" in self.model_settings:
                provider_kwargs["repeat_penalty"] = self.model_settings["repeat_penalty"]

        if self.provider_name == "auto":
            self._llm = auto_select_provider(prefer_local=False)
            self.provider_name = getattr(self._llm, "provider_name", "auto")
            self.model = getattr(self._llm, "model", self.model)
            self._is_local = is_local_provider(self.provider_name)
        else:
            create_kwargs = dict(provider_kwargs)
            if self.provider_name == "horus":
                if self.model is not None:
                    create_kwargs["model_id"] = self.model
            else:
                create_kwargs["model"] = self.model

            self._llm = create_provider(
                self.provider_name,
                **create_kwargs,
            )

        if hasattr(self._llm, "load") and callable(getattr(self._llm, "load")):
            try:
                self._llm.load()
            except TypeError:
                pass
        self._memory = None

    # ── Direct chat ───────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        system_prompt: str = None,
        history: list = None,
        stream: bool = False,
    ):
        """Send a single message and get a response.

        Args:
            message: The user message.
            system_prompt: Override system prompt for this call.
            history: Previous messages as ``[{"role": ..., "content": ...}]``.
            stream: Return a streaming generator if True.

        Returns:
            Response string, or streaming iterator.

        Example::

            ai = NeuralNode(provider="groq")
            print(ai.chat("What is the capital of France?"))
        """
        messages = []
        sys = system_prompt or self.system_prompt
        if sys:
            messages.append(Message.system(sys))

        if history:
            for h in history:
                role = h.get("role", "user")
                content = h.get("content", "")
                if role == "user":
                    messages.append(Message.user(content))
                elif role == "assistant":
                    messages.append(Message.assistant(content))

        messages.append(Message.user(message))
        response = self._llm.chat(messages, stream=stream)

        if stream:
            return (chunk.content for chunk in ensure_streaming_chunks(response))
        return ensure_llm_response(response, default_model=self.model or "unknown", default_provider=self.provider_name).content

    def chat_with_image(
        self,
        message: str,
        image: str = None,
        image_url: str = None,
        system_prompt: str = None,
    ) -> str:
        """Send a message with an image to a vision-capable model.

        Args:
            message: The text question or instruction.
            image: Local file path to an image.
            image_url: Public URL to an image.
            system_prompt: Override the system prompt.

        Returns:
            Model response string.

        Example::

            ai = NeuralNode(provider="google", model="gemini-1.5-pro")
            print(ai.chat_with_image("What is in this photo?", image="cat.jpg"))
        """
        if not _VISION_AVAILABLE:
            raise ImportError("Vision module not available. Check neuralnode/vision/__init__.py")

        msgs = []
        sys = system_prompt or self.system_prompt
        if sys:
            msgs.append(Message.system(sys))
        msgs.append(build_vision_message(message, image=image, image_url=image_url))
        response = self._llm.chat(msgs)
        return ensure_llm_response(response, default_model=self.model or "unknown", default_provider=self.provider_name).content

    # ── Agent factory ─────────────────────────────────────────────────────────

    def agent(
        self,
        tools: list = None,
        skills: list = None,
        system_prompt: str = None,
        thinking: bool = None,
        max_iterations: int = 15,
        agent_type: str = "react",
    ):
        """Create an AI agent backed by this model.

        Args:
            tools: List of ``BaseTool`` instances.
            skills: Shorthand skill names: ``"web_search"``, ``"datetime"``,
                ``"system_control"``, ``"file"``, ``"python"``, ``"calculator"``,
                ``"all"``.
            system_prompt: Custom system prompt for the agent.
            thinking: Enable chain-of-thought reasoning. Defaults to ``True``
                for local models, ``False`` for cloud APIs.
            max_iterations: Maximum agent reasoning iterations.
            agent_type: ``"react"`` (default), ``"simple"``, ``"cot"``.

        Returns:
            Agent instance (with ``.run(task)`` and ``.arun(task)`` methods).

        Example::

            ai = NeuralNode(provider="ollama", model="llama3.2")
            agent = ai.agent(skills=["web_search", "datetime"])
            print(agent.run("What time is it and what's the latest AI news?"))
        """
        all_tools = list(tools or [])
        if skills:
            all_tools.extend(_resolve_skills(skills))

        sys = system_prompt or self.system_prompt

        # Thinking defaults to True for local models
        use_thinking = thinking if thinking is not None else self._is_local

        if agent_type == "simple":
            base = SimpleAgent(self._llm, all_tools, sys)
        elif agent_type == "cot":
            base = ChainOfThoughtAgent(self._llm, all_tools, sys)
        else:
            base = ReActAgent(self._llm, all_tools, sys)

        base.max_iterations = max_iterations

        if use_thinking:
            return ThinkingAgent(base)
        return base

    # ── RAG factory ───────────────────────────────────────────────────────────

    def rag(
        self,
        store: str = "memory",
        persist_dir: str = None,
        embedding_provider=None,
        top_k: int = 5,
    ) -> RAG:
        """Create a RAG system using this model for generation and embedding.

        Args:
            store: ``"memory"`` (default), ``"chroma"``, or ``"faiss"``.
            persist_dir: Persist directory for ChromaDB.
            top_k: Number of documents to retrieve.

        Returns:
            Configured RAG instance.

        Example::

            ai = NeuralNode(provider="ollama", model="nomic-embed-text")
            rag = ai.rag(store="chroma", persist_dir="./my_db")
            rag.add_documents(["notes.pdf", "data.txt"])
            print(rag.query("What is the main topic?"))
        """
        return create_rag(
            self._llm,
            embedding_provider=embedding_provider,
            store=store,
            persist_dir=persist_dir,
            top_k=top_k,
        )

    # ── Memory ────────────────────────────────────────────────────────────────

    def with_memory(self, memory: Memory = None):
        """Attach memory to this NeuralNode instance.

        Args:
            memory: Memory instance (defaults to ConversationMemory).

        Returns:
            self, for chaining.

        Example::

            ai = NeuralNode(provider="ollama").with_memory()
            ai.chat("My name is Alice")
            ai.chat("What is my name?")  # Remembers "Alice"
        """
        self._memory = memory or ConversationMemory()
        return self

    def chat_with_memory(self, message: str, system_prompt: str = None) -> str:
        """Chat using the attached memory for conversation history.

        Call ``with_memory()`` first, or pass a memory instance.

        Args:
            message: User message.
            system_prompt: Override system prompt.

        Returns:
            Response string.
        """
        if self._memory is None:
            self.with_memory()

        history = [
            {"role": m.role.value, "content": m.content}
            for m in self._memory.get()
            if m.role.value in ("user", "assistant")
        ]

        response = self.chat(message, system_prompt=system_prompt, history=history)

        self._memory.add(Message.user(message))
        self._memory.add(Message.assistant(response))

        return response

    # ── Embeddings ────────────────────────────────────────────────────────────

    def embed(self, texts: list) -> list:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of strings.

        Returns:
            List of embedding vectors.
        """
        return self._llm.embed(texts)

    def get_capabilities(self):
        """Backward-compatible capabilities accessor."""
        return self.capabilities

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def provider(self):
        return self.provider_name

    @property
    def is_local(self):
        """True if the provider is a local model (e.g. Ollama, LlamaCpp)."""
        return self._is_local

    @property
    def capabilities(self):
        """Provider capabilities dict."""
        return self._llm.capabilities

    def __repr__(self):
        return f"NeuralNode(provider={self.provider_name!r}, model={self.model!r}, local={self._is_local})"


# ── Convenience aliases ───────────────────────────────────────────────────────

# Tool short-names for ergonomic imports
WebSearch = WebSearchTool
FileReader = FileReaderTool
FileWriter = FileWriterTool
Calculator = CalculatorTool
DateTime = DateTimeTool
SystemOps = SystemOperationsTool
if _VISION_AVAILABLE:
    Vision = VisionTool


# ── __all__ ───────────────────────────────────────────────────────────────────

__all__ = [
    "__version__",

    # Main class
    "NeuralNode",

    # Core types
    "Message", "MessageRole", "ToolCall", "LLMResponse", "StreamingChunk",
    "ToolResult", "ParameterSpec", "BaseTool", "BaseProvider", "BaseAgent", "Memory",

    # Providers
    "ProviderRegistry", "create_provider", "auto_select_provider",
    "get_available_providers", "is_local_provider",
    "OllamaProvider", "AnthropicProvider", "GoogleProvider",
    "GroqProvider", "MistralProvider", "CohereProvider", "DeepSeekProvider",
    "LlamaCppProvider", "TransformersProvider", "UniversalLocalProvider",
    "Horus", "HorusModel", "HorusProvider",

    # Memory
    "ConversationMemory", "FileMemory", "SlidingWindowMemory", "SummaryMemory",

    # RAG
    "RAG", "RAGPipeline", "create_rag",
    "Document", "Chunk", "TextSplitter", "RecursiveCharacterTextSplitter",
    "InMemoryVectorStore", "ChromaDBStore", "FAISSVectorStore",
    "DocumentLoader", "load_documents", "split_documents",

    # Agents
    "SimpleAgent", "ReActAgent", "ChainOfThoughtAgent", "ReflectiveAgent",
    "ThinkingAgent", "AgentBuilder",
    "create_agent", "create_react_agent", "create_thinking_agent",
    "AgentExecutorOutput",

    # Reasoning & Thinking
    "UniversalReasoner", "ReasoningResult", "ReasoningStrategy", "reason",
    "ThinkingMode", "ThinkingResponse", "enable_thinking",

    # Tools (full names)
    "WebSearchTool", "FileReaderTool", "FileWriterTool",
    "CalculatorTool", "PythonREPLTool", "DateTimeTool", "SystemOperationsTool",

    # Tool short aliases
    "WebSearch", "FileReader", "FileWriter", "Calculator", "DateTime", "SystemOps",

    # Chains
    "BaseChain", "ChainOutput", "LLMChain", "SequentialChain",
    "RouterChain", "RetrievalQAChain", "TransformChain", "APIChain",
    "create_llm_chain", "create_sequential_chain", "create_retrieval_qa_chain",

    # Utilities
    "get_logger", "setup_logging",
    "ensure_feature_dependencies",
    "check_openai_blocked",
]

# Add optional exports
if _CHAINS_AVAILABLE:
    __all__ += [
        "BaseChain", "ChainOutput", "LLMChain", "SequentialChain",
        "RouterChain", "RetrievalQAChain", "TransformChain", "APIChain",
        "create_llm_chain", "create_sequential_chain", "create_retrieval_qa_chain"
    ]
if _VISION_AVAILABLE:
    __all__ += ["VisionTool", "build_vision_message", "capture_screenshot", "Vision"]
if _TTS_AVAILABLE:
    __all__ += ["NeuralTTS", "tts_list_voices"]
if _SPEECH_AVAILABLE:
    __all__ += ["NeuralSpeech", "SPEECH_LANGUAGES"]
if _REPLICA_AVAILABLE:
    __all__ += ["ReplicaTTS", "replica_list_voices", "replica_voice_list", "replica_speak", "replica"]
if _TELEGRAM_AVAILABLE:
    __all__ += [
        "TelegramBot",
        "TelegramAgent",
        "TelegramBotConfig",
        "TelegramBotBuilder",
        "create_telegram_bot",
    ]
if _DISCORD_AVAILABLE:
    __all__ += ["DiscordBot"]
if _SLACK_AVAILABLE:
    __all__ += ["SlackBot"]
if _TURBOQUANT_AVAILABLE:
    __all__ += ["TurboQuant", "TurboQuantEngine", "TurboQuantConfig"]
if _THINKING_AVAILABLE:
    __all__ += ["ThinkingMode", "ThinkingResponse", "enable_thinking"]
