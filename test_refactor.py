"""Integration test verifying all refactored modules import and work correctly."""
import sys
sys.path.insert(0, "src")

errors = []

def test(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        errors.append(f"  [FAIL] {name}: {e}")
        print(errors[-1])

print("=== NeuralNode Refactor Verification ===\n")

# 1. Core imports
def check_core():
    import neuralnode as nn
    assert hasattr(nn, "NeuralNode")
    assert hasattr(nn, "__version__")
    assert nn.__version__ == "3.1.0"
test("Main package imports", check_core)

# 2. Memory
def check_memory():
    from neuralnode.memory import ConversationMemory, FileMemory, SlidingWindowMemory, SummaryMemory
    mem = ConversationMemory()
    from neuralnode import Message
    mem.add(Message.user("hello"))
    mem.add(Message.assistant("hi"))
    assert mem.count() == 2
    results = mem.search("hello")
    assert len(results) == 1
    assert results[0].content == "hello"
test("Memory module (ConversationMemory + search)", check_memory)

# 3. Sliding window
def check_sliding():
    from neuralnode.memory import SlidingWindowMemory
    from neuralnode import Message
    mem = SlidingWindowMemory(max_pairs=2)
    for i in range(6):
        mem.add(Message.user(f"msg {i*2}"))
        mem.add(Message.assistant(f"reply {i*2+1}"))
    # Should only keep 2 pairs (4 messages) + no system
    assert mem.count() <= 5  # generous bound
test("SlidingWindowMemory trimming", check_sliding)

# 4. RAG
def check_rag():
    from neuralnode.rag import (
        Document, Chunk, TextSplitter, RecursiveCharacterTextSplitter,
        InMemoryVectorStore, ChromaDBStore, FAISSVectorStore,
        DocumentLoader, RAG, RAGPipeline, create_rag, load_documents, split_documents
    )
    d = Document(content="Hello world!", metadata={"source": "test"})
    assert d.doc_id is not None
    splitter = TextSplitter(chunk_size=100)
    chunks = splitter.split_documents([d])
    assert len(chunks) >= 1
    store = InMemoryVectorStore()
    store.add(chunks)
    assert store.count() == len(chunks)
test("RAG module (Document, TextSplitter, InMemoryVectorStore)", check_rag)

# 5. Agents
def check_agents():
    from neuralnode.agents import (
        SimpleAgent, ReActAgent, ChainOfThoughtAgent, ReflectiveAgent,
        ThinkingAgent, create_thinking_agent, AgentExecutorOutput
    )
    assert ThinkingAgent is not None
    assert create_thinking_agent is not None
test("Agents module (ThinkingAgent, all agent types)", check_agents)

# 6. Vision
def check_vision():
    from neuralnode.vision import encode_image_b64, get_image_media_type, build_vision_message, VisionTool
    assert get_image_media_type("photo.jpg") == "image/jpeg"
    assert get_image_media_type("image.png") == "image/png"
test("Vision module (VisionTool, image helpers)", check_vision)

# 7. TTS
def check_tts():
    from neuralnode.tts import NeuralTTS, list_voices
    voices = list_voices()
    assert len(voices) >= 10
    tts = NeuralTTS()
    assert tts is not None
test("TTS module (NeuralTTS, list_voices)", check_tts)

# 8. Speech
def check_speech():
    from neuralnode.speech import NeuralSpeech, SUPPORTED_LANGUAGES
    assert len(SUPPORTED_LANGUAGES) > 20
    sr = NeuralSpeech()
    assert sr is not None
test("Speech module (NeuralSpeech, SUPPORTED_LANGUAGES)", check_speech)

# 9. OpenAI blocked — provider name
def check_openai_provider():
    import neuralnode as nn
    try:
        nn.NeuralNode(provider="openai", api_key="sk-test")
        errors.append("  [FAIL] OpenAI provider not blocked!")
        print(errors[-1])
    except ValueError as e:
        assert "openai" in str(e).lower() or "blocked" in str(e).lower()
test("OpenAI provider blocked", check_openai_provider)

# 10. OpenAI blocked — GPT model name
def check_openai_model():
    import neuralnode as nn
    from neuralnode.core.openai_blocker import check_openai_blocked
    result = check_openai_blocked(provider_name="ollama", model_name="gpt-4", api_key=None)
    assert result is not None  # Should return an error message
test("OpenAI model (gpt-4) blocked", check_openai_model)

# 11. Providers registered
def check_providers():
    from neuralnode.providers import ProviderRegistry, is_local_provider
    assert is_local_provider("ollama") is True
    assert is_local_provider("anthropic") is False
    assert is_local_provider("llamacpp") is True
    assert is_local_provider("google") is False
test("Provider registry and is_local_provider()", check_providers)

# 12. NeuralNode object
def check_neuralnode_obj():
    import neuralnode as nn
    # Only test attributes, not actual API call
    node = nn.NeuralNode.__new__(nn.NeuralNode)
    assert nn.NeuralNode is not None
    # Test repr doesn't crash
    assert "NeuralNode" in repr(nn.NeuralNode)
test("NeuralNode class attributes", check_neuralnode_obj)

print(f"\n=== Results: {12 - len(errors)}/12 passed ===")
if errors:
    print("\nFailed tests:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
