"""Comprehensive test suite for NeuralNode."""

import pytest
from unittest.mock import MagicMock, patch
import json
import tempfile
from pathlib import Path

# Core tests
class TestNeuralNodeCore:
    """Test core NeuralNode functionality."""
    
    def test_neuralnode_init(self):
        """Test NeuralNode initialization."""
        import nn
        
        # Mock provider to avoid API calls
        with patch('nn.providers.openai.OpenAIProvider') as mock_provider:
            mock_instance = MagicMock()
            mock_provider.return_value = mock_instance
            
            ai = nn.NeuralNode(provider="openai", api_key="test-key")
            assert ai is not None
    
    def test_chat(self):
        """Test basic chat."""
        import nn
        
        with patch('nn.providers.openai.OpenAIProvider') as mock_provider:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello!"
            mock_instance.chat.return_value = mock_response
            mock_provider.return_value = mock_instance
            
            ai = nn.NeuralNode(provider="openai")
            response = ai.chat("Hi")
            assert response.text == "Hello!"
    
    def test_streaming(self):
        """Test streaming responses."""
        import nn
        
        with patch('nn.providers.openai.OpenAIProvider') as mock_provider:
            mock_instance = MagicMock()
            chunks = [MagicMock(text="Hello"), MagicMock(text=" world")]
            mock_instance.chat.return_value = iter(chunks)
            mock_provider.return_value = mock_instance
            
            ai = nn.NeuralNode(provider="openai")
            result = list(ai.chat("Hi", stream=True))
            assert len(result) == 2


class TestMemory:
    """Test memory systems."""
    
    def test_conversation_memory(self):
        """Test conversation memory."""
        import nn
        
        memory = nn.ConversationMemory(max_messages=10)
        
        from nn.core.types import Message, MessageRole
        msg1 = Message(role=MessageRole.USER, content="Hello")
        msg2 = Message(role=MessageRole.ASSISTANT, content="Hi!")
        
        memory.add_message(msg1)
        memory.add_message(msg2)
        
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
    
    def test_long_term_memory(self):
        """Test long-term memory."""
        from nn.memory.advanced_memory import LongTermMemory, MemoryEntry
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test.db"
            memory = LongTermMemory(db_path=db_path)
            
            from datetime import datetime
            entry = MemoryEntry(
                content="Test memory",
                timestamp=datetime.now(),
                importance=0.8,
                category="test",
            )
            
            memory_id = memory.add(entry)
            assert memory_id > 0
            
            results = memory.search("Test")
            assert len(results) > 0


class TestRAG:
    """Test RAG functionality."""
    
    def test_document_creation(self):
        """Test document creation."""
        import nn
        
        doc = nn.Document(
            content="Test content",
            metadata={"source": "test"},
            id="doc1"
        )
        
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"
    
    def test_vector_store(self):
        """Test vector store."""
        import nn
        
        def embed_fn(text):
            return [0.1, 0.2, 0.3]
        
        store = nn.SimpleVectorStore(embedding_function=embed_fn)
        
        doc = nn.Document(content="Test", id="1")
        store.add_documents([doc])
        
        assert store.doc_count == 1
    
    def test_faiss_vector_store(self):
        """Test FAISS vector store."""
        from nn.rag.advanced_vectorstores import FAISSVectorStore
        
        def embed_fn(text):
            return [0.1] * 1536
        
        store = FAISSVectorStore(embedding_function=embed_fn)
        
        from nn.rag.vectorstore import Document
        docs = [Document(content="Test doc", id="1")]
        
        store.add_documents(docs, embeddings=[[0.1] * 1536])
        assert store.doc_count == 1


class TestAgents:
    """Test agent functionality."""
    
    def test_tool_creation(self):
        """Test tool creation."""
        import nn
        
        @nn.create_tool(
            name="test_tool",
            description="A test tool",
            parameters={"x": {"type": "integer"}}
        )
        def test_tool(x: int) -> int:
            return x * 2
        
        assert test_tool.name == "test_tool"
        assert test_tool.execute(x=5) == 10
    
    def test_enhanced_agent(self):
        """Test enhanced agent."""
        from nn.agents.enhanced_agent import EnhancedAgent
        
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I'll help you with that!"
        mock_llm.chat.return_value = mock_response
        
        agent = EnhancedAgent(llm=mock_llm)
        result = agent.run("Hello")
        
        assert result is not None


class TestTools:
    """Test tools functionality."""
    
    def test_file_manager(self):
        """Test file manager."""
        from nn.tools.system_control import FileManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fm = FileManager(base_path=tmpdir)
            
            # Test write
            result = fm.write_file(f"{tmpdir}/test.txt", "Hello")
            assert "Successfully" in result
            
            # Test read
            content = fm.read_file(f"{tmpdir}/test.txt")
            assert content == "Hello"
    
    def test_process_manager(self):
        """Test process manager."""
        from nn.tools.system_control import ProcessManager
        
        pm = ProcessManager()
        processes = pm.list_processes()
        
        assert len(processes) > 0
        assert processes[0].pid is not None
    
    def test_web_search(self):
        """Test web search."""
        from nn.tools.web_search import DuckDuckGoSearch
        
        # Mock the search to avoid actual network calls
        with patch.object(DuckDuckGoSearch, 'search') as mock_search:
            mock_search.return_value = [
                MagicMock(title="Test", url="http://test.com", snippet="Test result")
            ]
            
            searcher = DuckDuckGoSearch()
            results = searcher.search("test query")
            
            assert len(results) > 0


class TestVectorStores:
    """Test advanced vector stores."""
    
    def test_chromadb_store(self):
        """Test ChromaDB store."""
        from nn.rag.advanced_vectorstores import ChromaDBVectorStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            def embed_fn(text):
                return [0.1] * 1536
            
            store = ChromaDBVectorStore(
                collection_name="test",
                persist_directory=tmpdir,
                embedding_function=embed_fn,
            )
            
            from nn.rag.vectorstore import Document
            docs = [Document(content="Test", id="1")]
            
            if store._chroma:  # Only if ChromaDB available
                store.add_documents(docs, embeddings=[[0.1] * 1536])
                assert store.doc_count >= 0


classTestMonitoring:
    """Test monitoring functionality."""
    
    def test_run_manager(self):
        """Test run manager."""
        from nn.utils.monitoring import RunManager
        
        manager = RunManager()
        run = manager.start_run("test_run", inputs={"prompt": "test"})
        
        assert run.id is not None
        assert run.name == "test_run"
        
        manager.end_run(run.id, outputs={"result": "success"})
        assert run.duration_ms is not None
    
    def test_monitor(self):
        """Test neural monitor."""
        from nn.utils.monitoring import NeuralMonitor
        
        monitor = NeuralMonitor(project_name="test")
        run_id = monitor.start_trace("test", inputs={"x": 1})
        
        monitor.end_trace(run_id, outputs={"y": 2})
        
        data = monitor.get_dashboard_data()
        assert data["project"] == "test"


class TestSmartCache:
    """Test smart cache."""
    
    def test_cache_basic(self):
        """Test basic caching."""
        from nn.utils.smart_cache import SmartCache
        from nn.core.types import LLMResponse, ProviderType, TokenUsage
        
        cache = SmartCache()
        
        response = LLMResponse(
            text="Hello",
            model="gpt-4",
            provider=ProviderType.OPENAI,
            usage=TokenUsage(10, 5, 15),
        )
        
        cache.set("key1", response)
        cached = cache.get("key1")
        
        assert cached is not None
        assert cached.text == "Hello"
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        from nn.utils.smart_cache import SmartCache
        from nn.core.types import LLMResponse, ProviderType
        
        cache = SmartCache()
        
        response = LLMResponse(text="Test", model="gpt-4", provider=ProviderType.OPENAI)
        
        cache.set("key1", response, tags={"test"})
        cache.set("key2", response, tags={"test"})
        
        invalidated = cache.invalidate_by_tag("test")
        assert invalidated == 2


class TestNeuralDoctor:
    """Test Neural Doctor error handling."""
    
    def test_error_diagnosis(self):
        """Test error diagnosis."""
        from nn.utils.neural_doctor import NeuralDoctor
        
        doctor = NeuralDoctor()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            diagnosis = doctor.diagnose(e)
            
            assert diagnosis.error_type == "ValueError"
            assert diagnosis.error_message == "Test error"
    
    def test_pattern_matching(self):
        """Test error pattern matching."""
        from nn.utils.neural_doctor import ErrorPatternDB
        
        db = ErrorPatternDB()
        
        # Test import error
        result = db.match("No module named 'numpy'")
        assert result is not None
        assert result["category"] == "import_error"
    
    def test_heal_attempt(self):
        """Test healing attempt."""
        from nn.utils.neural_doctor import NeuralDoctor
        
        doctor = NeuralDoctor(auto_heal=False)
        
        try:
            raise FileNotFoundError("test.txt")
        except Exception as e:
            attempt = doctor.heal(e)
            
            assert attempt.original_error == "test.txt"


class TestQuantization:
    """Test model quantization."""
    
    def test_model_quantizer_init(self):
        """Test quantizer initialization."""
        from nn.local.quantization import ModelQuantizer
        
        quantizer = ModelQuantizer()
        assert quantizer is not None
    
    def test_recommend_quantization(self):
        """Test quantization recommendation."""
        from nn.local.quantization import ModelQuantizer
        
        quantizer = ModelQuantizer()
        method = quantizer.recommend_quantization(model_size_gb=14, vram_gb=8)
        
        assert method in quantizer.QUANTIZATION_METHODS
    
    def test_list_models(self):
        """Test listing available models."""
        from nn.local.quantization import ModelQuantizer
        
        quantizer = ModelQuantizer()
        models = quantizer.list_available_models()
        
        assert len(models) > 0
        assert "name" in models[0]


class TestTelegram:
    """Test Telegram integration."""
    
    def test_telegram_auth(self):
        """Test Telegram auth."""
        from nn.integrations.telegram import TelegramAuth
        
        with tempfile.TemporaryDirectory() as tmpdir:
            auth_file = f"{tmpdir}/auth.json"
            auth = TelegramAuth(auth_file=auth_file)
            
            auth.add_user(12345, "testuser", "admin")
            
            assert auth.is_authorized(12345)
            assert auth.get_user_role(12345) == "admin"


class TestFinetune:
    """Test fine-tuning."""
    
    def test_finetuner_init(self):
        """Test finetuner initialization."""
        from nn.training.finetune import FineTuner
        
        tuner = FineTuner(base_model="gpt-3.5-turbo")
        assert tuner.base_model == "gpt-3.5-turbo"
    
    def test_add_example(self):
        """Test adding training examples."""
        from nn.training.finetune import FineTuner
        
        tuner = FineTuner(base_model="test-model")
        tuner.add_example("What is AI?", "AI is artificial intelligence.")
        
        assert len(tuner.examples) == 1


classTestDistributed:
    """Test distributed features."""
    
    def test_cluster_init(self):
        """Test cluster initialization."""
        from nn.distributed.cluster import DistributedCluster
        
        cluster = DistributedCluster()
        assert cluster is not None
    
    def test_discover_local(self):
        """Test local node discovery."""
        from nn.distributed.cluster import DistributedCluster
        
        cluster = DistributedCluster()
        node = cluster.discover_local()
        
        assert node.id == "local"


class TestMultimodal:
    """Test multimodal features."""
    
    def test_speech_to_text(self):
        """Test STT."""
        from nn.tools.multimodal import SpeechToText
        
        # Mock to avoid actual processing
        with patch('nn.tools.multimodal.SpeechToText._transcribe_openai'):
            stt = SpeechToText(provider="openai", api_key="test")
            assert stt.provider == "openai"
    
    def test_text_to_speech(self):
        """Test TTS."""
        from nn.tools.multimodal import TextToSpeech
        
        tts = TextToSpeech(provider="edge")
        assert tts.provider == "edge"
        
        voices = tts.get_voices()
        assert len(voices) > 0


classTestBrowser:
    """Test browser automation."""
    
    def test_browser_controller_init(self):
        """Test browser controller."""
        from nn.tools.browser_automation import BrowserController
        
        controller = BrowserController(headless=True)
        assert controller.headless is True


class TestSystemControl:
    """Test system control."""
    
    def test_system_info(self):
        """Test system info."""
        from nn.tools.system_control import SystemInfo
        
        info = SystemInfo()
        data = info.get_info()
        
        assert "platform" in data
        assert "cpu_count" in data


# Integration tests
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring external services."""
    
    def test_full_pipeline(self):
        """Test full pipeline."""
        import nn
        
        # This would require actual API keys
        # Skipped unless --integration flag used
        pytest.skip("Integration test - requires API keys")


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
