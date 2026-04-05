"""Test RAG system."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import neuralnode as nn
from neuralnode.core import Message, LLMResponse


class TestDocument:
    """Test Document dataclass."""
    
    def test_document_creation(self):
        """Test basic document creation."""
        doc = nn.Document(
            content="Test content",
            source="test.txt",
            metadata={"author": "test"}
        )
        assert doc.content == "Test content"
        assert doc.source == "test.txt"
        assert doc.metadata["author"] == "test"
    
    def test_document_to_dict(self):
        """Test document serialization."""
        doc = nn.Document(content="Test", source="file.txt")
        data = doc.to_dict()
        assert data["content"] == "Test"
        assert data["source"] == "file.txt"


class TestTextSplitter:
    """Test text splitting functionality."""
    
    def test_recursive_character_splitter(self):
        """Test recursive character text splitter."""
        splitter = nn.RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20
        )
        
        text = "This is a long text. " * 50
        chunks = splitter.split(text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # Each chunk should be within size limit (approximately)
        for chunk in chunks:
            assert len(chunk) <= 120  # chunk_size + some buffer
    
    def test_split_documents(self):
        """Test splitting documents."""
        splitter = nn.RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        
        docs = [
            nn.Document(content="Text one. " * 20, source="doc1.txt"),
            nn.Document(content="Text two. " * 20, source="doc2.txt"),
        ]
        
        chunks = splitter.split_documents(docs)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, nn.Chunk)


class TestInMemoryVectorStore:
    """Test in-memory vector store."""
    
    def test_init(self):
        """Test initialization."""
        store = nn.InMemoryVectorStore()
        assert store is not None
    
    def test_add_and_search(self):
        """Test adding documents and searching."""
        store = nn.InMemoryVectorStore()
        
        # Mock embeddings
        chunks = [
            nn.Chunk(content="Python programming", source="doc1.txt", embedding=[1.0, 0.0, 0.0]),
            nn.Chunk(content="JavaScript coding", source="doc2.txt", embedding=[0.0, 1.0, 0.0]),
            nn.Chunk(content="Machine learning", source="doc3.txt", embedding=[0.5, 0.5, 0.0]),
        ]
        
        store.add(chunks)
        
        # Search for Python
        results = store.search([1.0, 0.0, 0.0], k=2)
        assert len(results) <= 2
        
    def test_clear(self):
        """Test clearing the store."""
        store = nn.InMemoryVectorStore()
        chunks = [nn.Chunk(content="Test", source="test.txt", embedding=[1.0, 0.0])]
        store.add(chunks)
        
        store.clear()
        results = store.search([1.0, 0.0], k=1)
        assert len(results) == 0


class TestRAG:
    """Test RAG system."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = MagicMock()
        llm.embed.return_value = [[1.0, 0.0, 0.0]]
        llm.chat.return_value = LLMResponse(
            content="Based on the context, the answer is...",
            model="mock",
            provider="mock"
        )
        return llm
    
    def test_init(self, mock_llm):
        """Test RAG initialization."""
        rag = nn.RAG(llm=mock_llm, vector_store=nn.InMemoryVectorStore())
        assert rag.llm == mock_llm
        assert rag.vector_store is not None
    
    def test_add_documents(self, mock_llm, tmp_path):
        """Test adding documents."""
        rag = nn.RAG(llm=mock_llm, vector_store=nn.InMemoryVectorStore())
        
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content for RAG system.")
        
        rag.add_documents([str(test_file)])
        
        # Verify that embed was called
        assert mock_llm.embed.called
    
    def test_query(self, mock_llm):
        """Test querying RAG."""
        rag = nn.RAG(llm=mock_llm, vector_store=nn.InMemoryVectorStore())
        
        # Add some test chunks directly
        chunks = [
            nn.Chunk(content="Python is a programming language", source="doc.txt", embedding=[1.0, 0.0, 0.0]),
        ]
        rag.vector_store.add(chunks)
        
        response = rag.query("What is Python?")
        
        assert isinstance(response, str)
        assert mock_llm.chat.called
    
    def test_query_with_top_k(self, mock_llm):
        """Test querying with custom top_k."""
        rag = nn.RAG(llm=mock_llm, vector_store=nn.InMemoryVectorStore(), top_k=3)
        assert rag.top_k == 3


class TestDocumentLoader:
    """Test document loaders."""
    
    def test_text_loader(self, tmp_path):
        """Test text file loader."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        loader = nn.TextLoader()
        docs = loader.load(str(test_file))
        
        assert len(docs) == 1
        assert docs[0].content == "Hello World"
        assert docs[0].source == str(test_file)
    
    def test_text_loader_not_found(self):
        """Test loading non-existent file."""
        loader = nn.TextLoader()
        docs = loader.load("/nonexistent/file.txt")
        assert len(docs) == 0


class TestLoadDocuments:
    """Test load_documents function."""
    
    def test_load_single_file(self, tmp_path):
        """Test loading single file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Content")
        
        docs = nn.load_documents(str(test_file))
        assert len(docs) == 1
    
    def test_load_directory(self, tmp_path):
        """Test loading directory."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        
        docs = nn.load_documents(str(tmp_path))
        assert len(docs) == 2
