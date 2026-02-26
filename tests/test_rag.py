"""Tests for RAG functionality."""

import pytest

from neuralnode.rag import Document, TextSplitter, SimpleVectorStore, RAG


class TestDocument:
    def test_document_creation(self):
        doc = Document(content="Hello world", metadata={"key": "value"})
        assert doc.content == "Hello world"
        assert doc.metadata == {"key": "value"}
        assert doc.id is None

    def test_document_with_id(self):
        doc = Document(content="Test", id="doc1")
        assert doc.id == "doc1"


class TestTextSplitter:
    def test_split_simple_text(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        text = "This is a test. " * 20
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)

    def test_split_with_overlap(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "Word " * 50
        chunks = splitter.split(text)
        assert len(chunks) > 1

    def test_split_documents(self):
        splitter = TextSplitter(chunk_size=100, chunk_overlap=0)
        docs = [
            Document(content="Hello world " * 20, id="doc1"),
        ]
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 1
        assert all(chunk.id.startswith("doc1_chunk_") for chunk in chunks)


class TestSimpleVectorStore:
    def test_add_document(self):
        store = SimpleVectorStore()
        doc = Document(content="test")
        store.add(doc)
        assert len(store.documents) == 1

    def test_add_multiple_documents(self):
        store = SimpleVectorStore()
        docs = [Document(content=f"doc {i}") for i in range(5)]
        store.add(docs)
        assert len(store.documents) == 5

    def test_similarity_search(self):
        # Mock embedding function
        def embed(text):
            # Simple hash-based embedding
            return [ord(c) / 255.0 for c in text[:10].ljust(10)]
        
        store = SimpleVectorStore(embedding_function=embed)
        store.add([
            Document(content="apple orange fruit"),
            Document(content="car truck vehicle"),
            Document(content="python programming code"),
        ])
        
        results = store.similarity_search("fruit and vegetables", k=2)
        assert len(results) == 2

    def test_empty_search(self):
        store = SimpleVectorStore()
        def embed(text):
            return [0.0] * 10
        
        results = store.similarity_search("query", embedding_function=embed)
        assert results == []

    def test_clear(self):
        store = SimpleVectorStore()
        store.add(Document(content="test"))
        store.clear()
        assert len(store.documents) == 0
        assert len(store.vectors) == 0


class TestRAG:
    def test_rag_creation(self):
        mock_llm = object()
        rag = RAG(llm=mock_llm)
        assert rag.llm is mock_llm

    def test_add_string_documents(self):
        mock_llm = object()
        
        def mock_embed(text):
            return [0.5] * 10
        
        rag = RAG(llm=mock_llm, embedding_function=mock_embed)
        rag.add_documents(["Document 1", "Document 2"])
        assert len(rag.vector_store.documents) == 2

    def test_add_document_objects(self):
        mock_llm = object()
        
        def mock_embed(text):
            return [0.5] * 10
        
        rag = RAG(llm=mock_llm, embedding_function=mock_embed)
        rag.add_documents(Document(content="Test"))
        assert len(rag.vector_store.documents) == 1


class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = SimpleVectorStore._cosine_similarity(a, b)
        assert sim == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = SimpleVectorStore._cosine_similarity(a, b)
        assert sim == 0.0

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        sim = SimpleVectorStore._cosine_similarity(a, b)
        assert sim == -1.0
