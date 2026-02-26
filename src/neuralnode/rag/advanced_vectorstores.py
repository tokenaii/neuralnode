"""Advanced Vector Stores with FAISS and ChromaDB support."""

from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import json
import pickle

from ..rag.vectorstore import Document


class BaseVectorStore(ABC):
    """Abstract base for vector stores."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """Load the vector store from disk."""
        pass
    
    @property
    @abstractmethod
    def doc_count(self) -> int:
        """Number of documents in the store."""
        pass


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store for fast similarity search."""
    
    def __init__(self, embedding_function: Optional[Callable[[str], List[float]]] = None, dimension: int = 1536):
        self.embedding_function = embedding_function
        self.dimension = dimension
        self._documents: Dict[str, Document] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._index = None
        self._doc_ids: List[str] = []
        
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        except ImportError:
            self._faiss = None
            print("Warning: FAISS not installed. Using fallback similarity search.")
            print("Install with: pip install faiss-cpu  or  pip install faiss-gpu")
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """Normalize vector for cosine similarity."""
        import math
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector
        return [x / norm for x in vector]
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add documents to FAISS index."""
        if not documents:
            return
        
        # Get embeddings
        if embeddings is None and self.embedding_function:
            embeddings = [self.embedding_function(doc.content) for doc in documents]
        elif embeddings is None:
            raise ValueError("Either provide embeddings or embedding_function")
        
        # Normalize embeddings for cosine similarity
        embeddings = [self._normalize(emb) for emb in embeddings]
        
        # Store documents
        for i, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{len(self._documents)}"
            doc.id = doc_id
            self._documents[doc_id] = doc
            self._embeddings[doc_id] = embeddings[i]
            self._doc_ids.append(doc_id)
        
        # Add to FAISS index if available
        if self._faiss and self._index:
            import numpy as np
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self._index.add(embeddings_array)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search using FAISS index."""
        if not self._documents:
            return []
        
        # Normalize query
        query_embedding = self._normalize(query_embedding)
        
        if self._faiss and self._index and self._index.ntotal > 0:
            import numpy as np
            query_array = np.array([query_embedding], dtype=np.float32)
            distances, indices = self._index.search(query_array, min(k, len(self._documents)))
            
            results = []
            for idx in indices[0]:
                if idx >= 0 and idx < len(self._doc_ids):
                    doc_id = self._doc_ids[idx]
                    results.append(self._documents[doc_id])
            return results
        else:
            # Fallback to cosine similarity
            return self._fallback_search(query_embedding, k)
    
    def _fallback_search(self, query_embedding: List[float], k: int) -> List[Document]:
        """Fallback cosine similarity search."""
        import math
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot / (norm_a * norm_b)
        
        similarities = [
            (doc_id, cosine_similarity(query_embedding, emb))
            for doc_id, emb in self._embeddings.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [self._documents[doc_id] for doc_id, _ in similarities[:k]]
    
    def delete(self, doc_id: str) -> bool:
        """Delete document (note: FAISS doesn't support deletion, mark as deleted)."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._embeddings[doc_id]
            if doc_id in self._doc_ids:
                self._doc_ids.remove(doc_id)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all documents and reset index."""
        self._documents.clear()
        self._embeddings.clear()
        self._doc_ids.clear()
        if self._faiss and self._index:
            self._index.reset()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save FAISS index and documents."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self._faiss and self._index:
            self._faiss.write_index(self._index, str(path / "index.faiss"))
        
        # Save documents and metadata
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump({
                "documents": self._documents,
                "embeddings": self._embeddings,
                "doc_ids": self._doc_ids,
                "dimension": self.dimension,
            }, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load FAISS index and documents."""
        path = Path(path)
        
        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            data = pickle.load(f)
            self._documents = data["documents"]
            self._embeddings = data["embeddings"]
            self._doc_ids = data["doc_ids"]
            self.dimension = data.get("dimension", self.dimension)
        
        # Load FAISS index
        if self._faiss and (path / "index.faiss").exists():
            self._index = self._faiss.read_index(str(path / "index.faiss"))
    
    @property
    def doc_count(self) -> int:
        return len(self._documents)


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB persistent vector store."""
    
    def __init__(
        self,
        collection_name: str = "neuralnode",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self._client = None
        self._collection = None
        
        try:
            import chromadb
            self._chroma = chromadb
            
            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.Client()
            
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except ImportError:
            self._chroma = None
            print("Warning: ChromaDB not installed. Install with: pip install chromadb")
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Add documents to ChromaDB."""
        if not self._collection:
            raise RuntimeError("ChromaDB not initialized")
        
        if not documents:
            return
        
        # Get embeddings
        if embeddings is None and self.embedding_function:
            embeddings = [self.embedding_function(doc.content) for doc in documents]
        elif embeddings is None:
            raise ValueError("Either provide embeddings or embedding_function")
        
        ids = [doc.id or f"doc_{i}" for i, doc in enumerate(documents)]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        # Update document IDs
        for i, doc in enumerate(documents):
            doc.id = ids[i]
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search ChromaDB."""
        if not self._collection:
            raise RuntimeError("ChromaDB not initialized")
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        
        documents = []
        for i, doc_id in enumerate(results["ids"][0]):
            doc = Document(
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                id=doc_id,
            )
            documents.append(doc)
        
        return documents
    
    def delete(self, doc_id: str) -> bool:
        """Delete document from ChromaDB."""
        if not self._collection:
            return False
        
        try:
            self._collection.delete(ids=[doc_id])
            return True
        except Exception:
            return False
    
    def clear(self) -> None:
        """Clear collection."""
        if self._collection:
            self._collection.delete(where={})  # Delete all
    
    def save(self, path: Union[str, Path]) -> None:
        """ChromaDB auto-persists, just verify."""
        pass  # PersistentClient already saves automatically
    
    def load(self, path: Union[str, Path]) -> None:
        """Reload collection."""
        if self._chroma and self.persist_directory:
            self._client = self._chroma.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    @property
    def doc_count(self) -> int:
        if self._collection:
            return self._collection.count()
        return 0


class PineconeVectorStore(BaseVectorStore):
    """Pinecone cloud vector store."""
    
    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str = "us-east1-gcp",
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        dimension: int = 1536,
    ):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.embedding_function = embedding_function
        self.dimension = dimension
        self._index = None
        self._pinecone = None
        
        try:
            import pinecone
            self._pinecone = pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if not exists
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                )
            
            self._index = pinecone.Index(index_name)
        except ImportError:
            print("Warning: Pinecone not installed. Install with: pip install pinecone-client")
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """Upload documents to Pinecone."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")
        
        if not documents:
            return
        
        if embeddings is None and self.embedding_function:
            embeddings = [self.embedding_function(doc.content) for doc in documents]
        elif embeddings is None:
            raise ValueError("Either provide embeddings or embedding_function")
        
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = doc.id or f"doc_{i}"
            doc.id = doc_id
            vectors.append({
                "id": doc_id,
                "values": emb,
                "metadata": {**doc.metadata, "content": doc.content},
            })
        
        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self._index.upsert(vectors=batch)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """Search Pinecone."""
        if not self._index:
            raise RuntimeError("Pinecone not initialized")
        
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
        )
        
        documents = []
        for match in results["matches"]:
            doc = Document(
                content=match["metadata"].get("content", ""),
                metadata={k: v for k, v in match["metadata"].items() if k != "content"},
                id=match["id"],
            )
            documents.append(doc)
        
        return documents
    
    def delete(self, doc_id: str) -> bool:
        """Delete from Pinecone."""
        if self._index:
            self._index.delete(ids=[doc_id])
            return True
        return False
    
    def clear(self) -> None:
        """Delete all vectors."""
        if self._index:
            self._index.delete(delete_all=True)
    
    def save(self, path: Union[str, Path]) -> None:
        """Pinecone is cloud-based, nothing to save locally."""
        pass
    
    def load(self, path: Union[str, Path]) -> None:
        """Reconnect to Pinecone."""
        if self._pinecone:
            self._pinecone.init(api_key=self.api_key, environment=self.environment)
            self._index = self._pinecone.Index(self.index_name)
    
    @property
    def doc_count(self) -> int:
        if self._index:
            stats = self._index.describe_index_stats()
            return stats.total_vector_count
        return 0
