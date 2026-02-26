"""Advanced Vector Store implementations using FAISS and ChromaDB."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle
import json
from pathlib import Path


@dataclass
class Document:
    """Document with content and metadata."""
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FAISSVectorStore:
    """High-performance vector store using Facebook AI Similarity Search (FAISS)."""
    
    def __init__(self, embedding_dim: int = 384, index_type: str = "flat", use_gpu: bool = False):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: "flat" (exact), "ivf" (inverted file), "hnsw" (approximate)
            use_gpu: Use GPU acceleration if available
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu and self._check_gpu_available()
        
        self.documents: List[Document] = []
        self.index = None
        self._faiss = None
        self._init_faiss()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available."""
        try:
            import faiss
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            self._faiss = faiss
            
            # Create index based on type
            if self.index_type == "flat":
                # Exact search - best accuracy, slower for large datasets
                if self.use_gpu:
                    self.index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), self.embedding_dim)
                else:
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            elif self.index_type == "ivf":
                # Inverted file index - faster search, slightly less accurate
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                nlist = 100  # Number of clusters
                if self.use_gpu:
                    self.index = faiss.GpuIndexIVFFlat(faiss.StandardGpuResources(), quantizer, nlist, faiss.METRIC_INNER_PRODUCT)
                else:
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.nprobe = 10  # Number of clusters to search
            
            elif self.index_type == "hnsw":
                # Hierarchical Navigable Small World - very fast, good accuracy
                self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                self.index.hnsw.efConstruction = 40
                self.index.hnsw.efSearch = 16
            
            else:
                raise ValueError(f"Unknown index type: {self.index_type}")
            
        except ImportError:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu (or faiss-gpu)"
            )
    
    def add(self, document: Document) -> None:
        """Add document to store."""
        if document.embedding is None:
            raise ValueError("Document must have an embedding")
        
        # Ensure embedding is correct shape
        embedding = np.array(document.embedding, dtype=np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity (FAISS uses inner product)
        faiss.normalize_L2(embedding)
        
        # Add to index
        self.index.add(embedding)
        
        # Store document
        self.documents.append(document)
    
    def add_batch(self, documents: List[Document]) -> None:
        """Add multiple documents efficiently."""
        if not documents:
            return
        
        embeddings = []
        valid_docs = []
        
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
                valid_docs.append(doc)
        
        if not embeddings:
            return
        
        # Stack embeddings
        embeddings = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents
        self.documents.extend(valid_docs)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filter_metadata: Optional metadata filters
        
        Returns:
            List of (document, score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Prepare query
        query = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        distances, indices = self.index.search(query, k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            
            doc = self.documents[idx]
            
            # Apply metadata filter
            if filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            
            results.append((doc, float(distance)))
        
        return results
    
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID (mark as deleted - FAISS doesn't support true deletion)."""
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                doc.metadata['_deleted'] = True
                return True
        return False
    
    def save(self, path: str) -> None:
        """Save index and documents to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents (without embeddings to save space)
        docs_to_save = []
        for doc in self.documents:
            docs_to_save.append({
                'content': doc.content,
                'metadata': doc.metadata,
                'id': doc.id,
            })
        
        with open(path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(docs_to_save, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str, embedding_fn=None) -> "FAISSVectorStore":
        """Load index and documents from disk."""
        path = Path(path)
        
        # Load documents
        with open(path / "documents.json", 'r', encoding='utf-8') as f:
            docs_data = json.load(f)
        
        # Determine embedding dimension from index
        import faiss
        index = faiss.read_index(str(path / "index.faiss"))
        dim = index.d
        
        # Create store
        store = cls(embedding_dim=dim)
        store.index = index
        
        # Restore documents
        for doc_data in docs_data:
            store.documents.append(Document(
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                id=doc_data['id'],
            ))
        
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_type,
            "index_size": self.index.ntotal if self.index else 0,
            "gpu_enabled": self.use_gpu,
        }


class ChromaDBStore:
    """Vector store using ChromaDB with persistent storage."""
    
    def __init__(self, collection_name: str = "default", persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (optional)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._init_chroma()
    
    def _init_chroma(self):
        """Initialize ChromaDB client."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self.persist_directory:
                self.client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory,
                ))
            else:
                self.client = chromadb.Client()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )
    
    def add(self, document: Document) -> None:
        """Add document to store."""
        doc_id = document.id or str(len(self.collection.get()['ids']) if self.collection.get()['ids'] else 0)
        
        self.collection.add(
            ids=[doc_id],
            documents=[document.content],
            embeddings=[document.embedding.tolist()] if document.embedding is not None else None,
            metadatas=[document.metadata or {}]
        )
    
    def add_batch(self, documents: List[Document]) -> None:
        """Add multiple documents."""
        ids = [doc.id or str(i) for i, doc in enumerate(documents)]
        texts = [doc.content for doc in documents]
        embeddings = [doc.embedding.tolist() for doc in documents if doc.embedding is not None]
        metadatas = [doc.metadata or {} for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings if embeddings else None,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        where_clause = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_clause,
        )
        
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                id=results['ids'][0][i],
            )
            distance = results['distances'][0][i] if results['distances'] else 0.0
            documents.append((doc, distance))
        
        return documents
    
    def delete(self, doc_id: str) -> bool:
        """Delete document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except:
            return False
    
    def save(self) -> None:
        """Persist to disk (automatic with ChromaDB)."""
        if hasattr(self.client, 'persist'):
            self.client.persist()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }


# Backward compatibility - SimpleVectorStore with graceful degradation
class SimpleVectorStore:
    """
    Simple in-memory vector store with automatic upgrade to FAISS when available.
    Provides graceful degradation when FAISS is not installed.
    """
    
    def __init__(self, embedding_dim: int = 384, auto_upgrade: bool = True):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            auto_upgrade: Automatically use FAISS if available
        """
        self.embedding_dim = embedding_dim
        self.auto_upgrade = auto_upgrade
        
        # Try to use FAISS first
        if auto_upgrade:
            try:
                self._backend = FAISSVectorStore(embedding_dim)
                self._backend_type = "faiss"
                print("✅ Using FAISS for fast vector search")
                return
            except ImportError:
                pass
            
            # Try ChromaDB
            try:
                self._backend = ChromaDBStore(embedding_dim=embedding_dim)
                self._backend_type = "chromadb"
                print("✅ Using ChromaDB for persistent vector storage")
                return
            except ImportError:
                pass
        
        # Fallback to simple numpy implementation
        self._backend = self._SimpleBackend(embedding_dim)
        self._backend_type = "simple"
        print("⚠️  Using simple numpy backend (install faiss-cpu or chromadb for better performance)")
    
    class _SimpleBackend:
        """Simple numpy-based backend for when FAISS/ChromaDB not available."""
        
        def __init__(self, dim: int):
            self.dim = dim
            self.documents: List[Document] = []
        
        def add(self, doc: Document):
            self.documents.append(doc)
        
        def add_batch(self, docs: List[Document]):
            self.documents.extend(docs)
        
        def search(self, query_embedding: np.ndarray, k: int = 5, filter_metadata=None):
            if not self.documents:
                return []
            
            # Compute cosine similarity
            query = query_embedding / np.linalg.norm(query_embedding)
            
            similarities = []
            for doc in self.documents:
                if doc.embedding is None:
                    continue
                
                # Check metadata filter
                if filter_metadata:
                    if not all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                        continue
                
                doc_vec = doc.embedding / np.linalg.norm(doc.embedding)
                sim = np.dot(query, doc_vec)
                similarities.append((doc, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
        
        def delete(self, doc_id: str):
            for i, doc in enumerate(self.documents):
                if doc.id == doc_id:
                    del self.documents[i]
                    return True
            return False
        
        def save(self, path: str):
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "simple_store.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
        
        def get_stats(self):
            return {
                "total_documents": len(self.documents),
                "backend": "simple_numpy",
            }
    
    def add(self, document: Document) -> None:
        """Add document."""
        self._backend.add(document)
    
    def add_batch(self, documents: List[Document]) -> None:
        """Add multiple documents."""
        self._backend.add_batch(documents)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents."""
        return self._backend.search(query_embedding, k, filter_metadata)
    
    def delete(self, doc_id: str) -> bool:
        """Delete document."""
        return self._backend.delete(doc_id)
    
    def save(self, path: str) -> None:
        """Save to disk."""
        self._backend.save(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        stats = self._backend.get_stats()
        stats["backend_type"] = self._backend_type
        return stats


# Convenience functions
def create_vector_store(
    backend: str = "auto",
    embedding_dim: int = 384,
    **kwargs
):
    """
    Create vector store with specified backend.
    
    Args:
        backend: "auto", "faiss", "chromadb", or "simple"
        embedding_dim: Embedding dimension
        **kwargs: Additional backend-specific arguments
    
    Returns:
        Vector store instance
    """
    if backend == "faiss" or (backend == "auto" and _faiss_available()):
        return FAISSVectorStore(embedding_dim, **kwargs)
    elif backend == "chromadb" or (backend == "auto" and _chromadb_available()):
        return ChromaDBStore(**kwargs)
    else:
        return SimpleVectorStore(embedding_dim, auto_upgrade=False)


def _faiss_available() -> bool:
    """Check if FAISS is available."""
    try:
        import faiss
        return True
    except ImportError:
        return False


def _chromadb_available() -> bool:
    """Check if ChromaDB is available."""
    try:
        import chromadb
        return True
    except ImportError:
        return False
