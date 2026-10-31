"""
Comprehensive Vector Database Integrations for NeuralNode
Pinecone, Weaviate, Milvus, Chroma, Qdrant, Redis, FAISS
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to vector store."""
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar texts."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store."""
    
    def __init__(
        self,
        collection_name: str = "neuralnode",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        client: Optional[Any] = None,
    ):
        try:
            import chromadb
            from chromadb.config import Settings
            
            if client:
                self._client = client
            else:
                settings = Settings(
                    persist_directory=persist_directory,
                    anonymized_telemetry=False
                ) if persist_directory else None
                
                self._client = chromadb.Client(settings) if settings else chromadb.Client()
            
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
            
        except ImportError:
            raise ImportError("chromadb required. Install: pip install chromadb")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Chroma."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        self._collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Chroma."""
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            **kwargs
        )
        
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        self._collection.delete(ids=ids)
    
    def clear(self) -> None:
        """Clear collection."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(self._collection.name)


class FAISSVectorStore(BaseVectorStore):
    """FAISS local vector store."""
    
    def __init__(
        self,
        embedding_function: Any,
        index: Optional[Any] = None,
        docstore: Optional[Dict] = None,
        index_to_id: Optional[Dict] = None,
    ):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError("faiss required. Install: pip install faiss-cpu or faiss-gpu")
        
        self._embedding_function = embedding_function
        self._docstore = docstore or {}
        self._index_to_id = index_to_id or {}
        
        if index:
            self._index = index
        else:
            # Will be initialized on first add
            self._index = None
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to FAISS."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings
        embeddings = self._embedding_function.embed(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Initialize index if needed
        if self._index is None:
            dimension = embeddings.shape[1]
            self._index = self._faiss.IndexFlatL2(dimension)
        
        # Add to index
        start_idx = self._index.ntotal
        self._index.add(embeddings)
        
        # Store documents
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            index_id = start_idx + i
            self._docstore[doc_id] = {"content": text, "metadata": metadata}
            self._index_to_id[index_id] = doc_id
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search FAISS."""
        # Get query embedding
        query_embedding = self._embedding_function.embed([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Search
        distances, indices = self._index.search(query_embedding, k)
        
        documents = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            doc_id = self._index_to_id.get(int(idx))
            if doc_id:
                doc = self._docstore[doc_id]
                documents.append({
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "distance": float(distance)
                })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents (note: FAISS doesn't support deletion without rebuild)."""
        # Mark as deleted in docstore
        for doc_id in ids:
            if doc_id in self._docstore:
                self._docstore[doc_id]["deleted"] = True
    
    def clear(self) -> None:
        """Clear all documents."""
        self._index = None
        self._docstore = {}
        self._index_to_id = {}
    
    def save_local(self, folder_path: str) -> None:
        """Save FAISS index locally."""
        import os
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Save FAISS index
        self._faiss.write_index(self._index, os.path.join(folder_path, "index.faiss"))
        
        # Save the document store as JSON.  Loading pickle from a path that
        # users can download or exchange would allow arbitrary code execution.
        with open(os.path.join(folder_path, "docstore.json"), "w", encoding="utf-8") as f:
            json.dump({
                "docstore": self._docstore,
                "index_to_id": self._index_to_id
            }, f, ensure_ascii=False)
    
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embedding_function: Any,
        *,
        allow_unsafe_pickle: bool = False,
    ) -> "FAISSVectorStore":
        """Load a FAISS index from local JSON metadata.

        Legacy ``docstore.pkl`` files are rejected by default because
        unpickling untrusted data can execute arbitrary Python code.  Set
        ``allow_unsafe_pickle=True`` only for a trusted, locally-created
        legacy file and migrate it by saving the store again.
        """
        import os
        
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss required")
        
        # Load index
        index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        
        metadata_path = os.path.join(folder_path, "docstore.json")
        legacy_path = os.path.join(folder_path, "docstore.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif allow_unsafe_pickle and os.path.exists(legacy_path):
            import pickle
            with open(legacy_path, "rb") as f:
                data = pickle.load(f)
        elif os.path.exists(legacy_path):
            raise ValueError(
                "Refusing to load legacy docstore.pkl because pickle can execute code. "
                "Use a trusted file with allow_unsafe_pickle=True, then save it as JSON."
            )
        else:
            raise FileNotFoundError(f"No docstore.json found in {folder_path}")
        
        return cls(
            embedding_function=embedding_function,
            index=index,
            docstore=data["docstore"],
            index_to_id=data["index_to_id"]
        )


class PineconeVectorStore(BaseVectorStore):
    """Pinecone cloud vector store."""
    
    def __init__(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        namespace: Optional[str] = None,
        embedding_function: Optional[Any] = None,
    ):
        try:
            import pinecone
        except ImportError:
            raise ImportError("pinecone-client required. Install: pip install pinecone-client")
        
        api_key = api_key or pinecone.ApiKey()
        
        if environment:
            pinecone.init(api_key=api_key, environment=environment)
        
        self._index = pinecone.Index(index_name)
        self._namespace = namespace
        self._embedding_function = embedding_function
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Pinecone."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings
        if self._embedding_function:
            embeddings = self._embedding_function.embed(texts)
        else:
            raise ValueError("embedding_function required for Pinecone")
        
        # Prepare vectors
        vectors = []
        for i, (embedding, metadata, text) in enumerate(zip(embeddings, metadatas, texts)):
            metadata["text"] = text
            vectors.append((ids[i], embedding, metadata))
        
        # Upsert
        kwargs = {"namespace": self._namespace} if self._namespace else {}
        self._index.upsert(vectors=vectors, **kwargs)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Pinecone."""
        # Get query embedding
        if self._embedding_function:
            query_embedding = self._embedding_function.embed([query])[0]
        else:
            raise ValueError("embedding_function required")
        
        # Query
        search_kwargs = {"namespace": self._namespace} if self._namespace else {}
        results = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            **search_kwargs
        )
        
        documents = []
        for match in results["matches"]:
            documents.append({
                "id": match["id"],
                "content": match["metadata"].get("text", ""),
                "metadata": {k: v for k, v in match["metadata"].items() if k != "text"},
                "score": match["score"]
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        kwargs = {"namespace": self._namespace} if self._namespace else {}
        self._index.delete(ids=ids, **kwargs)
    
    def clear(self) -> None:
        """Clear all documents."""
        self._index.delete(delete_all=True, namespace=self._namespace)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store."""
    
    def __init__(
        self,
        collection_name: str = "neuralnode",
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        vector_size: int = 1536,
    ):
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError("qdrant-client required. Install: pip install qdrant-client")
        
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            location = location or ":memory:"
            self._client = QdrantClient(location=location)
        
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._vector_size = vector_size
        
        # Create collection if doesn't exist
        try:
            self._client.get_collection(collection_name)
        except:
            from qdrant_client.models import Distance, VectorParams
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Qdrant."""
        from qdrant_client.models import PointStruct
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings
        if self._embedding_function:
            embeddings = self._embedding_function.embed(texts)
        else:
            raise ValueError("embedding_function required")
        
        # Prepare points
        points = []
        for i, (embedding, metadata, text) in enumerate(zip(embeddings, metadatas, texts)):
            metadata["text"] = text
            points.append(PointStruct(
                id=ids[i],
                vector=embedding,
                payload=metadata
            ))
        
        # Upsert
        self._client.upsert(
            collection_name=self._collection_name,
            points=points
        )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Qdrant."""
        # Get query embedding
        if self._embedding_function:
            query_embedding = self._embedding_function.embed([query])[0]
        else:
            raise ValueError("embedding_function required")
        
        # Search
        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        documents = []
        for result in results:
            documents.append({
                "id": str(result.id),
                "content": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"},
                "score": result.score
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids)
        )
    
    def clear(self) -> None:
        """Clear collection."""
        self._client.delete_collection(self._collection_name)
        from qdrant_client.models import Distance, VectorParams
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE)
        )


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store."""
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        index_name: str = "NeuralNode",
        text_key: str = "text",
        embedding_function: Optional[Any] = None,
    ):
        try:
            import weaviate
        except ImportError:
            raise ImportError("weaviate-client required. Install: pip install weaviate-client")
        
        auth_config = weaviate.auth.AuthApiKey(api_key) if api_key else None
        
        if url:
            self._client = weaviate.Client(url=url, auth_client_secret=auth_config)
        else:
            raise ValueError("URL required for Weaviate")
        
        self._index_name = index_name
        self._text_key = text_key
        self._embedding_function = embedding_function
        
        # Create schema if doesn't exist
        self._create_schema()
    
    def _create_schema(self) -> None:
        """Create Weaviate schema."""
        schema = {
            "class": self._index_name,
            "properties": [
                {"name": self._text_key, "dataType": ["text"]},
            ]
        }
        
        try:
            self._client.schema.create_class(schema)
        except:
            pass  # Class may already exist
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Weaviate."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add objects
        with self._client.batch as batch:
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                data_object = {self._text_key: text, **metadata}
                batch.add_data_object(
                    data_object=data_object,
                    class_name=self._index_name,
                    uuid=ids[i]
                )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Weaviate."""
        if self._embedding_function:
            # Vector search
            query_embedding = self._embedding_function.embed([query])[0]
            results = (
                self._client.query
                .get(self._index_name, [self._text_key])
                .with_near_vector({"vector": query_embedding})
                .with_limit(k)
                .with_additional(["certainty"])
                .do()
            )
        else:
            # BM25 search
            results = (
                self._client.query
                .get(self._index_name, [self._text_key])
                .with_bm25(query={"query": query})
                .with_limit(k)
                .do()
            )
        
        documents = []
        for item in results["data"]["Get"][self._index_name]:
            documents.append({
                "id": item.get("_additional", {}).get("id", ""),
                "content": item.get(self._text_key, ""),
                "metadata": {k: v for k, v in item.items() if k not in [self._text_key, "_additional"]},
                "score": item.get("_additional", {}).get("certainty")
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        for doc_id in ids:
            self._client.data_object.delete(uuid=doc_id, class_name=self._index_name)
    
    def clear(self) -> None:
        """Clear all documents."""
        self._client.schema.delete_all()
        self._create_schema()


class MilvusVectorStore(BaseVectorStore):
    """Milvus/Zilliz vector store."""
    
    def __init__(
        self,
        collection_name: str = "neuralnode",
        uri: str = "http://localhost:19530",
        token: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        vector_dim: int = 1536,
    ):
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError:
            raise ImportError("pymilvus required. Install: pip install pymilvus")
        
        self._client = MilvusClient(uri=uri, token=token)
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._vector_dim = vector_dim
        
        # Create collection if doesn't exist
        if not self._client.has_collection(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                dimension=vector_dim,
                metric_type="COSINE"
            )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Milvus."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings
        if self._embedding_function:
            embeddings = self._embedding_function.embed(texts)
        else:
            raise ValueError("embedding_function required")
        
        # Prepare data
        data = []
        for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
            data.append({
                "id": ids[i],
                "vector": embedding,
                "text": text,
                **metadata
            })
        
        # Insert
        self._client.insert(collection_name=self._collection_name, data=data)
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Milvus."""
        # Get query embedding
        if self._embedding_function:
            query_embedding = self._embedding_function.embed([query])[0]
        else:
            raise ValueError("embedding_function required")
        
        # Search
        results = self._client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            limit=k,
            output_fields=["text"]
        )
        
        documents = []
        for hits in results:
            for hit in hits:
                documents.append({
                    "id": hit["id"],
                    "content": hit["entity"]["text"],
                    "metadata": {k: v for k, v in hit["entity"].items() if k not in ["text", "vector"]},
                    "score": hit["distance"]
                })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        self._client.delete(collection_name=self._collection_name, ids=ids)
    
    def clear(self) -> None:
        """Clear collection."""
        self._client.drop_collection(self._collection_name)
        self._client.create_collection(
            collection_name=self._collection_name,
            dimension=self._vector_dim,
            metric_type="COSINE"
        )


class RedisVectorStore(BaseVectorStore):
    """Redis vector store."""
    
    def __init__(
        self,
        index_name: str = "neuralnode",
        redis_url: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        vector_dim: int = 1536,
    ):
        try:
            from redis import Redis
            from redis.commands.search.field import VectorField, TextField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        except ImportError:
            raise ImportError("redis required. Install: pip install redis")
        
        redis_url = redis_url or "redis://localhost:6379"
        self._client = Redis.from_url(redis_url)
        self._index_name = index_name
        self._embedding_function = embedding_function
        self._vector_dim = vector_dim
        
        # Create index if doesn't exist
        try:
            self._client.ft(index_name).info()
        except:
            schema = (
                TextField("text"),
                VectorField(
                    "vector",
                    "FLAT",
                    {"TYPE": "FLOAT32", "DIM": vector_dim, "DISTANCE_METRIC": "COSINE"}
                )
            )
            self._client.ft(index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[f"{index_name}:"], index_type=IndexType.HASH)
            )
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to Redis."""
        import numpy as np
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Get embeddings
        if self._embedding_function:
            embeddings = self._embedding_function.embed(texts)
        else:
            raise ValueError("embedding_function required")
        
        # Store
        pipe = self._client.pipeline()
        for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
            key = f"{self._index_name}:{ids[i]}"
            vector = np.array(embedding, dtype=np.float32).tobytes()
            
            data = {"text": text, "vector": vector, **metadata}
            pipe.hset(key, mapping=data)
        
        pipe.execute()
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search Redis."""
        import numpy as np
        
        # Get query embedding
        if self._embedding_function:
            query_embedding = self._embedding_function.embed([query])[0]
        else:
            raise ValueError("embedding_function required")
        
        # Search
        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
        
        results = self._client.ft(self._index_name).search(
            f"*=>[KNN {k} @vector $vec]",
            query_params={"vec": query_vector},
            return_fields=["text"],
        )
        
        documents = []
        for doc in results.docs:
            documents.append({
                "id": doc.id.replace(f"{self._index_name}:", ""),
                "content": doc.text,
                "metadata": {},
                "score": getattr(doc, "vector_score", None)
            })
        
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents."""
        for doc_id in ids:
            key = f"{self._index_name}:{doc_id}"
            self._client.delete(key)
    
    def clear(self) -> None:
        """Clear all documents."""
        for key in self._client.scan_iter(match=f"{self._index_name}:*"):
            self._client.delete(key)


# Convenience functions
def create_vector_store(
    store_type: str,
    embedding_function: Any,
    **kwargs
) -> BaseVectorStore:
    """Create a vector store by type."""
    store_types = {
        "chroma": ChromaVectorStore,
        "faiss": FAISSVectorStore,
        "pinecone": PineconeVectorStore,
        "qdrant": QdrantVectorStore,
        "weaviate": WeaviateVectorStore,
        "milvus": MilvusVectorStore,
        "redis": RedisVectorStore,
    }
    
    store_cls = store_types.get(store_type.lower())
    if not store_cls:
        raise ValueError(f"Unknown store type: {store_type}. Available: {list(store_types.keys())}")
    
    if store_type.lower() in ["faiss", "pinecone", "qdrant", "weaviate", "milvus", "redis"]:
        return store_cls(embedding_function=embedding_function, **kwargs)
    else:
        return store_cls(embedding_function=embedding_function, **kwargs)


__all__ = [
    # Base
    "BaseVectorStore",
    
    # Implementations
    "ChromaVectorStore",
    "FAISSVectorStore",
    "PineconeVectorStore",
    "QdrantVectorStore",
    "WeaviateVectorStore",
    "MilvusVectorStore",
    "RedisVectorStore",
    
    # Convenience functions
    "create_vector_store",
]
