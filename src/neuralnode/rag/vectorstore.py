"""RAG (Retrieval-Augmented Generation) module."""

from typing import Any, Callable, Dict, List, Optional, Union
import re


class Document:
    """A document for RAG."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.id = id

    def __repr__(self) -> str:
        return f"Document(id={self.id}, content={self.content[:50]}...)"


class TextSplitter:
    """Split text into chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        # Simple character-based splitting
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a natural boundary
            if end < len(text):
                # Look for separator within overlap window
                search_start = max(start, end - self.chunk_overlap)
                separator_pos = text.rfind(self.separator, search_start, end)
                if separator_pos != -1:
                    end = separator_pos + len(self.separator)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        for doc in documents:
            texts = self.split(doc.content)
            for i, text in enumerate(texts):
                chunk = Document(
                    content=text,
                    metadata={**doc.metadata, "chunk_index": i},
                    id=f"{doc.id}_chunk_{i}" if doc.id else None,
                )
                chunks.append(chunk)
        return chunks


class SimpleVectorStore:
    """Simple in-memory vector store."""

    def __init__(self, embedding_function: Optional[Callable[[str], List[float]]] = None):
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []
        self.embedding_function = embedding_function

    def add(self, documents: Union[Document, List[Document]]) -> None:
        """Add documents to the store."""
        if isinstance(documents, Document):
            documents = [documents]
        
        for doc in documents:
            self.documents.append(doc)
            if self.embedding_function:
                vector = self.embedding_function(doc.content)
                self.vectors.append(vector)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            embedding_function: Optional override embedding function
            
        Returns:
            List of most similar documents
        """
        embed_fn = embedding_function or self.embedding_function
        if not embed_fn:
            raise ValueError("No embedding function provided")
        
        if not self.documents:
            return []
        
        query_vector = embed_fn(query)
        
        # Calculate cosine similarity
        similarities = []
        for vector in self.vectors:
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append(similarity)
        
        # Get top k
        top_indices = sorted(
            range(len(similarities)),
            key=lambda i: similarities[i],
            reverse=True,
        )[:k]
        
        return [self.documents[i] for i in top_indices]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        return dot_product / (magnitude_a * magnitude_b)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents.clear()
        self.vectors.clear()


class RAG:
    """Retrieval-Augmented Generation system."""

    def __init__(
        self,
        llm: Any,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        vector_store: Optional[SimpleVectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
    ):
        """
        Initialize RAG system.
        
        Args:
            llm: NeuralNode instance or similar
            embedding_function: Function to generate embeddings
            vector_store: Optional custom vector store
            text_splitter: Optional custom text splitter
        """
        self.llm = llm
        self.embedding_function = embedding_function
        self.vector_store = vector_store or SimpleVectorStore(embedding_function)
        self.text_splitter = text_splitter or TextSplitter()

    def add_documents(
        self,
        documents: Union[str, Document, List[Union[str, Document]]],
        split: bool = True,
    ) -> None:
        """
        Add documents to the RAG system.
        
        Args:
            documents: String, Document, or list of either
            split: Whether to split documents into chunks
        """
        # Normalize to list of Documents
        if isinstance(documents, (str, Document)):
            documents = [documents]
        
        doc_objects = []
        for doc in documents:
            if isinstance(doc, str):
                doc_objects.append(Document(content=doc))
            else:
                doc_objects.append(doc)
        
        # Split if requested
        if split:
            doc_objects = self.text_splitter.split_documents(doc_objects)
        
        self.vector_store.add(doc_objects)

    def query(
        self,
        question: str,
        k: int = 4,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Query the RAG system.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            system_prompt: Optional custom system prompt
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=k)
        
        if not docs:
            context = "No relevant documents found."
        else:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                context_parts.append(f"Document {i}:\n{doc.content}")
            context = "\n\n".join(context_parts)
        
        # Build prompt with context
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Generate response
        if hasattr(self.llm, 'chat'):
            response = self.llm.chat(prompt, system=system_prompt)
            return response.text if hasattr(response, 'text') else str(response)
        else:
            return str(self.llm(prompt))

    async def aquery(
        self,
        question: str,
        k: int = 4,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Async version of query."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.query(question, k, system_prompt)
        )


# File loaders
class FileLoader:
    """Load documents from files."""

    @staticmethod
    def load_text(path: str) -> Document:
        """Load text file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return Document(
            content=content,
            metadata={"source": path, "type": "text"},
        )

    @staticmethod
    def load_pdf(path: str) -> List[Document]:
        """Load PDF file."""
        try:
            import pypdf
            
            documents = []
            with open(path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append(Document(
                            content=text,
                            metadata={
                                "source": path,
                                "type": "pdf",
                                "page": i + 1,
                            },
                        ))
            return documents
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install with: pip install pypdf"
            )

    @staticmethod
    def load_web(url: str) -> Document:
        """Load web page."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator="\n", strip=True)
            
            return Document(
                content=text,
                metadata={"source": url, "type": "web"},
            )
        except ImportError:
            raise ImportError(
                "requests and beautifulsoup4 are required for web loading. "
                "Install with: pip install requests beautifulsoup4"
            )
