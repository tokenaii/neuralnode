"""NeuralNode RAG — Retrieval-Augmented Generation.

Provides document loaders, text splitting, vector stores, and a
high-level RAG pipeline.

Example::

    from neuralnode.rag import RAG, create_rag

    rag = create_rag(llm, store="memory")
    rag.add_documents(["report.pdf", "notes.txt"])
    rag.add_text("Some extra raw text here.")
    answer = rag.query("What are the main findings?")

    # Return sources without an LLM call
    chunks = rag.search("main findings")
    for c in chunks:
        print(c.content, c.metadata.get("source"))
"""

import os
import re
import hashlib
import math
import json
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

from ..core import BaseProvider, Message, ensure_llm_response, ensure_streaming_chunks
from ..utils.logger import get_logger, log_execution



# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Document:
    """A raw document before splitting."""
    content: str
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source
        if not self.source:
            self.source = self.metadata.get("source")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize document for backward compatibility."""
        return {
            "content": self.content,
            "source": self.source,
            "metadata": dict(self.metadata),
            "doc_id": self.doc_id,
            "embedding": self.embedding,
        }


@dataclass
class Chunk:
    """A chunk of a document after splitting."""
    content: str
    doc_id: str = "unknown"
    chunk_index: int = 0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source
        if not self.source:
            self.source = self.metadata.get("source")


# ── Text Splitters ─────────────────────────────────────────────────────────────

class TextSplitter:
    """Split text into overlapping chunks.

    Args:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters to repeat between consecutive chunks.
        separator: Primary split character sequence.

    Example::

        splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(long_text)
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        length_function: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.length_function = length_function
        self.logger = get_logger()

    def split_text(self, text: str) -> List[str]:
        """Split a text string into chunks."""
        if self.length_function(text) <= self.chunk_size:
            return [text]

        splits = text.split(self.separator)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for split in splits:
            split_len = self.length_function(split)
            if current_len + split_len > self.chunk_size:
                if current:
                    chunks.append(self.separator.join(current))
                if self.chunk_overlap > 0 and current:
                    overlap = self._get_overlap(current)
                    current = [overlap, split]
                    current_len = self.length_function(overlap) + split_len
                else:
                    current = [split]
                    current_len = split_len
            else:
                current.append(split)
                current_len += split_len

        if current:
            chunks.append(self.separator.join(current))
        return chunks

    def split(self, text: str) -> List[str]:
        """Backward-compatible alias for ``split_text``."""
        return self.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """Split a list of Document objects into Chunk objects."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            texts = self.split_text(doc.content)
            for i, text in enumerate(texts):
                all_chunks.append(Chunk(
                    content=text,
                    doc_id=doc.doc_id,  # type: ignore
                    chunk_index=i,
                    metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(texts)},
                ))
        self.logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def _get_overlap(self, chunks: List[str]) -> str:
        text = self.separator.join(chunks)
        return text[-self.chunk_overlap:] if len(text) > self.chunk_overlap else text


# Aliases for compatibility
BaseTextSplitter = TextSplitter
CharacterTextSplitter = TextSplitter


class RecursiveCharacterTextSplitter(TextSplitter):
    """Recursively splits on multiple separators (\n\n → \n → space)."""

    def __init__(self, **kwargs):
        separators = kwargs.pop("separators", ["\n\n", "\n", ". ", " ", ""])
        super().__init__(**kwargs)
        self._separators = separators

    def split_text(self, text: str) -> List[str]:
        return self._split_recursive(text, self._separators)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if self.length_function(text) <= self.chunk_size:
            return [text]
        sep = separators[0] if separators else ""
        parts = text.split(sep) if sep else list(text)
        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for part in parts:
            part_len = self.length_function(part + sep)
            if current_len + part_len > self.chunk_size:
                if not current and part_len > self.chunk_size and len(separators) > 1:
                    # Part itself exceeds chunk_size and cannot be split by the
                    # current separator; recurse with finer separators.
                    chunks.extend(self._split_recursive(part, separators[1:]))
                    continue
                if current:
                    chunk_text = sep.join(current)
                    if self.length_function(chunk_text) > self.chunk_size and len(separators) > 1:
                        chunks.extend(self._split_recursive(chunk_text, separators[1:]))
                    else:
                        chunks.append(chunk_text)
                current = [part]
                current_len = part_len
            else:
                current.append(part)
                current_len += part_len
        if current:
            chunks.append(sep.join(current))
        return chunks


class TokenTextSplitter(TextSplitter):
    """Split by approximate token count (1 token ≈ 4 chars)."""
    def __init__(self, max_tokens: int = 512, **kwargs):
        super().__init__(chunk_size=max_tokens * 4, **kwargs)


class MarkdownHeaderTextSplitter:
    """Split markdown by headers."""

    def split_text(self, text: str) -> List[str]:
        sections = re.split(r"\n(?=#{1,6} )", text)
        return [s.strip() for s in sections if s.strip()]


# ── Document Loaders ───────────────────────────────────────────────────────────

class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, source: Union[str, Path]) -> List[Document]:
        pass

    @abstractmethod
    def supports(self, source: Union[str, Path]) -> bool:
        pass


class TextLoader(BaseDocumentLoader):
    """Load plain text / code / markdown / JSON / CSV files."""

    EXTENSIONS = {
        ".txt", ".md", ".rst", ".log", ".py", ".js", ".ts", ".html",
        ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh",
        ".json", ".csv",
    }

    def load(self, source: Union[str, Path]) -> List[Document]:
        path = Path(source)
        if not path.exists():
            return []
        content = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(
            content=content,
            source=str(path),
            metadata={"source": str(path), "filename": path.name, "type": "text"},
        )]

    def supports(self, source: Union[str, Path]) -> bool:
        return Path(source).suffix.lower() in self.EXTENSIONS


class PDFLoader(BaseDocumentLoader):
    """Load PDF files (requires ``pypdf``)."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        try:
            import pypdf
        except ImportError:
            raise ImportError("Install pypdf: pip install pypdf")

        path = Path(source)
        reader = pypdf.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- Page {i + 1} ---\n{text}")
        return [Document(
            content="\n\n".join(pages),
            metadata={"source": str(path), "filename": path.name, "type": "pdf", "pages": len(reader.pages)},
        )]

    def supports(self, source: Union[str, Path]) -> bool:
        return Path(source).suffix.lower() == ".pdf"


class DocxLoader(BaseDocumentLoader):
    """Load Word .docx files (requires ``python-docx``)."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        try:
            import docx
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = docx.Document(str(source))
        content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [Document(
            content=content,
            metadata={"source": str(source), "filename": Path(source).name, "type": "docx"},
        )]

    def supports(self, source: Union[str, Path]) -> bool:
        return Path(source).suffix.lower() in {".docx", ".doc"}


class WebLoader(BaseDocumentLoader):
    """Scrape a URL and return its text content."""

    def __init__(self, max_length: int = 15000):
        self.max_length = max_length

    def load(self, source: Union[str, Path]) -> List[Document]:
        import urllib.request, ssl
        url = str(source)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        headers = {"User-Agent": "Mozilla/5.0 (NeuralNode RAG)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            html = r.read().decode("utf-8", errors="ignore")
        text = self._strip_html(html)[: self.max_length]
        return [Document(content=text, metadata={"source": url, "type": "web"})]

    def _strip_html(self, html: str) -> str:
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<(p|div|h[1-6]|br|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "", html)
        return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", html)).strip()

    def supports(self, source: Union[str, Path]) -> bool:
        return str(source).startswith(("http://", "https://"))


class CSVLoader(BaseDocumentLoader):
    """Load CSV files dict-by-dict."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        import csv
        path = Path(source)
        if not path.exists():
            return []
        documents = []
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                    documents.append(Document(
                        content=content,
                        source=str(path),
                        metadata={"source": str(path), "row": i, "type": "csv", **row}
                    ))
            return documents
        except Exception as e:
            logger.error(f"Failed to load CSV {path}: {e}")
            return []

    def supports(self, source: Union[str, Path]) -> bool:
        return Path(source).suffix.lower() == ".csv"


class JSONLoader(BaseDocumentLoader):
    """Load JSON files parsing arrays modularly."""

    def load(self, source: Union[str, Path]) -> List[Document]:
        path = Path(source)
        if not path.exists():
            return []
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(content)
            documents = []
            if isinstance(data, list):
                for i, item in enumerate(data):
                    item_str = json.dumps(item, indent=2) if isinstance(item, (dict, list)) else str(item)
                    documents.append(Document(
                        content=item_str,
                        source=str(path),
                        metadata={"source": str(path), "index": i, "type": "json"}
                    ))
            else:
                documents.append(Document(
                    content=json.dumps(data, indent=2),
                    source=str(path),
                    metadata={"source": str(path), "type": "json"}
                ))
            return documents
        except Exception as e:
            logger.error(f"Failed to load JSON {path}: {e}")
            return []

    def supports(self, source: Union[str, Path]) -> bool:
        return Path(source).suffix.lower() == ".json"


class DocumentLoader:
    """Universal document loader with auto-detection.

    Example::

        loader = DocumentLoader()
        docs = loader.load("file.pdf")
        docs = loader.load("https://example.com")
        docs = loader.load_directory("./my_docs/")
    """

    def __init__(self):
        self.loaders: List[BaseDocumentLoader] = [
            TextLoader(), PDFLoader(), DocxLoader(), WebLoader(), CSVLoader(), JSONLoader()
        ]
        self.logger = get_logger()

    def load(self, source: Union[str, Path]) -> List[Document]:
        for loader in self.loaders:
            if loader.supports(source):
                return loader.load(source)
        raise ValueError(f"No loader available for: {source}")

    def load_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
    ) -> List[Document]:
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return []
        docs = []
        glob_fn = path.rglob if recursive else path.glob
        for file_path in glob_fn(pattern):
            if file_path.is_file():
                try:
                    docs.extend(self.load(file_path))
                except Exception as e:
                    self.logger.warning(f"Error loading {file_path}: {e}")
        self.logger.info(f"Loaded {len(docs)} documents from {directory}")
        return docs


# Aliases
JSONLoader = TextLoader
CSVLoader = TextLoader
DirectoryLoader = DocumentLoader


def load_documents(sources: List[Union[str, Path]]) -> List[Document]:
    """Convenience function to load from a list of sources."""
    if isinstance(sources, (str, Path)):
        sources = [sources]
    loader = DocumentLoader()
    docs: List[Document] = []
    for src in sources:
        path = Path(src) if not str(src).startswith("http") else src
        if isinstance(path, Path) and path.is_dir():
            docs.extend(loader.load_directory(path))
        else:
            docs.extend(loader.load(src))
    return docs


def split_documents(docs: List[Document], **splitter_kwargs) -> List[Chunk]:
    """Convenience function to split documents."""
    return TextSplitter(**splitter_kwargs).split_documents(docs)


# ── Vector Stores ──────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None: ...
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Chunk]: ...
    @abstractmethod
    def clear(self) -> None: ...
    @abstractmethod
    def count(self) -> int: ...


class InMemoryVectorStore(BaseVectorStore):
    """Simple cosine-similarity vector store (no external deps)."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.logger = get_logger()

    def add(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        self.logger.info(f"InMemoryVectorStore: {len(self.chunks)} total chunks")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Chunk]:
        scored = [
            (ch, self._cosine(query_embedding, ch.embedding))
            for ch in self.chunks
            if ch.embedding
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [ch for ch, _ in scored[:k]]

    def clear(self) -> None:
        self.chunks = []

    def count(self) -> int:
        return len(self.chunks)

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0


class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store (requires ``chromadb``)."""

    def __init__(
        self,
        collection_name: str = "neuralnode",
        persist_directory: Optional[str] = None,
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")
        self.client = (
            chromadb.PersistentClient(path=persist_directory)
            if persist_directory
            else chromadb.Client()
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.logger = get_logger()

    def add(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        ids = [f"{ch.doc_id}_{ch.chunk_index}" for ch in chunks]
        docs = [ch.content for ch in chunks]
        metas = [ch.metadata for ch in chunks]
        embs = [ch.embedding for ch in chunks if ch.embedding]
        if embs and len(embs) == len(chunks):
            self.collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)
        else:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
        self.logger.info(f"ChromaDB: added {len(chunks)} chunks")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Chunk]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        chunks: List[Chunk] = []
        for i, doc in enumerate(results["documents"][0]):
            meta = (results["metadatas"] or [[]])[0][i] if results.get("metadatas") else {}
            chunks.append(Chunk(
                content=doc,
                doc_id=meta.get("doc_id", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                metadata=meta,
            ))
        return chunks

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def count(self) -> int:
        return self.collection.count()


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store (requires ``faiss-cpu`` or ``faiss-gpu``)."""

    def __init__(self, embedding_dim: Optional[int] = None):
        try:
            import faiss as _faiss  # noqa: F401
        except ImportError:
            raise ImportError("Install faiss: pip install faiss-cpu")
        self._faiss = _faiss
        self.embedding_dim = embedding_dim
        self._index = None
        self.chunks: List[Chunk] = []
        self.logger = get_logger()

    def _init_index(self, dim: int):
        if self._index is None:
            self.embedding_dim = dim
            self._index = self._faiss.IndexFlatL2(dim)

    def add(self, chunks: List[Chunk]) -> None:
        import numpy as np
        emb_chunks = [ch for ch in chunks if ch.embedding]
        if not emb_chunks:
            return
        dim = len(emb_chunks[0].embedding)
        self._init_index(dim)
        vectors = np.array([ch.embedding for ch in emb_chunks], dtype="float32")
        self._index.add(vectors)  # type: ignore
        self.chunks.extend(emb_chunks)
        self.logger.info(f"FAISS: {len(self.chunks)} total vectors")

    def search(self, query_embedding: List[float], k: int = 5) -> List[Chunk]:
        import numpy as np
        if self._index is None or not self.chunks:
            return []
        vec = np.array([query_embedding], dtype="float32")
        _, indices = self._index.search(vec, min(k, len(self.chunks)))  # type: ignore
        return [self.chunks[i] for i in indices[0] if 0 <= i < len(self.chunks)]

    def clear(self) -> None:
        self._index = None
        self.chunks = []

    def count(self) -> int:
        return len(self.chunks)


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class RAG:
    """Retrieval-Augmented Generation system.

    Combines document loading → splitting → embedding → retrieval → generation.

    Example::

        from neuralnode.rag import RAG, create_rag

        rag = create_rag(llm, store="memory")
        rag.add_documents(["report.pdf"])
        rag.add_text("Some extra context.")
        print(rag.query("What does the report say about revenue?"))
    """

    def __init__(
        self,
        llm: BaseProvider,
        embedding_provider: Optional[Any] = None,
        vector_store: Optional[BaseVectorStore] = None,
        text_splitter: Optional[TextSplitter] = None,
        document_loader: Optional[DocumentLoader] = None,
        top_k: int = 5,
    ):
        self.logger = get_logger()
        self.llm = llm
        self.embedding_provider = embedding_provider or self._resolve_embedding_provider(llm)
        self.vector_store = vector_store or InMemoryVectorStore()
        self.text_splitter = text_splitter or TextSplitter()
        self.document_loader = document_loader or DocumentLoader()
        self.top_k = top_k

    @log_execution()
    def add_documents(
        self,
        sources: List[Union[str, Path]],
        split: bool = True,
    ) -> None:
        """Load and index documents from files, directories, or URLs."""
        docs: List[Document] = []
        for src in sources:
            path = Path(src) if not str(src).startswith("http") else src
            if isinstance(path, Path) and path.is_dir():
                docs.extend(self.document_loader.load_directory(path))
            else:
                docs.extend(self.document_loader.load(src))

        chunks = self.text_splitter.split_documents(docs) if split else [
            Chunk(content=d.content, doc_id=d.doc_id, chunk_index=0, metadata=d.metadata)  # type: ignore
            for d in docs
        ]
        self._embed_and_store(chunks)
        self.logger.info(f"Added {len(docs)} docs → {len(chunks)} chunks")

    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        split: bool = True,
    ) -> None:
        """Index a raw text string directly (no file needed).

        Args:
            text: The text to add.
            metadata: Optional metadata dict.
            split: Whether to split into chunks.
        """
        doc = Document(content=text, metadata=metadata or {"source": "inline"})
        chunks = self.text_splitter.split_documents([doc]) if split else [
            Chunk(content=text, doc_id=doc.doc_id, chunk_index=0, metadata=doc.metadata)  # type: ignore
        ]
        self._embed_and_store(chunks)

    def search(self, query: str, k: Optional[int] = None) -> List[Chunk]:
        """Retrieve relevant chunks without calling the LLM.

        Args:
            query: Search query.
            k: Number of results (defaults to ``self.top_k``).

        Returns:
            List of matching Chunk objects with ``.content`` and ``.metadata``.
        """
        embedding = self._embed_query(query)
        return self.vector_store.search(embedding, k=k or self.top_k)

    @log_execution()
    def query(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        include_sources: bool = False,
    ) -> Union[str, Iterator[str], Dict[str, Any]]:
        """Answer a question using retrieved context + the LLM.

        Args:
            question: The question to answer.
            system_prompt: Optional custom system prompt.
            stream: If True, returns an iterator of string chunks.
            include_sources: If True, returns dict with answer and sources.

        Returns:
            Answer string, iterator, or dict with answer and sources.
        """
        chunks = self.search(question)
        if not chunks:
            no_ctx = "I don't have enough information in the provided documents to answer that."
            if include_sources:
                return {"answer": no_ctx, "sources": []}
            return iter([no_ctx]) if stream else no_ctx

        context = self._build_context(chunks)
        system = system_prompt or "You are a helpful assistant. Answer based on the provided context only."
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        messages = [Message.system(system), Message.user(prompt)]
        response = self.llm.chat(messages, stream=stream)

        if stream:
            return (chunk.content for chunk in ensure_streaming_chunks(response))  # type: ignore
        
        answer = ensure_llm_response(response).content
        
        if include_sources:
            sources = [
                {
                    "content": ch.content,
                    "metadata": ch.metadata,
                    "doc_id": ch.doc_id,
                    "chunk_index": ch.chunk_index,
                }
                for ch in chunks
            ]
            return {"answer": answer, "sources": sources}
        
        return answer

    def count(self) -> int:
        """Return total number of indexed chunks."""
        return self.vector_store.count()

    def clear(self) -> None:
        """Remove all indexed documents."""
        self.vector_store.clear()
        self.logger.info("RAG cleared")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> List[float]:
        try:
            return self.embedding_provider.embed([query])[0]
        except Exception as exc:
            self.logger.warning("Query embedding provider failed (%s). Falling back to lexical embeddings.", exc)
            self.embedding_provider = _LexicalEmbeddingProvider()
            return self.embedding_provider.embed([query])[0]

    def _embed_and_store(self, chunks: List[Chunk]) -> None:
        texts = [ch.content for ch in chunks]
        try:
            embeddings = self.embedding_provider.embed(texts)
            for ch, emb in zip(chunks, embeddings):
                ch.embedding = emb
        except Exception as e:
            self.logger.warning("Embedding provider failed (%s). Switching to lexical fallback.", e)
            self.embedding_provider = _LexicalEmbeddingProvider()
            embeddings = self.embedding_provider.embed(texts)
            for ch, emb in zip(chunks, embeddings):
                ch.embedding = emb
        self.vector_store.add(chunks)

    def _resolve_embedding_provider(self, llm: BaseProvider) -> Any:
        """Prefer provider embeddings, then sentence-transformers, then lexical fallback."""
        try:
            if hasattr(llm, "supports_embeddings") and llm.supports_embeddings():
                return llm
        except Exception:
            pass

        try:
            from ..providers import SentenceTransformersProvider

            provider = SentenceTransformersProvider(model="sentence-transformers/all-MiniLM-L6-v2")
            self.logger.info("RAG is using SentenceTransformersProvider for embeddings.")
            return provider
        except Exception as exc:
            self.logger.warning(
                "Sentence-transformers embeddings unavailable (%s). Falling back to lexical embeddings.",
                exc,
            )

        return _LexicalEmbeddingProvider()

    @staticmethod
    def _build_context(chunks: List[Chunk]) -> str:
        parts = []
        for i, ch in enumerate(chunks, 1):
            src = ch.metadata.get("source", "Unknown")
            parts.append(f"[Document {i} — {src}]\n{ch.content}")
        return "\n\n".join(parts)


# ── RAG Pipeline alias ─────────────────────────────────────────────────────────
RAGPipeline = RAG


def create_rag(
    llm: BaseProvider,
    embedding_provider: Optional[Any] = None,
    store: str = "memory",
    persist_dir: Optional[str] = None,
    collection_name: str = "neuralnode",
    **kwargs,
) -> RAG:
    """Create a RAG system with the specified vector store.

    Args:
        llm: Any NeuralNode LLM provider.
        store: ``"memory"``, ``"chroma"``, or ``"faiss"``.
        persist_dir: Path to persist ChromaDB on disk (only for chroma).
        collection_name: ChromaDB collection name.
        **kwargs: Passed to the RAG constructor.

    Returns:
        Configured RAG instance.
    """
    if store == "memory":
        vector_store: BaseVectorStore = InMemoryVectorStore()
    elif store == "chroma":
        vector_store = ChromaDBStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
    elif store == "faiss":
        vector_store = FAISSVectorStore()
    else:
        raise ValueError(f"Unknown store type: {store!r}. Use 'memory', 'chroma', or 'faiss'.")
    return RAG(llm=llm, embedding_provider=embedding_provider, vector_store=vector_store, **kwargs)


class _LexicalEmbeddingProvider:
    """Dependency-free fallback embeddings for local-only RAG usage."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return [self._embed_text(text) for text in texts]

    def _embed_text(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        counts = Counter(tokens)
        vector = [0.0] * self.dimensions

        for token, weight in counts.items():
            # MD5 is used only as a deterministic bucket hash, never for
            # authentication or integrity.  Mark that intent explicitly.
            index = int(hashlib.md5(token.encode("utf-8"), usedforsecurity=False).hexdigest(), 16) % self.dimensions
            vector[index] += float(weight)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


# ── Exports ────────────────────────────────────────────────────────────────────

__all__ = [
    # Data
    "Document", "Chunk",
    # Splitters
    "TextSplitter", "BaseTextSplitter", "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter", "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    # Loaders
    "BaseDocumentLoader", "TextLoader", "PDFLoader", "DocxLoader",
    "WebLoader", "DocumentLoader",
    "JSONLoader", "CSVLoader", "DirectoryLoader",
    "load_documents", "split_documents",
    # Vector Stores
    "BaseVectorStore", "InMemoryVectorStore", "ChromaDBStore", "FAISSVectorStore",
    # RAG
    "RAG", "RAGPipeline", "create_rag",
]
