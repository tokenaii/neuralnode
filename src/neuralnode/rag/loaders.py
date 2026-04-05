"""
Comprehensive RAG/Retrieval System for NeuralNode
Document Loaders, Text Splitters, and RAG Pipeline
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Iterator, Callable, Type
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with content and metadata."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __repr__(self):
        return f"Document(id={self.id}, content={self.page_content[:50]}..., metadata={self.metadata})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_content": self.page_content,
            "metadata": self.metadata,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        return cls(
            page_content=data["page_content"],
            metadata=data.get("metadata", {}),
            id=data.get("id")
        )


# ==================== Document Loaders ====================

class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from source."""
        pass
    
    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents."""
        pass


class TextLoader(BaseDocumentLoader):
    """Load plain text files."""
    
    def __init__(self, file_path: Union[str, Path], encoding: str = "utf-8"):
        self.file_path = Path(file_path)
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """Load text file."""
        try:
            text = self.file_path.read_text(encoding=self.encoding)
            return [Document(
                page_content=text,
                metadata={"source": str(self.file_path)}
            )]
        except Exception as e:
            logger.error(f"Failed to load {self.file_path}: {e}")
            return []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load."""
        for doc in self.load():
            yield doc


class JSONLoader(BaseDocumentLoader):
    """Load JSON files."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        jq_schema: Optional[str] = None,
        content_key: Optional[str] = None,
        encoding: str = "utf-8"
    ):
        self.file_path = Path(file_path)
        self.jq_schema = jq_schema
        self.content_key = content_key
        self.encoding = encoding
    
    def load(self) -> List[Document]:
        """Load JSON file."""
        try:
            text = self.file_path.read_text(encoding=self.encoding)
            data = json.loads(text)
            
            documents = []
            
            # If content_key specified, extract that field
            if self.content_key:
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        content = item.get(self.content_key, "") if isinstance(item, dict) else str(item)
                        documents.append(Document(
                            page_content=str(content),
                            metadata={"source": str(self.file_path), "index": i}
                        ))
                elif isinstance(data, dict):
                    content = data.get(self.content_key, "")
                    documents.append(Document(
                        page_content=str(content),
                        metadata={"source": str(self.file_path)}
                    ))
            else:
                # Store entire JSON as string
                documents.append(Document(
                    page_content=json.dumps(data, indent=2),
                    metadata={"source": str(self.file_path)}
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load JSON {self.file_path}: {e}")
            return []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load."""
        for doc in self.load():
            yield doc


class CSVLoader(BaseDocumentLoader):
    """Load CSV files."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        source_column: Optional[str] = None,
        encoding: str = "utf-8",
        delimiter: str = ","
    ):
        self.file_path = Path(file_path)
        self.source_column = source_column
        self.encoding = encoding
        self.delimiter = delimiter
    
    def load(self) -> List[Document]:
        """Load CSV file."""
        try:
            import csv
            
            documents = []
            with open(self.file_path, 'r', encoding=self.encoding, newline='') as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                for i, row in enumerate(reader):
                    # Convert row to text
                    content = "\n".join([f"{k}: {v}" for k, v in row.items()])
                    
                    source = row.get(self.source_column, str(self.file_path)) if self.source_column else str(self.file_path)
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": source,
                            "row": i,
                            **row
                        }
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load CSV {self.file_path}: {e}")
            return []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load."""
        for doc in self.load():
            yield doc


class PDFLoader(BaseDocumentLoader):
    """Load PDF files."""
    
    def __init__(self, file_path: Union[str, Path], extract_images: bool = False):
        self.file_path = Path(file_path)
        self.extract_images = extract_images
    
    def load(self) -> List[Document]:
        """Load PDF file."""
        try:
            import pypdf
            
            documents = []
            with open(self.file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": str(self.file_path),
                            "page": i + 1,
                            "total_pages": len(reader.pages)
                        }
                    ))
            
            return documents
            
        except ImportError:
            logger.error("pypdf required for PDF loading. Install: pip install pypdf")
            return []
        except Exception as e:
            logger.error(f"Failed to load PDF {self.file_path}: {e}")
            return []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load."""
        for doc in self.load():
            yield doc


class DirectoryLoader(BaseDocumentLoader):
    """Load all files from a directory."""
    
    def __init__(
        self,
        path: Union[str, Path],
        glob: str = "**/*",
        exclude: Optional[List[str]] = None,
        loader_cls: Optional[Type[BaseDocumentLoader]] = None
    ):
        self.path = Path(path)
        self.glob = glob
        self.exclude = exclude or []
        self.loader_cls = loader_cls
    
    def load(self) -> List[Document]:
        """Load all matching files."""
        documents = []
        
        for file_path in self.path.glob(self.glob):
            # Check exclusions
            if any(exclude in str(file_path) for exclude in self.exclude):
                continue
            
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Get appropriate loader
            loader = self._get_loader(file_path)
            if loader:
                try:
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
        
        return documents
    
    def _get_loader(self, file_path: Path) -> Optional[BaseDocumentLoader]:
        """Get appropriate loader for file type."""
        if self.loader_cls:
            return self.loader_cls(file_path)
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return TextLoader(file_path)
        elif suffix == '.json':
            return JSONLoader(file_path)
        elif suffix == '.csv':
            return CSVLoader(file_path)
        elif suffix == '.pdf':
            return PDFLoader(file_path)
        else:
            return TextLoader(file_path)
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load."""
        for file_path in self.path.glob(self.glob):
            if any(exclude in str(file_path) for exclude in self.exclude):
                continue
            if file_path.is_dir():
                continue
            
            loader = self._get_loader(file_path)
            if loader:
                for doc in loader.lazy_load():
                    yield doc


# ==================== Text Splitters ====================

class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks)
                    }
                ))
        
        return chunks


class CharacterTextSplitter(BaseTextSplitter):
    """Split text by character count."""
    
    def __init__(self, separator: str = "\n\n", **kwargs):
        super().__init__(**kwargs)
        self.separator = separator
    
    def split_text(self, text: str) -> List[str]:
        """Split text by character count."""
        # First split by separator
        if self.separator:
            initial_splits = text.split(self.separator)
        else:
            initial_splits = [text]
        
        # Then merge to chunk_size
        chunks = []
        current_chunk = ""
        
        for split in initial_splits:
            if self.length_function(current_chunk) + self.length_function(split) <= self.chunk_size:
                current_chunk += (self.separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            overlapped_chunks = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    overlap = chunks[i-1][-self.chunk_overlap:]
                    chunk = overlap + chunk
                overlapped_chunks.append(chunk)
            chunks = overlapped_chunks
        
        return chunks


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """Recursively split text by multiple separators."""
    
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        is_separator_regex: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.is_separator_regex = is_separator_regex
    
    def split_text(self, text: str) -> List[str]:
        """Recursively split text."""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split by separators."""
        if not text:
            return []
        
        if not separators:
            return [text] if self.length_function(text) <= self.chunk_size else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        if self.is_separator_regex:
            import re
            splits = re.split(separator, text)
        else:
            splits = text.split(separator)
        
        # Merge splits to chunk_size
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if self.length_function(current_chunk) + self.length_function(split) <= self.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    if self.length_function(current_chunk) > self.chunk_size:
                        # Need to split further
                        chunks.extend(self._split_text_recursive(current_chunk, remaining_separators))
                    else:
                        chunks.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            if self.length_function(current_chunk) > self.chunk_size:
                chunks.extend(self._split_text_recursive(current_chunk, remaining_separators))
            else:
                chunks.append(current_chunk)
        
        return chunks


class TokenTextSplitter(BaseTextSplitter):
    """Split text by token count (requires tiktoken)."""
    
    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.encoding_name = encoding_name
        self.model_name = model_name
        self._encoding = None
    
    def _get_encoding(self):
        """Get tiktoken encoding."""
        if self._encoding is None:
            try:
                import tiktoken
                if self.model_name:
                    self._encoding = tiktoken.encoding_for_model(self.model_name)
                else:
                    self._encoding = tiktoken.get_encoding(self.encoding_name)
            except ImportError:
                logger.error("tiktoken required. Install: pip install tiktoken")
                # Fallback to character count
                self._encoding = None
        return self._encoding
    
    def _length_function(self, text: str) -> int:
        """Count tokens."""
        encoding = self._get_encoding()
        if encoding:
            return len(encoding.encode(text))
        return len(text)
    
    def split_text(self, text: str) -> List[str]:
        """Split by token count."""
        encoding = self._get_encoding()
        
        if not encoding:
            # Fallback to character splitting
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
        
        tokens = encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - self.chunk_overlap
        
        return chunks


class MarkdownHeaderTextSplitter(BaseTextSplitter):
    """Split Markdown by headers."""
    
    def __init__(self, headers_to_split_on: List[tuple] = None, **kwargs):
        super().__init__(**kwargs)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split Markdown by headers."""
        import re
        
        chunks = []
        current_chunk = ""
        current_headers = {}
        
        lines = text.split("\n")
        
        for line in lines:
            # Check if line is a header
            header_match = None
            for header_char, header_name in self.headers_to_split_on:
                if line.startswith(header_char + " "):
                    header_match = (header_char, header_name)
                    break
            
            if header_match:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=current_headers.copy()
                    ))
                
                # Start new chunk
                header_char, header_name = header_match
                header_level = len(header_char)
                
                # Update headers
                current_headers[header_name] = line[len(header_char)+1:].strip()
                # Remove lower level headers
                for hc, hn in self.headers_to_split_on:
                    if len(hc) > header_level and hn in current_headers:
                        del current_headers[hn]
                
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=current_headers
            ))
        
        # Return as strings or Documents based on context
        return [c.page_content for c in chunks]


# ==================== RAG Pipeline ====================

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(
        self,
        llm: Any,
        retriever: Any,
        prompt_template: Optional[str] = None,
        return_source_documents: bool = False,
    ):
        self.llm = llm
        self.retriever = retriever
        self.return_source_documents = return_source_documents
        
        self.prompt_template = prompt_template or """Use the following context to answer the question. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """Query with RAG."""
        # Retrieve documents
        if hasattr(self.retriever, "search"):
            docs = self.retriever.search(question, k=kwargs.get("k", 4))
        elif hasattr(self.retriever, "similarity_search"):
            docs = self.retriever.similarity_search(question, k=kwargs.get("k", 4))
        else:
            docs = []
        
        # Build context
        context_parts = []
        for i, doc in enumerate(docs):
            if isinstance(doc, Document):
                content = doc.page_content
            elif isinstance(doc, dict):
                content = doc.get("page_content", doc.get("content", ""))
            else:
                content = str(doc)
            context_parts.append(f"[Document {i+1}]\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self.prompt_template.format(context=context, question=question)
        
        # Generate answer
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.chat(messages)
        
        if isinstance(response, dict):
            answer = response.get("content", "")
        else:
            answer = str(response)
        
        result = {"answer": answer}
        if self.return_source_documents:
            result["source_documents"] = docs
        
        return result
    
    async def aquery(self, question: str, **kwargs) -> Dict[str, Any]:
        """Async query."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, **kwargs))


# Convenience functions
def load_documents(source: Union[str, Path], **kwargs) -> List[Document]:
    """Load documents from file or directory."""
    path = Path(source)
    
    if path.is_dir():
        loader = DirectoryLoader(path, **kwargs)
    elif path.suffix == '.pdf':
        loader = PDFLoader(path, **kwargs)
    elif path.suffix == '.csv':
        loader = CSVLoader(path, **kwargs)
    elif path.suffix == '.json':
        loader = JSONLoader(path, **kwargs)
    else:
        loader = TextLoader(path, **kwargs)
    
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs
) -> List[Document]:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )
    return splitter.split_documents(documents)


__all__ = [
    # Documents
    "Document",
    
    # Loaders
    "BaseDocumentLoader",
    "TextLoader",
    "JSONLoader",
    "CSVLoader",
    "PDFLoader",
    "DirectoryLoader",
    
    # Splitters
    "BaseTextSplitter",
    "CharacterTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    
    # RAG
    "RAGPipeline",
    
    # Convenience functions
    "load_documents",
    "split_documents",
]
