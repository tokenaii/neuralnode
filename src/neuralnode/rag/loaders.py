"""Extended document loaders for various file formats."""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import csv
import io

from ..rag.vectorstore import Document


class CSVLoader:
    """Load CSV files as documents."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        source_column: Optional[str] = None,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        csv_args: Optional[Dict[str, Any]] = None,
    ):
        self.file_path = Path(file_path)
        self.source_column = source_column
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.csv_args = csv_args or {}
    
    def load(self) -> List[Document]:
        """Load CSV file as documents."""
        documents = []
        
        with open(self.file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, **self.csv_args)
            
            for i, row in enumerate(reader):
                # Determine content
                if self.content_columns:
                    content_parts = []
                    for col in self.content_columns:
                        if col in row:
                            content_parts.append(f"{col}: {row[col]}")
                    content = "\n".join(content_parts)
                else:
                    # Use all columns as content
                    content = "\n".join(f"{k}: {v}" for k, v in row.items())
                
                # Determine source
                if self.source_column and self.source_column in row:
                    source = row[self.source_column]
                else:
                    source = str(self.file_path)
                
                # Build metadata
                metadata = {
                    "source": source,
                    "row": i,
                    "file_type": "csv",
                }
                
                # Add specified metadata columns
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
                
                doc = Document(
                    content=content,
                    metadata=metadata,
                    id=f"{self.file_path.stem}_row_{i}",
                )
                documents.append(doc)
        
        return documents


class MarkdownLoader:
    """Load Markdown files with header-based splitting."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        split_headers: bool = True,
    ):
        self.file_path = Path(file_path)
        self.split_headers = split_headers
    
    def load(self) -> List[Document]:
        """Load markdown file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not self.split_headers:
            return [Document(
                content=content,
                metadata={"source": str(self.file_path), "type": "markdown"},
                id=self.file_path.stem,
            )]
        
        # Split by headers
        import re
        
        # Pattern to match markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        documents = []
        sections = re.split(f'(?=^{header_pattern})', content, flags=re.MULTILINE)
        
        current_section = ""
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Check if section starts with header
            match = re.match(header_pattern, section, re.MULTILINE)
            if match:
                level = len(match.group(1))
                title = match.group(2)
                
                doc = Document(
                    content=section.strip(),
                    metadata={
                        "source": str(self.file_path),
                        "type": "markdown",
                        "header_level": level,
                        "title": title,
                        "section": i,
                    },
                    id=f"{self.file_path.stem}_section_{i}",
                )
                documents.append(doc)
            else:
                # Content before first header
                current_section = section
        
        # Add any remaining content
        if current_section.strip() and not documents:
            documents.append(Document(
                content=current_section.strip(),
                metadata={"source": str(self.file_path), "type": "markdown"},
                id=f"{self.file_path.stem}_intro",
            ))
        
        return documents


class HTMLLoader:
    """Load HTML files."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        extract_links: bool = False,
    ):
        self.file_path = Path(file_path)
        self.extract_links = extract_links
    
    def load(self) -> List[Document]:
        """Load HTML file."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML loading. "
                "Install with: pip install beautifulsoup4"
            )
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            html = f.read()
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        
        metadata = {
            "source": str(self.file_path),
            "type": "html",
            "title": soup.title.string if soup.title else None,
        }
        
        if self.extract_links:
            links = [a.get("href") for a in soup.find_all("a") if a.get("href")]
            metadata["links"] = links
        
        return [Document(
            content=text,
            metadata=metadata,
            id=self.file_path.stem,
        )]


class JSONLoader:
    """Load JSON files as documents."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        content_key: str = "content",
        metadata_keys: Optional[List[str]] = None,
        jq_schema: Optional[str] = None,
    ):
        self.file_path = Path(file_path)
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.jq_schema = jq_schema
    
    def load(self) -> List[Document]:
        """Load JSON file."""
        import json
        
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        # Handle list of items
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = self._extract_content(item)
                metadata = self._extract_metadata(item)
                metadata.update({"source": str(self.file_path), "index": i})
                
                documents.append(Document(
                    content=content,
                    metadata=metadata,
                    id=f"{self.file_path.stem}_{i}",
                ))
        else:
            content = self._extract_content(data)
            metadata = self._extract_metadata(data)
            metadata["source"] = str(self.file_path)
            
            documents.append(Document(
                content=content,
                metadata=metadata,
                id=self.file_path.stem,
            ))
        
        return documents
    
    def _extract_content(self, item: Dict[str, Any]) -> str:
        """Extract content from item."""
        if self.content_key in item:
            value = item[self.content_key]
            if isinstance(value, str):
                return value
            return json.dumps(value)
        return json.dumps(item)
    
    def _extract_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from item."""
        metadata = {}
        for key in self.metadata_keys:
            if key in item:
                metadata[key] = item[key]
        return metadata


class DirectoryLoader:
    """Load all documents from a directory."""
    
    def __init__(
        self,
        path: Union[str, Path],
        glob_pattern: str = "**/*",
        exclude: Optional[List[str]] = None,
        loader_mapping: Optional[Dict[str, Any]] = None,
    ):
        self.path = Path(path)
        self.glob_pattern = glob_pattern
        self.exclude = exclude or []
        self.loader_mapping = loader_mapping or {
            ".txt": None,  # Use default
            ".md": MarkdownLoader,
            ".markdown": MarkdownLoader,
            ".csv": CSVLoader,
            ".html": HTMLLoader,
            ".htm": HTMLLoader,
            ".json": JSONLoader,
        }
    
    def load(self) -> List[Document]:
        """Load all documents from directory."""
        documents = []
        
        for file_path in self.path.glob(self.glob_pattern):
            # Check exclusions
            if any(ex in str(file_path) for ex in self.exclude):
                continue
            
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Get appropriate loader
            ext = file_path.suffix.lower()
            loader_class = self.loader_mapping.get(ext)
            
            if loader_class:
                try:
                    loader = loader_class(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            elif ext in [".txt"]:
                # Default text loading
                try:
                    from ..rag.vectorstore import FileLoader
                    doc = FileLoader.load_text(file_path)
                    documents.append(doc)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        return documents


# Update the main FileLoader to include new loaders
class ExtendedFileLoader:
    """Extended file loader supporting multiple formats."""
    
    @staticmethod
    def load_csv(path: str, **kwargs) -> List[Document]:
        """Load CSV file."""
        return CSVLoader(path, **kwargs).load()
    
    @staticmethod
    def load_markdown(path: str, **kwargs) -> List[Document]:
        """Load Markdown file."""
        return MarkdownLoader(path, **kwargs).load()
    
    @staticmethod
    def load_html(path: str, **kwargs) -> List[Document]:
        """Load HTML file."""
        return HTMLLoader(path, **kwargs).load()
    
    @staticmethod
    def load_json(path: str, **kwargs) -> List[Document]:
        """Load JSON file."""
        return JSONLoader(path, **kwargs).load()
    
    @staticmethod
    def load_directory(path: str, **kwargs) -> List[Document]:
        """Load all files from directory."""
        return DirectoryLoader(path, **kwargs).load()
