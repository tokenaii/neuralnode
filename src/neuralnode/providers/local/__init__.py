"""Local model management for NeuralNode."""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from urllib.request import urlopen, Request
from urllib.error import URLError
import ssl


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    id: str
    name: str
    description: str
    download_url: str
    size_mb: int
    license: str
    quantization: str
    tags: List[str]


class ModelRegistry:
    """Registry of available local models."""
    
    # Built-in registry of recommended models
    DEFAULT_MODELS = {
        "llama-3.2-1b": ModelInfo(
            id="llama-3.2-1b",
            name="Llama 3.2 1B",
            description="Fast, lightweight model good for simple tasks",
            download_url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            size_mb=800,
            license="Llama 3.2 License",
            quantization="Q4_K_M",
            tags=["fast", "lightweight", "general"]
        ),
        "llama-3.2-3b": ModelInfo(
            id="llama-3.2-3b",
            name="Llama 3.2 3B",
            description="Balanced model with good performance",
            download_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            size_mb=2000,
            license="Llama 3.2 License",
            quantization="Q4_K_M",
            tags=["balanced", "general"]
        ),
        "qwen2.5-3b": ModelInfo(
            id="qwen2.5-3b",
            name="Qwen 2.5 3B",
            description="Strong multilingual model",
            download_url="https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            size_mb=2000,
            license="Apache 2.0",
            quantization="Q4_K_M",
            tags=["multilingual", "balanced"]
        ),
        "phi-4": ModelInfo(
            id="phi-4",
            name="Phi-4",
            description="Microsoft's high-quality small model",
            download_url="https://huggingface.co/bartowski/Phi-4-GGUF/resolve/main/Phi-4-Q4_K_M.gguf",
            size_mb=8000,
            license="MIT",
            quantization="Q4_K_M",
            tags=["high-quality", "coding"]
        ),
    }
    
    def __init__(self):
        self.models = dict(self.DEFAULT_MODELS)
    
    def list_models(
        self,
        tag: Optional[str] = None,
        max_size_mb: Optional[int] = None
    ) -> List[ModelInfo]:
        """List available models with optional filtering."""
        models = list(self.models.values())
        
        if tag:
            models = [m for m in models if tag in m.tags]
        
        if max_size_mb:
            models = [m for m in models if m.size_mb <= max_size_mb]
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self.models.get(model_id)
    
    def search(self, query: str) -> List[ModelInfo]:
        """Search models by name or description."""
        query_lower = query.lower()
        return [
            m for m in self.models.values()
            if query_lower in m.name.lower() or query_lower in m.description.lower()
        ]


class ModelDownloader:
    """Download and manage local models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.neuralnode/models"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def list_downloaded(self) -> List[Dict[str, Any]]:
        """List models already downloaded."""
        models = []
        
        for file_path in self.cache_dir.glob("*.gguf"):
            stat = file_path.stat()
            models.append({
                "path": str(file_path),
                "name": file_path.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime
            })
        
        return sorted(models, key=lambda x: x["name"])
    
    def is_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded."""
        model = self.registry.get_model(model_id)
        if not model:
            return False
        
        # Check for file by model name pattern
        expected_path = self._get_model_path(model_id)
        return expected_path.exists()
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get the expected path for a model."""
        model = self.registry.get_model(model_id)
        if model:
            # Use the filename from URL
            filename = model.download_url.split('/')[-1]
            return self.cache_dir / filename
        return self.cache_dir / f"{model_id}.gguf"
    
    def download(
        self,
        model_id: str,
        progress_callback: Optional[callable] = None
    ) -> str:
        """Download a model from the registry.
        
        Returns:
            Path to downloaded model file
        """
        model = self.registry.get_model(model_id)
        if not model:
            raise ValueError(f"Unknown model: {model_id}")
        
        output_path = self._get_model_path(model_id)
        
        # Check if already exists
        if output_path.exists():
            return str(output_path)
        
        # Download
        try:
            headers = {
                "User-Agent": "NeuralNode-Downloader/1.0"
            }
            
            req = Request(model.download_url, headers=headers)
            
            with urlopen(req, context=self.ssl_context, timeout=60) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size:
                            progress = (downloaded / total_size) * 100
                            progress_callback(progress, downloaded, total_size)
            
            return str(output_path)
            
        except URLError as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Download failed: {e}")
    
    def delete(self, model_id: str) -> bool:
        """Delete a downloaded model."""
        path = self._get_model_path(model_id)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def get_path(self, model_id: str) -> Optional[str]:
        """Get the path to a model if it exists."""
        path = self._get_model_path(model_id)
        return str(path) if path.exists() else None


class LocalModelManager:
    """High-level manager for local models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.downloader = ModelDownloader(cache_dir)
        self.registry = self.downloader.registry
    
    def list_available(self) -> List[ModelInfo]:
        """List models available for download."""
        return self.registry.list_models()
    
    def list_installed(self) -> List[Dict[str, Any]]:
        """List installed models."""
        return self.downloader.list_downloaded()
    
    def install(self, model_id: str, progress_callback: Optional[callable] = None) -> str:
        """Download and install a model."""
        return self.downloader.download(model_id, progress_callback)
    
    def uninstall(self, model_id: str) -> bool:
        """Remove an installed model."""
        return self.downloader.delete(model_id)
    
    def is_installed(self, model_id: str) -> bool:
        """Check if a model is installed."""
        return self.downloader.is_downloaded(model_id)
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """Get path to installed model."""
        return self.downloader.get_path(model_id)
    
    def recommend_model(
        self,
        ram_gb: Optional[float] = None,
        vram_gb: Optional[float] = None,
        priority: str = "balanced"  # speed, quality, balanced
    ) -> Optional[ModelInfo]:
        """Recommend a model based on hardware."""
        if ram_gb is None:
            # Try to detect
            try:
                import psutil
                ram_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                ram_gb = 8  # Assume 8GB
        
        # Filter by size
        max_size = min(ram_gb * 0.75 * 1024, 16000)  # Max 75% of RAM or 16GB
        
        candidates = self.registry.list_models(max_size_mb=int(max_size))
        
        if not candidates:
            return None
        
        # Score candidates
        def score_model(m: ModelInfo) -> float:
            s = 0.0
            
            # Size score (smaller is better for speed, larger for quality)
            if priority == "speed":
                s += (2000 - m.size_mb) / 2000 * 10
            elif priority == "quality":
                s += m.size_mb / 16000 * 10
            else:  # balanced
                s += 5 if 1500 < m.size_mb < 4000 else 2
            
            # Prefer certain tags
            if priority == "speed" and "fast" in m.tags:
                s += 3
            if priority == "quality" and "high-quality" in m.tags:
                s += 3
            
            return s
        
        return max(candidates, key=score_model)


def list_models(tag: Optional[str] = None) -> List[ModelInfo]:
    """Quick function to list available models."""
    registry = ModelRegistry()
    return registry.list_models(tag=tag)


def download_model(model_id: str, cache_dir: Optional[str] = None) -> str:
    """Quick function to download a model."""
    downloader = ModelDownloader(cache_dir)
    return downloader.download(model_id)
