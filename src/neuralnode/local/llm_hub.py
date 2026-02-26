"""Local LLM Hub for managing all local models."""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class ModelInfo:
    """Information about a local model."""
    name: str
    path: str
    size_gb: float
    format: str  # gguf, pytorch, onnx, etc.
    quantization: Optional[str] = None
    context_length: int = 4096
    parameters: str = "unknown"
    added_date: datetime = None
    last_used: datetime = None
    usage_count: int = 0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.added_date is None:
            self.added_date = datetime.now()
        if self.tags is None:
            self.tags = []


class LocalLLMHub:
    """Central hub for managing local LLM models."""
    
    def __init__(self, hub_dir: Optional[str] = None):
        self.hub_dir = Path(hub_dir) if hub_dir else Path.home() / ".neuralnode" / "hub"
        self.hub_dir.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelInfo] = {}
        self.index_file = self.hub_dir / "index.json"
        self._load_index()
    
    def _load_index(self) -> None:
        """Load model index from file."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                
                for name, info in data.items():
                    self.models[name] = ModelInfo(
                        name=info["name"],
                        path=info["path"],
                        size_gb=info["size_gb"],
                        format=info["format"],
                        quantization=info.get("quantization"),
                        context_length=info.get("context_length", 4096),
                        parameters=info.get("parameters", "unknown"),
                        added_date=datetime.fromisoformat(info["added_date"]),
                        tags=info.get("tags", []),
                    )
            except Exception as e:
                print(f"Error loading index: {e}")
    
    def _save_index(self) -> None:
        """Save model index to file."""
        data = {
            name: {
                "name": m.name,
                "path": m.path,
                "size_gb": m.size_gb,
                "format": m.format,
                "quantization": m.quantization,
                "context_length": m.context_length,
                "parameters": m.parameters,
                "added_date": m.added_date.isoformat(),
                "last_used": m.last_used.isoformat() if m.last_used else None,
                "usage_count": m.usage_count,
                "tags": m.tags,
            }
            for name, m in self.models.items()
        }
        
        with open(self.index_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[ModelInfo]:
        """Scan directory for model files."""
        found = []
        dir_path = Path(directory)
        
        if not dir_path.exists():
            return found
        
        pattern = "**/*" if recursive else "*"
        
        # Supported formats
        extensions = {
            ".gguf": "gguf",
            ".bin": "pytorch",
            ".pt": "pytorch",
            ".pth": "pytorch",
            ".safetensors": "safetensors",
            ".onnx": "onnx",
            ".ggml": "ggml",
        }
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extensions:
                    size_gb = file_path.stat().st_size / (1024**3)
                    
                    info = ModelInfo(
                        name=file_path.stem,
                        path=str(file_path),
                        size_gb=round(size_gb, 2),
                        format=extensions[ext],
                    )
                    
                    found.append(info)
                    self.models[info.name] = info
        
        self._save_index()
        return found
    
    def add_model(
        self,
        model_path: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelInfo:
        """Add a model to the hub."""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        name = name or path.stem
        size_gb = path.stat().st_size / (1024**3)
        
        # Detect format
        ext = path.suffix.lower()
        format_map = {
            ".gguf": "gguf",
            ".bin": "pytorch",
            ".pt": "pytorch",
            ".safetensors": "safetensors",
        }
        fmt = format_map.get(ext, "unknown")
        
        # Detect quantization from filename
        quant = None
        if "q4" in name.lower():
            quant = "Q4"
        elif "q5" in name.lower():
            quant = "Q5"
        elif "q8" in name.lower():
            quant = "Q8"
        
        info = ModelInfo(
            name=name,
            path=str(path.absolute()),
            size_gb=round(size_gb, 2),
            format=fmt,
            quantization=quant,
            tags=tags or [],
        )
        
        self.models[name] = info
        self._save_index()
        
        return info
    
    def remove_model(self, name: str, delete_file: bool = False) -> bool:
        """Remove a model from the hub."""
        if name not in self.models:
            return False
        
        info = self.models[name]
        
        if delete_file:
            try:
                Path(info.path).unlink()
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        del self.models[name]
        self._save_index()
        
        return True
    
    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info."""
        return self.models.get(name)
    
    def list_models(
        self,
        format_filter: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
    ) -> List[ModelInfo]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if format_filter:
            models = [m for m in models if m.format == format_filter]
        
        if tags_filter:
            models = [m for m in models if any(t in m.tags for t in tags_filter)]
        
        # Sort by last used (most recent first)
        models.sort(key=lambda m: m.last_used or datetime.min, reverse=True)
        
        return models
    
    def get_recommendation(self, vram_gb: float) -> List[ModelInfo]:
        """Get model recommendations based on available VRAM."""
        suitable = []
        
        for model in self.models.values():
            # Rough estimate: 1GB model size needs ~1.5GB VRAM
            required_vram = model.size_gb * 1.5
            
            if required_vram <= vram_gb:
                suitable.append(model)
        
        # Sort by size (largest first = most capable)
        suitable.sort(key=lambda m: m.size_gb, reverse=True)
        
        return suitable
    
    def mark_used(self, name: str) -> None:
        """Mark model as used."""
        if name in self.models:
            self.models[name].last_used = datetime.now()
            self.models[name].usage_count += 1
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get hub statistics."""
        total_size = sum(m.size_gb for m in self.models.values())
        formats = {}
        for m in self.models.values():
            formats[m.format] = formats.get(m.format, 0) + 1
        
        return {
            "total_models": len(self.models),
            "total_size_gb": round(total_size, 2),
            "formats": formats,
            "hub_directory": str(self.hub_dir),
        }
    
    def import_from_hf(
        self,
        model_id: str,
        quantization: Optional[str] = None,
    ) -> ModelInfo:
        """Import model from HuggingFace."""
        from ..local.quantization import ModelQuantizer
        
        quantizer = ModelQuantizer()
        
        print(f"Downloading {model_id}...")
        model_path = quantizer.download_model(model_id)
        
        if quantization and model_path:
            print(f"Quantizing to {quantization}...")
            model_path = quantizer.quantize(model_path, method=quantization)
        
        if model_path:
            return self.add_model(
                model_path,
                name=model_id.replace("/", "--"),
                tags=["huggingface"],
            )
        
        raise RuntimeError(f"Failed to import {model_id}")


class ModelLauncher:
    """Launch and serve local models."""
    
    def __init__(self, hub: LocalLLMHub):
        self.hub = hub
        self.active_servers: Dict[str, Any] = {}
    
    def launch(self, model_name: str, port: int = 8000) -> str:
        """Launch a model server."""
        model = self.hub.get_model(model_name)
        if not model:
            return f"Model {model_name} not found"
        
        # Would start actual server here
        # For now, just mark as active
        self.active_servers[model_name] = {
            "port": port,
            "started": datetime.now(),
        }
        
        self.hub.mark_used(model_name)
        
        return f"Model {model_name} running on port {port}"
    
    def stop(self, model_name: str) -> bool:
        """Stop a running model server."""
        if model_name in self.active_servers:
            del self.active_servers[model_name]
            return True
        return False
    
    def list_active(self) -> Dict[str, Any]:
        """List active model servers."""
        return self.active_servers.copy()


# Convenience functions
def create_hub() -> LocalLLMHub:
    """Quick hub creation."""
    return LocalLLMHub()


def scan_for_models(directory: str) -> List[ModelInfo]:
    """Quick model scan."""
    hub = LocalLLMHub()
    return hub.scan_directory(directory)


def get_model_recommendation(vram_gb: float) -> List[ModelInfo]:
    """Get model recommendations."""
    hub = LocalLLMHub()
    return hub.get_recommendation(vram_gb)
