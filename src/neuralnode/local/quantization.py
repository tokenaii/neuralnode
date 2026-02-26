"""Model quantization tools for local LLMs."""

import subprocess
import os
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "Q4_K_M"  # Q4_K_M, Q5_K_M, Q8_0, Q2_K, etc.
    output_format: str = "gguf"
    context_length: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers on GPU if available


class ModelQuantizer:
    """Automatic model quantization using llama.cpp."""
    
    QUANTIZATION_METHODS = {
        "Q2_K": {"bits": 2, "description": "Smallest, lowest quality"},
        "Q3_K_S": {"bits": 3, "description": "Small, low quality"},
        "Q3_K_M": {"bits": 3, "description": "Medium, balanced"},
        "Q3_K_L": {"bits": 3, "description": "Large, better quality"},
        "Q4_0": {"bits": 4, "description": "Legacy, balanced"},
        "Q4_K_S": {"bits": 4, "description": "Small, good quality"},
        "Q4_K_M": {"bits": 4, "description": "Recommended, best balance"},
        "Q5_0": {"bits": 5, "description": "Higher quality"},
        "Q5_K_S": {"bits": 5, "description": "Good quality"},
        "Q5_K_M": {"bits": 5, "description": "Best quality for size"},
        "Q6_K": {"bits": 6, "description": "Near FP16 quality"},
        "Q8_0": {"bits": 8, "description": "Almost lossless"},
        "F16": {"bits": 16, "description": "Half precision"},
    }
    
    def __init__(self, llama_cpp_path: Optional[str] = None):
        self.llama_cpp_path = llama_cpp_path or self._find_llama_cpp()
        self.models_dir = Path.home() / ".neuralnode" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def _find_llama_cpp(self) -> Optional[str]:
        """Find llama.cpp installation."""
        # Check common locations
        paths = [
            "llama.cpp",
            "llama-cpp",
            "convert.py",
            "/usr/local/bin/llama-quantize",
            str(Path.home() / "llama.cpp" / "convert.py"),
        ]
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def recommend_quantization(self, model_size_gb: float, vram_gb: float) -> str:
        """Recommend quantization based on hardware."""
        available_vram = vram_gb * 0.9  # Leave some headroom
        
        # Calculate bits needed
        for method in ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]:
            bits = self.QUANTIZATION_METHODS[method]["bits"]
            estimated_size = model_size_gb * (bits / 16)
            
            if estimated_size < available_vram:
                return method
        
        # Fallback to smallest
        return "Q4_K_M"
    
    def download_model(
        self,
        model_id: str,  # e.g., "meta-llama/Llama-2-7b" or HuggingFace model
        cache_dir: Optional[str] = None,
    ) -> str:
        """Download model from HuggingFace."""
        try:
            from huggingface_hub import snapshot_download
            
            cache_dir = cache_dir or str(self.models_dir / "hf_cache")
            
            print(f"Downloading {model_id}...")
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                local_dir=str(self.models_dir / model_id.replace("/", "--")),
                local_dir_use_symlinks=False,
            )
            
            return local_path
        except ImportError:
            print("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return ""
    
    def convert_to_gguf(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        config: Optional[QuantizationConfig] = None,
    ) -> str:
        """Convert HuggingFace model to GGUF format."""
        config = config or QuantizationConfig()
        
        model_path = Path(model_path)
        
        if output_path is None:
            model_name = model_path.name
            output_path = self.models_dir / f"{model_name}.gguf"
        else:
            output_path = Path(output_path)
        
        # Find convert script
        convert_script = self._find_convert_script()
        if not convert_script:
            print("llama.cpp convert script not found.")
            print("Please install llama.cpp or provide the path.")
            return ""
        
        # Run conversion
        cmd = [
            "python", convert_script,
            "--outfile", str(output_path),
            "--outtype", "f16",  # First convert to FP16
            str(model_path),
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"Conversion failed: {e}")
            return ""
    
    def quantize(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        method: str = "Q4_K_M",
    ) -> str:
        """Quantize a GGUF model."""
        input_path = Path(input_path)
        
        if output_path is None:
            model_name = input_path.stem
            output_path = self.models_dir / f"{model_name}-{method}.gguf"
        else:
            output_path = Path(output_path)
        
        # Find quantize binary
        quantize_bin = self._find_quantize_binary()
        if not quantize_bin:
            print("llama-quantize not found.")
            return ""
        
        cmd = [
            quantize_bin,
            str(input_path),
            str(output_path),
            method,
        ]
        
        try:
            print(f"Quantizing to {method}...")
            subprocess.run(cmd, check=True)
            
            # Get file sizes
            original_size = input_path.stat().st_size / (1024**3)  # GB
            quantized_size = output_path.stat().st_size / (1024**3)  # GB
            
            print(f"Original: {original_size:.2f} GB")
            print(f"Quantized: {quantized_size:.2f} GB")
            print(f"Compression: {(1 - quantized_size/original_size)*100:.1f}%")
            
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"Quantization failed: {e}")
            return ""
    
    def _find_convert_script(self) -> Optional[str]:
        """Find llama.cpp convert script."""
        if self.llama_cpp_path:
            convert = Path(self.llama_cpp_path) / "convert.py"
            if convert.exists():
                return str(convert)
            
            convert = Path(self.llama_cpp_path) / "convert-hf-to-gguf.py"
            if convert.exists():
                return str(convert)
        
        # Check common locations
        for path in [
            "convert.py",
            "convert-hf-to-gguf.py",
            "/usr/local/bin/convert-hf-to-gguf.py",
        ]:
            if os.path.exists(path):
                return path
        
        return None
    
    def _find_quantize_binary(self) -> Optional[str]:
        """Find llama-quantize binary."""
        if self.llama_cpp_path:
            quantize = Path(self.llama_cpp_path) / "build" / "bin" / "llama-quantize"
            if quantize.exists():
                return str(quantize)
            
            quantize = Path(self.llama_cpp_path) / "llama-quantize"
            if quantize.exists():
                return str(quantize)
        
        # Check in PATH
        for path in ["llama-quantize", "/usr/local/bin/llama-quantize"]:
            result = subprocess.run(["which", path], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        
        return None
    
    def auto_quantize(
        self,
        model_id: str,
        vram_gb: Optional[float] = None,
    ) -> str:
        """Full pipeline: download, convert, quantize with optimal settings."""
        # Get VRAM if not provided
        if vram_gb is None:
            from ..local.hardware import HardwareDetector
            hw = HardwareDetector()
            vram_gb = hw.vram_gb
        
        # Download
        model_path = self.download_model(model_id)
        if not model_path:
            return ""
        
        # Estimate model size
        model_size = self._estimate_model_size(model_path)
        
        # Recommend quantization
        method = self.recommend_quantization(model_size, vram_gb)
        print(f"Recommended quantization: {method}")
        
        # Convert to GGUF
        gguf_path = self.convert_to_gguf(model_path)
        if not gguf_path:
            return ""
        
        # Quantize
        quantized_path = self.quantize(gguf_path, method=method)
        
        return quantized_path
    
    def _estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB from files."""
        path = Path(model_path)
        total_size = 0
        
        for pattern in ["*.bin", "*.safetensors", "pytorch_model.bin"]:
            for file in path.rglob(pattern):
                total_size += file.stat().st_size
        
        return total_size / (1024**3)  # Convert to GB
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """List popular models available for download."""
        return [
            {
                "id": "meta-llama/Llama-2-7b-chat-hf",
                "name": "Llama 2 7B Chat",
                "size_gb": 13.5,
                "type": "chat",
                "license": "LLAMA 2",
            },
            {
                "id": "meta-llama/Llama-2-13b-chat-hf",
                "name": "Llama 2 13B Chat",
                "size_gb": 26.0,
                "type": "chat",
                "license": "LLAMA 2",
            },
            {
                "id": "mistralai/Mistral-7B-Instruct-v0.2",
                "name": "Mistral 7B Instruct",
                "size_gb": 14.0,
                "type": "instruct",
                "license": "Apache 2.0",
            },
            {
                "id": "microsoft/Phi-3-mini-4k-instruct",
                "name": "Phi-3 Mini",
                "size_gb": 7.6,
                "type": "instruct",
                "license": "MIT",
            },
            {
                "id": "Qwen/Qwen2-7B-Instruct",
                "name": "Qwen2 7B",
                "size_gb": 15.0,
                "type": "instruct",
                "license": "Qwen License",
            },
            {
                "id": "google/gemma-2-9b-it",
                "name": "Gemma 2 9B",
                "size_gb": 18.0,
                "type": "instruct",
                "license": "Gemma License",
            },
        ]


class ModelDownloader:
    """Download models from various sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".neuralnode" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_from_huggingface(
        self,
        model_id: str,
        filename: Optional[str] = None,
    ) -> str:
        """Download model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            
            if filename:
                # Download specific file
                path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    cache_dir=str(self.cache_dir / "hf_cache"),
                    local_dir=str(self.cache_dir),
                )
                return path
            else:
                # Download entire repo
                path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir / "hf_cache"),
                    local_dir=str(self.cache_dir / model_id.replace("/", "--")),
                )
                return path
        except ImportError:
            print("Install huggingface_hub: pip install huggingface_hub")
            return ""
    
    def download_gguf(
        self,
        url: str,
        filename: Optional[str] = None,
    ) -> str:
        """Download GGUF model from URL."""
        import urllib.request
        
        if filename is None:
            filename = url.split("/")[-1]
        
        output_path = self.cache_dir / filename
        
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to: {output_path}")
        
        return str(output_path)
    
    def list_local_models(self) -> List[Dict[str, Any]]:
        """List downloaded models."""
        models = []
        
        for file in self.cache_dir.rglob("*.gguf"):
            size_gb = file.stat().st_size / (1024**3)
            models.append({
                "name": file.stem,
                "path": str(file),
                "size_gb": round(size_gb, 2),
                "format": "GGUF",
            })
        
        for file in self.cache_dir.rglob("*.bin"):
            size_gb = file.stat().st_size / (1024**3)
            models.append({
                "name": file.stem,
                "path": str(file),
                "size_gb": round(size_gb, 2),
                "format": "BIN",
            })
        
        return models


# Convenience functions
def quantize_model(
    model_path: str,
    output_path: Optional[str] = None,
    method: str = "Q4_K_M",
) -> str:
    """Quick quantize function."""
    quantizer = ModelQuantizer()
    return quantizer.quantize(model_path, output_path, method)


def download_and_quantize(model_id: str, method: str = "Q4_K_M") -> str:
    """Download and quantize in one step."""
    quantizer = ModelQuantizer()
    return quantizer.auto_quantize(model_id)


def list_models() -> List[Dict[str, Any]]:
    """List available local models."""
    downloader = ModelDownloader()
    return downloader.list_local_models()
