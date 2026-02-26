"""Hardware detection for local model optimization."""

from typing import Optional, Dict, Any
import platform
import psutil


class HardwareDetector:
    """Detect system hardware capabilities."""

    def __init__(self):
        self._cuda_available: Optional[bool] = None
        self._vram_gb: Optional[float] = None
        self._ram_gb: Optional[float] = None
        self._cpu_count: Optional[int] = None
        self._gpu_info: Optional[Dict[str, Any]] = None
        self._detect()

    def _detect(self) -> None:
        """Detect hardware capabilities."""
        # System RAM
        mem = psutil.virtual_memory()
        self._ram_gb = mem.total / (1024 ** 3)
        
        # CPU
        self._cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count()
        
        # Try to detect GPU
        self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect GPU capabilities."""
        # Try PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_available = True
                device_count = torch.cuda.device_count()
                total_vram = 0
                gpu_info_list = []
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    vram = props.total_memory / (1024 ** 3)
                    total_vram += vram
                    gpu_info_list.append({
                        "name": props.name,
                        "vram_gb": vram,
                        "compute_capability": f"{props.major}.{props.minor}",
                    })
                
                self._vram_gb = total_vram
                self._gpu_info = {
                    "count": device_count,
                    "devices": gpu_info_list,
                }
                return
        except ImportError:
            pass
        
        # Try NVIDIA Management Library
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self._cuda_available = True
            total_vram = 0
            gpu_info_list = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                vram = info.total / (1024 ** 3)
                total_vram += vram
                gpu_info_list.append({
                    "name": name,
                    "vram_gb": vram,
                })
            
            self._vram_gb = total_vram
            self._gpu_info = {
                "count": device_count,
                "devices": gpu_info_list,
            }
            return
        except (ImportError, Exception):
            pass
        
        # No GPU detected
        self._cuda_available = False
        self._vram_gb = 0
        self._gpu_info = {"count": 0, "devices": []}

    @property
    def has_cuda(self) -> bool:
        """Check if CUDA is available."""
        return self._cuda_available or False

    @property
    def vram_gb(self) -> Optional[float]:
        """Get total VRAM in GB."""
        return self._vram_gb

    @property
    def ram_gb(self) -> Optional[float]:
        """Get total RAM in GB."""
        return self._ram_gb

    @property
    def cpu_count(self) -> Optional[int]:
        """Get number of CPU cores."""
        return self._cpu_count

    @property
    def gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information."""
        return self._gpu_info

    def recommend_model_config(self) -> Dict[str, Any]:
        """
        Recommend model configuration based on hardware.
        
        Returns:
            Dictionary with recommended settings
        """
        config = {
            "device": "cpu",
            "quantization": "Q4_K_M",
            "batch_size": 1,
            "context_length": 4096,
            "n_gpu_layers": 0,
        }
        
        if self.has_cuda and self._vram_gb:
            config["device"] = "cuda"
            
            if self._vram_gb >= 24:
                config["quantization"] = "Q8_0"
                config["batch_size"] = 4
                config["n_gpu_layers"] = 1000
                config["context_length"] = 8192
            elif self._vram_gb >= 16:
                config["quantization"] = "Q6_K"
                config["batch_size"] = 2
                config["n_gpu_layers"] = 1000
                config["context_length"] = 8192
            elif self._vram_gb >= 8:
                config["quantization"] = "Q4_K_M"
                config["batch_size"] = 1
                config["n_gpu_layers"] = 35
                config["context_length"] = 4096
            else:
                config["n_gpu_layers"] = 20
        
        # Adjust based on RAM
        if self._ram_gb:
            if self._ram_gb < 8:
                config["context_length"] = 2048
            elif self._ram_gb >= 64:
                config["batch_size"] = max(config["batch_size"], 2)
        
        return config

    def can_run_model(self, model_size: str) -> bool:
        """
        Check if this hardware can run a model of given size.
        
        Args:
            model_size: Model size like "7b", "13b", "70b"
            
        Returns:
            True if model can likely run
        """
        size_map = {
            "1b": 2,
            "3b": 6,
            "7b": 14,
            "8b": 16,
            "13b": 26,
            "70b": 140,
        }
        
        required_vram = size_map.get(model_size.lower(), 16)
        
        if self.has_cuda and self._vram_gb:
            return self._vram_gb >= required_vram * 0.6  # Can offload partially
        
        # Check RAM for CPU-only
        if self._ram_gb:
            return self._ram_gb >= required_vram
        
        return False

    def get_summary(self) -> str:
        """Get a human-readable hardware summary."""
        lines = [
            f"CPU: {self._cpu_count} cores",
            f"RAM: {self._ram_gb:.1f} GB" if self._ram_gb else "RAM: Unknown",
        ]
        
        if self.has_cuda:
            lines.append(f"GPU: {self._gpu_info['count']} device(s)")
            for i, gpu in enumerate(self._gpu_info.get("devices", [])):
                lines.append(f"  - GPU {i}: {gpu.get('name', 'Unknown')} ({gpu.get('vram_gb', 0):.1f} GB)")
        else:
            lines.append("GPU: None detected (CPU-only)")
        
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_count": self._cpu_count,
            "ram_gb": self._ram_gb,
            "has_cuda": self._cuda_available,
            "vram_gb": self._vram_gb,
            "gpu_info": self._gpu_info,
            "platform": platform.platform(),
        }
