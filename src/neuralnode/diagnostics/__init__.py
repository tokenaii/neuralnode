"""NeuralDoctor - Advanced Diagnostics, Error Handling & Resource Control System.

Features:
- Automatic GPU/CUDA detection
- Resource monitoring and control
- Auto-fix common problems
- Model compatibility checking
- System health diagnostics
"""

import os
import sys
import json
import subprocess
import platform
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import time


class DiagnosticLevel(Enum):
    """Diagnostic severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProblemType(Enum):
    """Types of problems NeuralDoctor can detect and fix."""

    CUDA_NOT_AVAILABLE = "cuda_not_available"
    OUT_OF_MEMORY = "out_of_memory"
    MODEL_LOAD_FAILED = "model_load_failed"
    DEPENDENCY_MISSING = "dependency_missing"
    API_KEY_MISSING = "api_key_missing"
    NETWORK_ERROR = "network_error"
    QUANTIZATION_ERROR = "quantization_error"
    TOKENIZER_ERROR = "tokenizer_error"
    DEVICE_MISMATCH = "device_mismatch"
    CONTEXT_LENGTH_ERROR = "context_length_error"


@dataclass
class DiagnosticReport:
    """Complete system diagnostic report."""

    timestamp: float
    os_info: Dict[str, Any]
    cpu_info: Dict[str, Any]
    gpu_info: List[Dict[str, Any]]
    cuda_info: Dict[str, Any]
    memory_info: Dict[str, Any]
    python_info: Dict[str, Any]
    dependencies: Dict[str, bool]
    issues: List[Dict[str, Any]]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "os": self.os_info,
            "cpu": self.cpu_info,
            "gpu": self.gpu_info,
            "cuda": self.cuda_info,
            "memory": self.memory_info,
            "python": self.python_info,
            "dependencies": self.dependencies,
            "issues": self.issues,
            "recommendations": self.recommendations,
        }


@dataclass
class ResourceUsage:
    """Current resource usage."""

    cpu_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_usage: List[Dict[str, float]]
    timestamp: float


class GPUManager:
    """GPU detection and management."""

    @staticmethod
    def detect_gpus() -> List[Dict[str, Any]]:
        """Detect all GPUs in the system."""
        gpus = []

        # Try nvidia-smi first
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,driver_version,cuda_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 8:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_total_mb": float(parts[2]),
                                "memory_free_mb": float(parts[3]),
                                "memory_used_mb": float(parts[4]),
                                "temperature_c": float(parts[5]),
                                "utilization_percent": float(parts[6]),
                                "driver_version": parts[7],
                                "cuda_version": parts[8] if len(parts) > 8 else "unknown",
                            }
                        )
        except:
            pass

        # Fallback to torch
        if not gpus:
            try:
                import torch

                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        props = torch.cuda.get_device_properties(i)
                        gpus.append(
                            {
                                "index": i,
                                "name": torch.cuda.get_device_name(i),
                                "memory_total_mb": props.total_memory / (1024 * 1024),
                                "memory_free_mb": torch.cuda.memory_reserved(i) / (1024 * 1024),
                                "memory_used_mb": torch.cuda.memory_allocated(i) / (1024 * 1024),
                                "temperature_c": 0,
                                "utilization_percent": 0,
                                "driver_version": "unknown",
                                "cuda_version": torch.version.cuda or "unknown",
                            }
                        )
            except:
                pass

        # ROCm (AMD GPUs)
        if not gpus:
            try:
                import torch

                if hasattr(torch, "_C") and hasattr(torch._C, "_rocm_version"):
                    gpus.append(
                        {
                            "index": 0,
                            "name": "AMD ROCm GPU",
                            "memory_total_mb": 0,
                            "memory_free_mb": 0,
                            "memory_used_mb": 0,
                            "temperature_c": 0,
                            "utilization_percent": 0,
                            "driver_version": "rocm",
                            "cuda_version": "rocm",
                        }
                    )
            except:
                pass

        return gpus

    @staticmethod
    def get_cuda_info() -> Dict[str, Any]:
        """Get CUDA information."""
        info = {
            "available": False,
            "version": None,
            "cudnn_version": None,
            "device_count": 0,
            "current_device": None,
        }

        try:
            import torch

            info["available"] = torch.cuda.is_available()
            info["version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()

            if torch.cuda.is_available():
                info["current_device"] = torch.cuda.current_device()

                # Try to get cuDNN version
                try:
                    import torch.backends.cudnn as cudnn

                    info["cudnn_version"] = cudnn.version()
                    info["cudnn_available"] = cudnn.is_available()
                except:
                    pass
        except ImportError:
            pass

        # Try nvidia-smi for more info
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
            info["nvidia_smi_available"] = result.returncode == 0
        except:
            info["nvidia_smi_available"] = False

        return info

    @staticmethod
    def get_optimal_device(prefer_gpu: bool = True) -> str:
        """Get optimal device for model execution."""
        if not prefer_gpu:
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except:
            pass

        return "cpu"

    @staticmethod
    def estimate_vram_usage(model_params: int, quantization: Optional[str] = None) -> float:
        """Estimate VRAM usage in MB."""
        bytes_per_param = 4  # Default float32

        if quantization:
            q_lower = quantization.lower()
            if "q4" in q_lower or "4bit" in q_lower:
                bytes_per_param = 0.5
            elif "q5" in q_lower:
                bytes_per_param = 0.625
            elif "q6" in q_lower:
                bytes_per_param = 0.75
            elif "q8" in q_lower or "8bit" in q_lower:
                bytes_per_param = 1
            elif "fp16" in q_lower or "16bit" in q_lower:
                bytes_per_param = 2

        return (model_params * bytes_per_param) / (1024 * 1024)


class ResourceMonitor:
    """Monitor system resources in real-time."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[ResourceUsage], None]] = []
        self._history: List[ResourceUsage] = []
        self._max_history = 100

    def start(self):
        """Start monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            usage = self._get_current_usage()
            self._history.append(usage)

            if len(self._history) > self._max_history:
                self._history.pop(0)

            for callback in self._callbacks:
                try:
                    callback(usage)
                except:
                    pass

            time.sleep(self.interval)

    def _get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            memory_used = mem.used / (1024**3)
            memory_available = mem.available / (1024**3)
        except:
            cpu_percent = 0
            memory_used = 0
            memory_available = 0

        # GPU usage
        gpu_usage = []
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_usage.append(
                            {
                                "index": i,
                                "utilization": float(parts[0]),
                                "memory_used_mb": float(parts[1]),
                            }
                        )
        except:
            pass

        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used,
            memory_available_gb=memory_available,
            gpu_usage=gpu_usage,
            timestamp=time.time(),
        )

    def on_resource_update(self, callback: Callable[[ResourceUsage], None]):
        """Register callback for resource updates."""
        self._callbacks.append(callback)

    def get_average_usage(self, last_n: int = 10) -> Optional[ResourceUsage]:
        """Get average usage over last N samples."""
        if len(self._history) < last_n:
            return None

        recent = self._history[-last_n:]

        return ResourceUsage(
            cpu_percent=sum(u.cpu_percent for u in recent) / len(recent),
            memory_used_gb=sum(u.memory_used_gb for u in recent) / len(recent),
            memory_available_gb=sum(u.memory_available_gb for u in recent) / len(recent),
            gpu_usage=[],  # Complex to average
            timestamp=time.time(),
        )


class NeuralDoctor:
    """Main NeuralDoctor class - Advanced diagnostics and auto-fix system."""

    def __init__(self):
        self.fix_attempts: List[Dict] = []
        self.applied_fixes: List[str] = []
        self.resource_monitor = ResourceMonitor()

    def run_full_diagnostic(self) -> DiagnosticReport:
        """Run complete system diagnostic."""
        print("🔍 NeuralDoctor: Running full system diagnostic...")

        issues = []
        recommendations = []

        # OS Info
        os_info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        # CPU Info
        cpu_info = self._get_cpu_info()

        # GPU Info
        gpu_info = GPUManager.detect_gpus()

        # CUDA Info
        cuda_info = GPUManager.get_cuda_info()

        # Memory Info
        memory_info = self._get_memory_info()

        # Python Info
        python_info = {
            "version": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
        }

        # Check dependencies
        dependencies = self._check_dependencies()

        # Analyze issues
        if not gpu_info:
            issues.append(
                {
                    "type": ProblemType.CUDA_NOT_AVAILABLE.value,
                    "level": DiagnosticLevel.WARNING.value,
                    "message": "No GPU detected. Models will run on CPU (slower).",
                    "fixable": False,
                }
            )
            recommendations.append("Consider using a quantized model (Q4/Q5) for CPU inference")

        if not cuda_info["available"]:
            issues.append(
                {
                    "type": ProblemType.CUDA_NOT_AVAILABLE.value,
                    "level": DiagnosticLevel.WARNING.value,
                    "message": "CUDA not available. PyTorch CPU version installed?",
                    "fixable": True,
                }
            )
            recommendations.append(
                "Install CUDA-enabled PyTorch: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
            )

        # Check for common dependency issues
        critical_deps = ["torch", "transformers"]
        for dep in critical_deps:
            if not dependencies.get(dep, False):
                issues.append(
                    {
                        "type": ProblemType.DEPENDENCY_MISSING.value,
                        "level": DiagnosticLevel.ERROR.value,
                        "message": f"Critical dependency '{dep}' not installed",
                        "fixable": True,
                    }
                )

        # Check Horus-specific dependencies
        horus_check = self.check_horus_requirements()
        if not horus_check["ready"]:
            recommendations.append(
                "Install Horus dependencies: pip install llama-cpp-python huggingface_hub"
            )

        # Memory recommendations
        if memory_info.get("total_gb", 0) < 8:
            recommendations.append("Limited RAM detected. Use smaller models or enable swap.")

        report = DiagnosticReport(
            timestamp=time.time(),
            os_info=os_info,
            cpu_info=cpu_info,
            gpu_info=gpu_info,
            cuda_info=cuda_info,
            memory_info=memory_info,
            python_info=python_info,
            dependencies=dependencies,
            issues=issues,
            recommendations=recommendations,
        )

        print(f"✅ Diagnostic complete. Found {len(issues)} issues.")
        for rec in recommendations:
            print(f"💡 {rec}")

        return report

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information."""
        info = {"cores": os.cpu_count(), "threads": os.cpu_count()}  # Approximation

        try:
            import psutil

            info["cores_physical"] = psutil.cpu_count(logical=False)
            info["threads"] = psutil.cpu_count(logical=True)
            info["freq_mhz"] = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        except:
            pass

        return info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information."""
        info = {"total_gb": 0, "available_gb": 0}

        try:
            import psutil

            mem = psutil.virtual_memory()
            info["total_gb"] = mem.total / (1024**3)
            info["available_gb"] = mem.available / (1024**3)
            info["percent_used"] = mem.percent
        except:
            pass

        return info

    def _check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are installed."""
        deps = [
            "torch",
            "transformers",
            "llama_cpp",
            "onnxruntime",
            "autoawq",
            "optimum",
            "accelerate",
            "bitsandbytes",
            "psutil",
            "requests",
            "numpy",
            "huggingface_hub",
        ]

        results = {}
        for dep in deps:
            try:
                __import__(dep)
                results[dep] = True
            except ImportError:
                results[dep] = False

        return results

    def check_horus_requirements(self) -> Dict[str, Any]:
        """
        Check if all requirements for Horus models are satisfied.

        Returns:
            Dict with status and recommendations
        """
        print("\n[HORUS MODELS CHECK]")
        print("-" * 50)

        deps = self._check_dependencies()
        horus_deps = {
            "transformers": "Required for full models (Safetensors)",
            "torch": "Required for all models",
            "llama_cpp": "Required for GGUF models",
            "huggingface_hub": "Required for downloading models",
        }

        missing = []
        installed = []

        for dep, description in horus_deps.items():
            if deps.get(dep, False):
                installed.append(f"  [OK] {dep}: {description}")
            else:
                missing.append(f"  [X] {dep}: {description}")

        for item in installed:
            print(item)
        for item in missing:
            print(item)

        if missing:
            print("\nTo install missing dependencies:")
            print("  pip install llama-cpp-python huggingface_hub transformers torch")
            return {"ready": False, "missing": len(missing)}
        else:
            print("\n[OK] All Horus requirements satisfied!")
            return {"ready": True, "missing": 0}

    def auto_fix(self, problem_type: ProblemType, context: Dict = None) -> Tuple[bool, str]:
        """Attempt to automatically fix a problem.

        Returns:
            (success, message)
        """
        context = context or {}

        if problem_type == ProblemType.DEPENDENCY_MISSING:
            package = context.get("package", "unknown")
            return self._fix_dependency(package)

        elif problem_type == ProblemType.CUDA_NOT_AVAILABLE:
            return self._fix_cuda()

        elif problem_type == ProblemType.OUT_OF_MEMORY:
            return self._fix_oom(context)

        elif problem_type == ProblemType.API_KEY_MISSING:
            return False, "API key must be provided by user"

        return False, f"Auto-fix not implemented for {problem_type.value}"

    def _fix_dependency(self, package: str) -> Tuple[bool, str]:
        """Try to install missing dependency."""
        print(f"🔧 Attempting to install {package}...")

        install_commands = {
            "torch": "pip install torch torchvision torchaudio",
            "transformers": "pip install transformers",
            "llama_cpp": "pip install llama-cpp-python",
            "huggingface_hub": "pip install huggingface_hub",
            "onnxruntime": "pip install onnxruntime",
            "autoawq": "pip install autoawq",
            "optimum": "pip install optimum[exporters]",
            "accelerate": "pip install accelerate",
            "bitsandbytes": "pip install bitsandbytes",
            "psutil": "pip install psutil",
        }

        cmd = install_commands.get(package)
        if not cmd:
            return False, f"No install command for {package}"

        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self.applied_fixes.append(f"installed_{package}")
                return True, f"Successfully installed {package}"
            else:
                return False, f"Failed to install {package}: {result.stderr}"
        except Exception as e:
            return False, f"Error installing {package}: {e}"

    def _fix_cuda(self) -> Tuple[bool, str]:
        """Try to fix CUDA issues."""
        print("🔧 Checking CUDA installation...")

        # Check if CUDA toolkit is installed
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if not cuda_path:
            return False, "CUDA_PATH not set. CUDA toolkit may not be installed."

        return True, f"CUDA found at {cuda_path}"

    def _fix_oom(self, context: Dict) -> Tuple[bool, str]:
        """Try to fix out-of-memory errors."""
        print("🔧 Attempting to fix OOM error...")

        suggestions = [
            "1. Use a smaller model or quantization (Q4 instead of Q8)",
            "2. Reduce context length (max_tokens)",
            "3. Close other applications to free RAM",
            "4. Enable swap/memory paging",
            "5. Use CPU instead of GPU for very large models",
        ]

        return False, "OOM fixes require user action:\n" + "\n".join(suggestions)

    def recommend_model_config(self, model_path: str) -> Dict[str, Any]:
        """Recommend optimal configuration for a model."""
        report = self.run_full_diagnostic()

        # Detect model size
        model_size_gb = self._estimate_model_size(model_path)

        config = {
            "device": "cpu",
            "quantization": None,
            "context_length": 2048,
            "gpu_layers": 0,
            "recommended": True,
        }

        # Check GPU availability
        if report.gpu_info:
            gpu = report.gpu_info[0]
            vram_gb = gpu["memory_total_mb"] / 1024

            if vram_gb > model_size_gb * 1.5:
                # Enough VRAM for full model
                config["device"] = "cuda"
                config["gpu_layers"] = -1  # All layers
                config["context_length"] = 4096
            elif vram_gb > model_size_gb * 0.8:
                # Partial GPU offloading
                config["device"] = "cuda"
                config["gpu_layers"] = 20  # Partial
                config["quantization"] = "q5_k_m"
            else:
                # Need quantization
                config["device"] = "cuda"
                config["quantization"] = "q4_0"
                config["gpu_layers"] = -1
        else:
            # CPU only
            ram_gb = report.memory_info.get("total_gb", 8)

            if ram_gb < model_size_gb * 2:
                config["quantization"] = "q4_0"
                config["context_length"] = 1024
                config["recommended"] = False
                config["warning"] = "Model may be too large for available RAM"

        return config

    def _estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB."""
        path = Path(model_path)

        if path.is_file():
            return path.stat().st_size / (1024**3)

        # Directory - sum all model files
        total_bytes = 0
        for ext in [".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"]:
            for file in path.glob(f"**/*{ext}"):
                total_bytes += file.stat().st_size

        return total_bytes / (1024**3)

    def monitor_model(self, model_name: str, process_func: Callable):
        """Monitor a model's resource usage during execution."""
        print(f"🔍 Monitoring {model_name}...")

        self.resource_monitor.start()

        def check_resources(usage: ResourceUsage):
            # Alert if resources are critical
            if usage.memory_available_gb < 1:
                print(f"⚠️  Low memory! Only {usage.memory_available_gb:.1f} GB available")

            for gpu in usage.gpu_usage:
                if gpu.get("utilization", 0) > 95:
                    print(f"⚠️  GPU {gpu['index']} at {gpu['utilization']}% utilization!")

        self.resource_monitor.on_resource_update(check_resources)

        try:
            result = process_func()
            return result
        finally:
            self.resource_monitor.stop()
            avg = self.resource_monitor.get_average_usage()
            if avg:
                print(
                    f"📊 Average CPU: {avg.cpu_percent:.1f}%, Memory: {avg.memory_used_gb:.1f} GB"
                )


# Global NeuralDoctor instance
_doctor: Optional[NeuralDoctor] = None


def get_doctor() -> NeuralDoctor:
    """Get global NeuralDoctor instance."""
    global _doctor
    if _doctor is None:
        _doctor = NeuralDoctor()
    return _doctor


def diagnose() -> DiagnosticReport:
    """Quick diagnostic function."""
    return get_doctor().run_full_diagnostic()


def recommend_for_model(model_path: str) -> Dict[str, Any]:
    """Get recommendations for a model."""
    return get_doctor().recommend_model_config(model_path)


def get_gpu_info() -> List[Dict[str, Any]]:
    """Quick GPU info."""
    return GPUManager.detect_gpus()


def get_cuda_version() -> Optional[str]:
    """Get CUDA version."""
    info = GPUManager.get_cuda_info()
    return info.get("version")
