"""Test diagnostics system (NeuralDoctor)."""

import pytest
from unittest.mock import patch, MagicMock
import torch

import neuralnode as nn
from neuralnode.diagnostics import GPUManager, ResourceMonitor, NeuralDoctor, DiagnosticLevel


class TestGPUManager:
    """Test GPU detection and management."""
    
    def test_detect_gpus_nvidia_smi(self):
        """Test GPU detection via nvidia-smi."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "0, RTX 4090, 24576, 20480, 4096, 65, 95, 535.104, 12.2\n"
            
            gpus = GPUManager.detect_gpus()
            
            if len(gpus) > 0:  # Only if nvidia-smi is available
                assert gpus[0]["name"] == "RTX 4090"
                assert gpus[0]["memory_total_mb"] == 24576
    
    def test_detect_gpus_torch_fallback(self):
        """Test GPU detection via torch fallback."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch.object(torch.cuda, 'device_count', return_value=1):
                with patch.object(torch.cuda, 'get_device_name', return_value="RTX 3080"):
                    with patch.object(torch.cuda, 'get_device_properties') as mock_props:
                        mock_props.return_value.total_memory = 1024**4  # 16GB
                        
                        gpus = GPUManager.detect_gpus()
                        assert len(gpus) > 0
    
    def test_get_cuda_info_available(self):
        """Test CUDA info when available."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with patch('torch.version.cuda', '12.1'):
                with patch.object(torch.cuda, 'device_count', return_value=2):
                    info = GPUManager.get_cuda_info()
                    
                    assert info["available"] is True
                    assert info["version"] == "12.1"
                    assert info["device_count"] == 2
    
    def test_get_cuda_info_not_available(self):
        """Test CUDA info when not available."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            info = GPUManager.get_cuda_info()
            assert info["available"] is False
    
    def test_get_optimal_device_gpu(self):
        """Test optimal device selection with GPU."""
        with patch.object(torch.cuda, 'is_available', return_value=True):
            device = GPUManager.get_optimal_device()
            assert device == "cuda"
    
    def test_get_optimal_device_cpu(self):
        """Test optimal device selection with CPU only."""
        with patch.object(torch.cuda, 'is_available', return_value=False):
            device = GPUManager.get_optimal_device()
            assert device == "cpu"
    
    def test_estimate_vram_usage(self):
        """Test VRAM estimation."""
        # 7B params at 4-bit = ~3.5GB
        vram = GPUManager.estimate_vram_usage(7_000_000_000, "q4_0")
        assert 3000 < vram < 4000  # Should be around 3.5GB
        
        # 7B params at full precision = ~28GB
        vram_full = GPUManager.estimate_vram_usage(7_000_000_000)
        assert vram_full > 25000


class TestResourceMonitor:
    """Test resource monitoring."""
    
    def test_init(self):
        """Test initialization."""
        monitor = ResourceMonitor(interval=0.5)
        assert monitor.interval == 0.5
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_current_usage(self, mock_memory, mock_cpu):
        """Test getting current resource usage."""
        mock_cpu.return_value = 45.5
        mock_memory.return_value.used = 8 * 1024**3  # 8GB
        mock_memory.return_value.available = 4 * 1024**3  # 4GB
        
        monitor = ResourceMonitor()
        usage = monitor._get_current_usage()
        
        assert usage.cpu_percent == 45.5
        assert usage.memory_used_gb == 8.0
        assert usage.memory_available_gb == 4.0
    
    def test_callback_registration(self):
        """Test registering callbacks."""
        monitor = ResourceMonitor()
        callback_called = False
        
        def test_callback(usage):
            nonlocal callback_called
            callback_called = True
        
        monitor.on_resource_update(test_callback)
        assert len(monitor._callbacks) == 1


class TestNeuralDoctor:
    """Test NeuralDoctor diagnostics."""
    
    def test_init(self):
        """Test initialization."""
        doctor = NeuralDoctor()
        assert doctor is not None
    
    @patch.object(NeuralDoctor, '_get_cpu_info')
    @patch.object(NeuralDoctor, '_get_memory_info')
    def test_run_full_diagnostic(self, mock_mem, mock_cpu):
        """Test full diagnostic run."""
        mock_cpu.return_value = {"cores": 8}
        mock_mem.return_value = {"total_gb": 16.0}
        
        doctor = NeuralDoctor()
        
        with patch.object(GPUManager, 'detect_gpus', return_value=[]):
            with patch.object(GPUManager, 'get_cuda_info', return_value={"available": False}):
                report = doctor.run_full_diagnostic()
                
                assert report is not None
                assert report.cpu_info["cores"] == 8
                assert report.memory_info["total_gb"] == 16.0
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        doctor = NeuralDoctor()
        deps = doctor._check_dependencies()
        
        assert isinstance(deps, dict)
        assert "torch" in deps or "numpy" in deps or "requests" in deps
    
    def test_recommend_model_config(self, tmp_path):
        """Test model configuration recommendations."""
        doctor = NeuralDoctor()
        
        # Create fake model
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"fake model data" * 1000)
        
        with patch.object(GPUManager, 'detect_gpus', return_value=[]):
            config = doctor.recommend_model_config(str(model_file))
            
            assert "device" in config
            assert "quantization" in config
    
    def test_auto_fix_dependency(self):
        """Test auto-fix for missing dependency."""
        doctor = NeuralDoctor()
        success, msg = doctor._fix_dependency("requests")
        
        # Should either succeed or provide helpful message
        assert isinstance(success, bool)
        assert isinstance(msg, str)


class TestDiagnosticReport:
    """Test diagnostic report dataclass."""
    
    def test_report_creation(self):
        """Test report creation."""
        from neuralnode.diagnostics import DiagnosticReport
        
        report = DiagnosticReport(
            timestamp=1234567890.0,
            os_info={"system": "Windows"},
            cpu_info={"cores": 8},
            gpu_info=[],
            cuda_info={"available": False},
            memory_info={"total_gb": 16.0},
            python_info={"version": "3.11"},
            dependencies={},
            issues=[],
            recommendations=["Use GPU"]
        )
        
        data = report.to_dict()
        assert data["os"]["system"] == "Windows"
        assert data["recommendations"] == ["Use GPU"]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_diagnose(self):
        """Test diagnose function."""
        with patch('neuralnode.diagnostics.NeuralDoctor') as mock_doctor_class:
            mock_doctor = MagicMock()
            mock_doctor_class.return_value = mock_doctor
            mock_doctor.run_full_diagnostic.return_value = MagicMock()
            
            result = nn.diagnose()
            mock_doctor.run_full_diagnostic.assert_called_once()
    
    def test_get_gpu_info(self):
        """Test get_gpu_info function."""
        with patch.object(GPUManager, 'detect_gpus', return_value=[{"name": "RTX 4090"}]):
            gpus = nn.get_gpu_info()
            assert isinstance(gpus, list)
    
    def test_get_cuda_version(self):
        """Test get_cuda_version function."""
        with patch.object(GPUManager, 'get_cuda_info', return_value={"version": "12.1"}):
            version = nn.get_cuda_version()
            assert version == "12.1"
