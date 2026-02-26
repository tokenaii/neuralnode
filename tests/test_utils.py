"""Tests for utility functions."""

import pytest
from datetime import datetime

from neuralnode.utils.time_context import (
    get_current_time_info,
    inject_time_context,
    format_time_for_prompt,
)
from neuralnode.local.hardware import HardwareDetector


class TestTimeContext:
    def test_get_current_time_info(self):
        info = get_current_time_info()
        assert "current_time" in info
        assert "current_date" in info
        assert "day_of_week" in info
        assert "timezone" in info
        assert "iso_timestamp" in info

    def test_inject_time_context(self):
        context = inject_time_context()
        assert "Current Date" in context
        assert "Current Time" in context

    def test_inject_with_custom_prompt(self):
        custom = "You are a helpful assistant."
        context = inject_time_context(custom)
        assert "You are a helpful assistant." in context
        assert "Current Date" in context

    def test_format_time_for_prompt(self):
        full = format_time_for_prompt("full")
        short = format_time_for_prompt("short")
        date_only = format_time_for_prompt("date_only")
        time_only = format_time_for_prompt("time_only")
        
        assert len(full) > len(date_only)
        assert len(full) > len(time_only)


class TestHardwareDetector:
    def test_hardware_detection(self):
        hw = HardwareDetector()
        # Should not raise any exceptions
        assert hw.ram_gb is not None
        assert hw.cpu_count is not None

    def test_has_cuda_property(self):
        hw = HardwareDetector()
        # Returns boolean, shouldn't crash
        _ = hw.has_cuda

    def test_recommend_model_config(self):
        hw = HardwareDetector()
        config = hw.recommend_model_config()
        
        assert "device" in config
        assert "quantization" in config
        assert "batch_size" in config
        assert "context_length" in config

    def test_can_run_model(self):
        hw = HardwareDetector()
        # Should return boolean for any model size
        result = hw.can_run_model("7b")
        assert isinstance(result, bool)

    def test_get_summary(self):
        hw = HardwareDetector()
        summary = hw.get_summary()
        assert "CPU" in summary
        assert "RAM" in summary

    def test_to_dict(self):
        hw = HardwareDetector()
        data = hw.to_dict()
        assert "cpu_count" in data
        assert "ram_gb" in data
        assert "has_cuda" in data
        assert "platform" in data


class TestAutoProvider:
    def test_get_available_providers(self):
        from neuralnode.utils.auto_provider import AutoProvider
        
        providers = AutoProvider.get_available_providers()
        assert isinstance(providers, list)
        # Should at least have HuggingFace (local)
        assert len(providers) >= 1

    def test_recommend_model(self):
        from neuralnode.utils.auto_provider import AutoProvider
        from neuralnode.local.hardware import HardwareDetector
        
        hw = HardwareDetector()
        model = AutoProvider.recommend_model(hw)
        assert isinstance(model, str)
        assert len(model) > 0
