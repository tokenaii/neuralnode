"""Runtime dependency installer for optional NeuralNode features."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Tuple

# feature -> [(import_name, pip_spec), ...]
FEATURE_DEPENDENCIES: Dict[str, List[Tuple[str, str]]] = {
    "horus_transformers": [
        ("torch", "torch>=2.0.0"),
        ("transformers", "transformers>=4.35.0"),
        ("accelerate", "accelerate>=0.24.0"),
        ("sentencepiece", "sentencepiece>=0.1.99"),
        ("google.protobuf", "protobuf>=3.20.0"),
        ("huggingface_hub", "huggingface_hub>=0.23.0"),
    ],
    "horus_gguf": [
        ("llama_cpp", "llama-cpp-python>=0.2.0"),
        ("huggingface_hub", "huggingface_hub>=0.23.0"),
    ],
    "replica_tts": [
        ("edge_tts", "edge-tts>=6.1.0"),
    ],
    "tts_edge": [
        ("edge_tts", "edge-tts>=6.1.0"),
    ],
    "speech": [
        ("speech_recognition", "SpeechRecognition>=3.10.0"),
    ],
    "speech_mp3": [
        ("pydub", "pydub>=0.25.1"),
    ],
    "telegram": [
        ("telegram", "python-telegram-bot>=20.0"),
    ],
}


def _is_installed(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def _auto_install_enabled(default: bool = True) -> bool:
    env = os.getenv("NEURALNODE_AUTO_INSTALL_DEPS", "")
    if not env:
        return default
    return env.strip().lower() in {"1", "true", "yes", "on"}


def ensure_feature_dependencies(feature: str, auto_install: bool = True) -> bool:
    """Ensure optional dependencies for a feature are installed.

    Returns True if all dependencies are available after the check/install attempt.
    """
    deps = FEATURE_DEPENDENCIES.get(feature, [])
    if not deps:
        return True

    missing = [(mod, pkg) for mod, pkg in deps if not _is_installed(mod)]
    if not missing:
        return True

    if not (_auto_install_enabled(True) and auto_install):
        return False

    for _, pip_spec in missing:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
        except Exception:
            return False

    return all(_is_installed(mod) for mod, _ in deps)


__all__ = ["FEATURE_DEPENDENCIES", "ensure_feature_dependencies"]

