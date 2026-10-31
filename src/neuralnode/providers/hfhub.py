"""HuggingFace Hub helpers: download ANY open-weights model from HF.

This module lets NeuralNode users pull any model on huggingface.co
(not just TokenAI models). It supports single-file formats (GGUF) and
repo formats (safetensors) with the minimal required files.

Example::

    from neuralnode.providers.hfhub import download_model, list_repo_files, HFModelRefs

    files = list_repo_files("gemma-ai/pygmalion-6b")
    path = download_model("foo/bar", file="model.gguf")

    # Then use:
    from neuralnode.providers.universal_local import UniversalLocalProvider
    ai = UniversalLocalProvider(model_path=path, device="auto")
"""

from __future__ import annotations

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

import requests

logger = logging.getLogger(__name__)

HF_API = "https://huggingface.co/api"
HF_RESOLVE = "https://huggingface.co/resolve/main"

# Common tokenizer/auxiliary files pulled along with weight files.
_AUX_EXTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "config.json",
    "LICENSE",
)


def _auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"User-Agent": "NeuralNode-HFHub/1.0"}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def repo_exists(repo_id: str) -> bool:
    """Return True if the HF repo is publicly available."""
    try:
        r = requests.get(f"{HF_API}/models/{repo_id}", timeout=30)
        return r.status_code == 200
    except Exception:
        return False


def repo_info(repo_id: str) -> Dict[str, Any]:
    """Fetch basic repo metadata from HuggingFace."""
    r = requests.get(f"{HF_API}/models/{repo_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def list_repo_files(repo_id: str) -> List[str]:
    """List file paths inside a HF repo."""
    r = requests.get(f"{HF_API}/models/{repo_id}/tree/main", timeout=30)
    r.raise_for_status()
    items = r.json()
    return [item["path"] for item in items if item.get("type") == "file"]


def infer_model_type(files: List[str]) -> str:
    """Infer the primary model format from repo file list."""
    for f in files:
        low = f.lower()
        if low.endswith(".gguf"):
            return "gguf"
    if any(f.lower().endswith(".safetensors") or f.lower().endswith(".bin") for f in files):
        return "safetensors"
    if any(f.lower().endswith(".onnx") for f in files):
        return "onnx"
    return "unknown"


def _download_file(
    repo_id: str, path: str, dest_dir: Path, progress_callback: Optional[Callable] = None
) -> Path:
    """Download a single file from HF repo into dest_dir preserving path."""
    out = dest_dir / Path(path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{HF_RESOLVE}/{repo_id}/{path}"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        downloaded = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total:
                    progress_callback(downloaded / total * 100, downloaded, total)
    return out


def _pick_weight_files(files: List[str], weight_suffixes=(".safetensors", ".bin")) -> List[str]:
    """Pick likely weight files, excluding training/misc files."""
    candidates = []
    for f in files:
        low = f.lower()
        if low.endswith(weight_suffixes) and not any(
            x in low for x in (".mp4", "optimizer", ".pt", ".pth", ".onnx")
        ):
            candidates.append(f)
    return candidates


def download_repo(
    repo_id: str,
    file: Optional[str] = None,
    cache_dir: Optional[str] = None,
    include_aux: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Path:
    """Download an HF model (or specific file) locally.

    For GGUF repos (or when ``file`` is given), downloads exactly the file.
    For safetensors repos, downloads the minimal required files: config/tokenizer
    plus every weight file needed (skipping extra aux files). Returns path to the
    primary weight file or a repo folder.
    """
    dest_root = (
        Path(cache_dir)
        if cache_dir
        else Path(os.path.expanduser(os.getenv("NEURALNODE_HF_CACHE", "~/.neuralnode/models/hf")))
    )
    dest = dest_root / repo_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)

    if file:
        return _download_file(repo_id, file, dest, progress_callback)

    files = list_repo_files(repo_id)
    if repo_info(repo_id).get("private"):
        raise RuntimeError("Private/gated HF repos are not supported.")

    kind = infer_model_type(files)
    if kind == "gguf":
        weight_files = [f for f in files if f.lower().endswith(".gguf")]
    else:
        weight_files = _pick_weight_files(files)

    if not weight_files:
        raise RuntimeError(f"No loadable weight files found in {repo_id} (kind={kind}).")

    primary = dest / Path(weight_files[0]).name
    if primary.exists() and primary.stat().st_size > 0:
        # Already present; verify the aux files too?
        if not include_aux:
            return primary
    else:
        for w in weight_files:
            _download_file(repo_id, w, dest, progress_callback)

    if include_aux:
        for f in files:
            low = f.lower()
            if low in _AUX_EXTS or low.endswith(tuple(_AUX_EXTS)):
                _download_file(repo_id, f, dest, progress_callback)

    return primary


def resolve_repo(repo_id: str) -> Path:
    """Ensure local copy of ``repo_id``; returns the primary model path."""
    return download_repo(repo_id)


def load_hf_model(repo_id: str, device: str = "auto", **kwargs):
    """Load any HF open-weights model into NeuralNode.

    Delegates to UniversalLocalProvider (or HorusProvider if the repo is a TokenAI
    Horus model). Supports GGUF, safetensors, AWQ/GPTQ, ONNX.
    """
    # If it's a known TokenAI Horus model, prefer HorusModel for best performance.
    try:
        from .horus import HorusModel

        models = HorusModel.list_available_models(canonical=True)
        canonical = [mid for mid in models.keys()]
        if repo_id in canonical or repo_id in canonical:
            cls = HorusModel
            return cls(model_id=repo_id, device=device, **kwargs)
    except Exception:
        pass

    path = download_repo(repo_id)
    from .universal_local import UniversalLocalProvider

    return UniversalLocalProvider(
        model_path=str(path),
        device=device,
        context_length=int(kwargs.pop("context_length", 4096) or 4096),
        **kwargs,
    )


__all__ = [
    "HF_API",
    "HF_RESOLVE",
    "repo_exists",
    "repo_info",
    "list_repo_files",
    "infer_model_type",
    "download_repo",
    "resolve_repo",
    "load_hf_model",
    "_AUX_EXTS",
]
