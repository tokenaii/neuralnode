"""Text-to-image inference for Hugging Face Diffusers models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


class HuggingFaceImageGenerator:
    """Load a Diffusers text-to-image model from Hugging Face and generate images.

    The Diffusers dependency is optional and imported only when the model is loaded.
    This class supports repositories accepted by ``AutoPipelineForText2Image``; it
    does not claim that every repository on Hugging Face is a text-to-image model.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        local_files_only: bool = False,
        **load_options: Any,
    ):
        if not model_id:
            raise ValueError("model_id must be a non-empty Hugging Face repository id or path.")
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.load_options = load_options
        self._pipeline = None

    def load(self):
        """Load the pipeline once and return it."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            import torch
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ImportError(
                "HuggingFace image generation requires torch and diffusers. "
                "Install them with `pip install 'neuralnode[image]'`."
            ) from exc

        device = self.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        dtype_name = self.torch_dtype
        if dtype_name is None:
            dtype_name = "float16" if device == "cuda" else "float32"
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")

        load_options = dict(self.load_options)
        load_options.update(
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
        )
        if self.cache_dir:
            load_options["cache_dir"] = self.cache_dir

        pipeline = AutoPipelineForText2Image.from_pretrained(self.model_id, **load_options)
        pipeline = pipeline.to(device)
        self.device = device
        self._pipeline = pipeline
        return pipeline

    def generate_image(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        **generation_options: Any,
    ):
        """Generate one image, optionally save it, and return the PIL image."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be at least 1.")
        for dimension, value in (("width", width), ("height", height)):
            if value is not None and (value < 8 or value % 8 != 0):
                raise ValueError(f"{dimension} must be a positive multiple of 8.")

        pipeline = self.load()
        options = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            **generation_options,
        }
        if negative_prompt is not None:
            options["negative_prompt"] = negative_prompt
        if width is not None:
            options["width"] = width
        if height is not None:
            options["height"] = height
        if seed is not None:
            import torch

            options["generator"] = torch.Generator(device=self.device).manual_seed(int(seed))

        images = getattr(pipeline(**options), "images", None)
        if not images:
            raise RuntimeError("Diffusers pipeline returned no images.")
        image = images[0]
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        return image


def load_hf_image_model(model_id: str, **kwargs) -> HuggingFaceImageGenerator:
    """Create a lazy-loading text-to-image generator for a Hugging Face model."""
    return HuggingFaceImageGenerator(model_id, **kwargs)


__all__ = ["HuggingFaceImageGenerator", "load_hf_image_model"]
