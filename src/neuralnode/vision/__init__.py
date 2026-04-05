"""NeuralNode Vision — Image and media support for vision-capable models.

Provides helpers for sending images to LLMs and a VisionTool for agents.

Example::

    import neuralnode as nn

    # Direct vision chat
    ai = nn.NeuralNode(provider="google", model="gemini-1.5-pro")
    response = ai.chat_with_image("What's in this photo?", image="photo.jpg")

    # From agent with VisionTool
    agent = ai.agent(tools=[nn.VisionTool()])
    result = agent.run("Analyze the chart in chart.png")
"""

import base64
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from ..core import BaseTool, ToolResult, ParameterSpec, Message, MessageRole, ensure_llm_response


def encode_image_b64(image_path: str) -> str:
    """Encode an image file as a base64 string.

    Args:
        image_path: Path to the image file (JPEG, PNG, GIF, WEBP).

    Returns:
        Base64-encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """Infer the MIME type from file extension."""
    ext = Path(image_path).suffix.lower()
    types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return types.get(ext, "image/jpeg")


def build_vision_message(
    text_prompt: str,
    image: Optional[str] = None,
    image_url: Optional[str] = None,
) -> Message:
    """Build a Message object with image data embedded in metadata.

    The provider implementations that support vision (Google, Anthropic,
    Ollama with vision models) know how to read ``message.metadata["images"]``
    and ``message.metadata["image_url"]``.

    Args:
        text_prompt: The text question / instruction.
        image: Local path to an image file.
        image_url: Public URL to an image.

    Returns:
        A user Message with image data in metadata.
    """
    metadata: Dict[str, Any] = {}

    if image:
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        b64 = encode_image_b64(str(path))
        media_type = get_image_media_type(str(path))
        metadata["images"] = [b64]
        metadata["image_media_type"] = media_type
        metadata["image_path"] = str(path)

    if image_url:
        metadata["image_url"] = image_url

    return Message(role=MessageRole.USER, content=text_prompt, metadata=metadata)


class VisionTool(BaseTool):
    """Agent tool for analyzing images with a vision-capable LLM.

    The agent can use this tool to analyze local images or image URLs.

    Example agent usage::

        agent = ai.agent(tools=[VisionTool()])
        result = agent.run("What does the chart in /tmp/chart.png show?")
    """

    name = "analyze_image"
    description = (
        "Analyze an image using a vision AI model. "
        "Provide a local file path or a public URL."
    )
    parameters = {
        "image_path": ParameterSpec(
            type="string",
            description="Local path to the image file (e.g. '/tmp/photo.jpg').",
            required=False,
            default=None,
        ),
        "image_url": ParameterSpec(
            type="string",
            description="Public URL to an image.",
            required=False,
            default=None,
        ),
        "prompt": ParameterSpec(
            type="string",
            description="What to ask about the image.",
            required=False,
            default="Describe this image in detail.",
        ),
    }

    def __init__(self, llm=None):
        """
        Args:
            llm: Vision-capable LLM provider. If None, the agent's LLM is used.
        """
        super().__init__()
        self._llm = llm

    def execute(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        prompt: str = "Describe this image in detail.",
    ) -> ToolResult:
        if not image_path and not image_url:
            return ToolResult.error("Provide either 'image_path' or 'image_url'.")

        if self._llm is None:
            return ToolResult.error(
                "No LLM attached to VisionTool. Pass the LLM during construction or use it inside an agent."
            )

        try:
            msg = build_vision_message(prompt, image=image_path, image_url=image_url)
            response = ensure_llm_response(self._llm.chat([msg]))
            return ToolResult.ok(content=response.content)
        except Exception as e:
            return ToolResult.error(f"Vision analysis failed: {e}")


def capture_screenshot(output: str = "screenshot.png") -> str:
    """Take a screenshot and save it to a file.

    Requires either ``pyautogui`` or ``Pillow``.

    Args:
        output: Output file path.

    Returns:
        Absolute path to the saved file.
    """
    try:
        import pyautogui
        img = pyautogui.screenshot()
        img.save(output)
        return str(Path(output).resolve())
    except ImportError:
        pass

    try:
        from PIL import ImageGrab
        img = ImageGrab.grab()
        img.save(output)
        return str(Path(output).resolve())
    except ImportError:
        raise ImportError("Install pyautogui or Pillow for screenshots: pip install pyautogui")


__all__ = [
    "encode_image_b64",
    "get_image_media_type",
    "build_vision_message",
    "VisionTool",
    "capture_screenshot",
]
