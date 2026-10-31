"""NeuralNode TTS - Text-to-Speech powered by edge_tts and ElevenLabs.

Simple usage:
    from neuralnode.tts import NeuralTTS

    tts = NeuralTTS()
    tts.speak("Hello world!", output="hello.mp3")

    # Or with a specific voice
    tts = NeuralTTS(voice="en-US-AriaNeural")
    tts.speak("Hello!", output="out.mp3")

    # With ElevenLabs API
    tts = NeuralTTS(provider="elevenlabs", api_key="your-key")
    tts.speak("Hello!", output="out.mp3")
"""

import asyncio
import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..utils.dependencies import ensure_feature_dependencies

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Curated voice list (~20 voices across languages)
# ──────────────────────────────────────────────

VOICES: Dict[str, Dict[str, Any]] = {
    # ── English (5 voices) ──
    "en-US-AriaNeural": {"lang": "en", "gender": "Female", "name": "Aria", "locale": "en-US"},
    "en-US-GuyNeural": {"lang": "en", "gender": "Male", "name": "Guy", "locale": "en-US"},
    "en-US-JennyNeural": {"lang": "en", "gender": "Female", "name": "Jenny", "locale": "en-US"},
    "en-GB-SoniaNeural": {"lang": "en", "gender": "Female", "name": "Sonia", "locale": "en-GB"},
    "en-GB-RyanNeural": {"lang": "en", "gender": "Male", "name": "Ryan", "locale": "en-GB"},
    # ── Arabic (3 voices) ──
    "ar-EG-SalmaNeural": {"lang": "ar", "gender": "Female", "name": "Salma", "locale": "ar-EG"},
    "ar-EG-ShakirNeural": {"lang": "ar", "gender": "Male", "name": "Shakir", "locale": "ar-EG"},
    "ar-SA-ZariyahNeural": {"lang": "ar", "gender": "Female", "name": "Zariyah", "locale": "ar-SA"},
    # ── French (2 voices) ──
    "fr-FR-DeniseNeural": {"lang": "fr", "gender": "Female", "name": "Denise", "locale": "fr-FR"},
    "fr-FR-HenriNeural": {"lang": "fr", "gender": "Male", "name": "Henri", "locale": "fr-FR"},
    # ── Spanish (2 voices) ──
    "es-ES-ElviraNeural": {"lang": "es", "gender": "Female", "name": "Elvira", "locale": "es-ES"},
    "es-MX-DaliaNeural": {"lang": "es", "gender": "Female", "name": "Dalia", "locale": "es-MX"},
    # ── German (2 voices) ──
    "de-DE-KatjaNeural": {"lang": "de", "gender": "Female", "name": "Katja", "locale": "de-DE"},
    "de-DE-ConradNeural": {"lang": "de", "gender": "Male", "name": "Conrad", "locale": "de-DE"},
    # ── Japanese (1 voice) ──
    "ja-JP-NanamiNeural": {"lang": "ja", "gender": "Female", "name": "Nanami", "locale": "ja-JP"},
    # ── Chinese (1 voice) ──
    "zh-CN-XiaoxiaoNeural": {
        "lang": "zh",
        "gender": "Female",
        "name": "Xiaoxiao",
        "locale": "zh-CN",
    },
    # ── Korean (1 voice) ──
    "ko-KR-SunHiNeural": {"lang": "ko", "gender": "Female", "name": "SunHi", "locale": "ko-KR"},
    # ── Italian (1 voice) ──
    "it-IT-ElsaNeural": {"lang": "it", "gender": "Female", "name": "Elsa", "locale": "it-IT"},
    # ── Portuguese (1 voice) ──
    "pt-BR-FranciscaNeural": {
        "lang": "pt",
        "gender": "Female",
        "name": "Francisca",
        "locale": "pt-BR",
    },
    # ── Turkish (1 voice) ──
    "tr-TR-EmelNeural": {"lang": "tr", "gender": "Female", "name": "Emel", "locale": "tr-TR"},
    # ── Hindi (1 voice) ──
}

DEFAULT_VOICE = "en-US-AriaNeural"


def list_voices(language: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available TTS voices, optionally filtered by language code.

    Args:
        language: ISO 639-1 code like 'en', 'ar', 'fr'. None = all voices.

    Returns:
        List of voice info dicts.
    """
    results = []
    for voice_id, info in VOICES.items():
        if language and info["lang"] != language:
            continue
        results.append({"voice_id": voice_id, **info})
    return results


tts_list_voices = list_voices


class NeuralTTS:
    """NeuralNode Text-to-Speech engine.

    Supports two providers:
        - ``"edge"`` (default) – uses Microsoft Edge TTS via ``edge_tts``
        - ``"elevenlabs"`` – uses ElevenLabs API (requires ``api_key``)

    Example::

        from neuralnode.tts import NeuralTTS

        # Edge TTS (free, no key needed)
        tts = NeuralTTS(voice="en-US-AriaNeural")
        tts.speak("Hello world!", output="hello.mp3")

        # ElevenLabs
        tts = NeuralTTS(provider="elevenlabs", api_key="sk-...")
        tts.speak("Hi!", output="hi.mp3")
    """

    def __init__(
        self,
        voice: Optional[str] = None,
        provider: str = "edge",
        api_key: Optional[str] = None,
        elevenlabs_voice_id: Optional[str] = None,
        elevenlabs_model: str = "eleven_monolingual_v1",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        auto_install_deps: bool = True,
    ):
        """
        Args:
            voice: Voice identifier. For edge provider use keys from VOICES dict.
            provider: ``'edge'`` or ``'elevenlabs'``.
            api_key: API key for ElevenLabs (ignored for edge).
            elevenlabs_voice_id: ElevenLabs voice ID (only for elevenlabs provider).
            elevenlabs_model: ElevenLabs model to use.
            rate: Speech rate adjustment, e.g. ``"+20%"`` or ``"-10%"``.
            volume: Volume adjustment.
            pitch: Pitch adjustment.
        """
        self.provider = provider.lower()
        self.voice = voice or DEFAULT_VOICE
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        self.auto_install_deps = auto_install_deps

        # ElevenLabs settings
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.elevenlabs_model = elevenlabs_model

        if self.provider == "elevenlabs" and not self.api_key:
            raise ValueError(
                "ElevenLabs provider requires an api_key or ELEVENLABS_API_KEY env var."
            )

    # ── public API ────────────────────────────

    def speak(self, text: str, output: str = "output.mp3") -> str:
        """Convert text to speech and save to a file.

        Args:
            text: The text to speak.
            output: Output file path (mp3).

        Returns:
            Absolute path to the saved audio file.
        """
        if self.provider == "elevenlabs":
            return self._speak_elevenlabs(text, output)
        return self._speak_edge(text, output)

    async def aspeak(self, text: str, output: str = "output.mp3") -> str:
        """Async version of speak (edge provider only)."""
        if self.provider == "elevenlabs":
            return self._speak_elevenlabs(text, output)
        return await self._aspeak_edge(text, output)

    def speak_stream(self, text: str):
        """Stream audio bytes (edge provider). Yields (bytes, subtitle) tuples."""
        if self.provider != "edge":
            raise NotImplementedError("Streaming only supported for edge provider.")
        return asyncio.get_event_loop().run_until_complete(self._stream_edge(text))

    @staticmethod
    def list_voices(language: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available curated voices."""
        return list_voices(language)

    @staticmethod
    def get_voice_for_language(language: str) -> Optional[str]:
        """Return the first curated voice that matches the given language."""
        lang = language.lower()
        for voice_id, info in VOICES.items():
            if info["lang"].lower() == lang or info["locale"].lower().startswith(lang):
                return voice_id
        return None

    def save_to_file(self, text: str, filename: str) -> str:
        """Backward-compatible alias for saving speech to a file."""
        return self.speak(text, output=filename)

    def save(self, text: str, filename: str) -> str:
        """Short alias used by existing examples."""
        return self.save_to_file(text, filename)

    # ── Edge TTS ──────────────────────────────

    def _speak_edge(self, text: str, output: str) -> str:
        """Synchronous wrapper around edge_tts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._aspeak_edge(text, output))
                    return future.result()
            else:
                return loop.run_until_complete(self._aspeak_edge(text, output))
        except RuntimeError:
            return asyncio.run(self._aspeak_edge(text, output))

    async def _aspeak_edge(self, text: str, output: str) -> str:
        """Core edge_tts synthesis."""
        try:
            if self.auto_install_deps:
                ensure_feature_dependencies("tts_edge", auto_install=True)
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge_tts is required for TTS. Install it:\n" "  pip install edge-tts"
            )

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        await communicate.save(output)
        return str(Path(output).resolve())

    async def _stream_edge(self, text: str):
        """Stream audio chunks from edge_tts."""
        try:
            if self.auto_install_deps:
                ensure_feature_dependencies("tts_edge", auto_install=True)
            import edge_tts
        except ImportError:
            raise ImportError("edge_tts is required. pip install edge-tts")

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )
        chunks = []
        async for chunk in communicate.stream():
            chunks.append(chunk)
        return chunks

    # ── ElevenLabs ────────────────────────────

    def _speak_elevenlabs(self, text: str, output: str) -> str:
        """ElevenLabs TTS via REST API."""
        import requests

        voice_id = self.elevenlabs_voice_id or "21m00Tcm4TlvDq8ikWAM"  # default Rachel

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key,
        }
        payload = {
            "text": text,
            "model_id": self.elevenlabs_model,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
            },
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        with open(output, "wb") as f:
            f.write(response.content)

        return str(Path(output).resolve())


class HTTPTTS:
    """Text-to-speech adapter for HTTP services returning audio bytes."""

    def __init__(
        self,
        base_url: str,
        voice: Optional[str] = None,
        language: str = "en",
        token: Optional[str] = None,
        endpoint: str = "/tts",
        timeout: float = 90,
        token_header: str = "X-Opet-Tts-Token",
    ):
        self.url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        self.voice = voice
        self.language = language
        self.token = token
        self.timeout = timeout
        self.token_header = token_header

    def speak(self, text: str, output: str = "output.wav") -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers[self.token_header] = self.token
        response = requests.post(
            self.url,
            json={"text": text, "language": self.language, "voice": self.voice},
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        path = Path(output)
        path.write_bytes(response.content)
        return str(path.resolve())


__all__ = [
    "NeuralTTS",
    "HTTPTTS",
    "list_voices",
    "tts_list_voices",
    "VOICES",
    "DEFAULT_VOICE",
]
