"""
Replica TTS for NeuralNode.

This module provides a dedicated import surface for edge_tts-based speech with
20 curated voices across multiple languages.

Users interact with stable custom Replica voice IDs, while the module resolves
them internally to the original edge_tts voice identifiers.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import importlib

from .utils.dependencies import ensure_feature_dependencies

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None

logger = logging.getLogger(__name__)


def _replica_voice_id(name: str, language: str) -> str:
    return f"replic-{name.lower()}-language{{{language.lower()}}}"


_VOICE_SPECS = [
    ("Aria", "en-us", "female", "American English female - professional", "en-US-AriaNeural"),
    ("Guy", "en-us", "male", "American English male - professional", "en-US-GuyNeural"),
    ("Jenny", "en-us", "female", "American English female - friendly", "en-US-JennyNeural"),
    ("Sonia", "en-gb", "female", "British English female", "en-GB-SoniaNeural"),
    ("Ryan", "en-gb", "male", "British English male", "en-GB-RyanNeural"),
    ("Salma", "ar-eg", "female", "Arabic Egyptian female", "ar-EG-SalmaNeural"),
    ("Shakir", "ar-eg", "male", "Arabic Egyptian male", "ar-EG-ShakirNeural"),
    ("Zariyah", "ar-sa", "female", "Arabic Saudi female", "ar-SA-ZariyahNeural"),
    ("Denise", "fr-fr", "female", "French female", "fr-FR-DeniseNeural"),
    ("Henri", "fr-fr", "male", "French male", "fr-FR-HenriNeural"),
    ("Elvira", "es-es", "female", "Spanish female", "es-ES-ElviraNeural"),
    ("Dalia", "es-mx", "female", "Mexican Spanish female", "es-MX-DaliaNeural"),
    ("Katja", "de-de", "female", "German female", "de-DE-KatjaNeural"),
    ("Conrad", "de-de", "male", "German male", "de-DE-ConradNeural"),
    ("Elsa", "it-it", "female", "Italian female", "it-IT-ElsaNeural"),
    ("Francisca", "pt-br", "female", "Brazilian Portuguese female", "pt-BR-FranciscaNeural"),
    ("Nanami", "ja-jp", "female", "Japanese female", "ja-JP-NanamiNeural"),
    ("Xiaoxiao", "zh-cn", "female", "Chinese (Simplified) female", "zh-CN-XiaoxiaoNeural"),
    ("SunHi", "ko-kr", "female", "Korean female", "ko-KR-SunHiNeural"),
    ("Emel", "tr-tr", "female", "Turkish female", "tr-TR-EmelNeural"),
]


AVAILABLE_VOICES = [
    {
        "id": _replica_voice_id(name, language),
        "name": name,
        "language": language,
        "gender": gender,
        "description": description,
        "edge_voice_id": edge_voice_id,
    }
    for name, language, gender, description, edge_voice_id in _VOICE_SPECS
]

VOICE_MAP = {voice["id"]: voice for voice in AVAILABLE_VOICES}
EDGE_VOICE_MAP = {voice["edge_voice_id"]: voice for voice in AVAILABLE_VOICES}
DEFAULT_VOICE_ID = AVAILABLE_VOICES[0]["id"]


def resolve_voice(voice_id: str) -> Dict:
    """Resolve a custom Replica voice ID or legacy edge_tts ID."""
    if voice_id in VOICE_MAP:
        return VOICE_MAP[voice_id]
    if voice_id in EDGE_VOICE_MAP:
        logger.warning(
            "Using legacy edge_tts voice ID '%s'. Prefer Replica custom ID '%s'.",
            voice_id,
            EDGE_VOICE_MAP[voice_id]["id"],
        )
        return EDGE_VOICE_MAP[voice_id]
    raise ValueError(
        f"Unknown Replica voice ID: {voice_id}. "
        f"Use one of: {[voice['id'] for voice in AVAILABLE_VOICES]}"
    )


def list_voices(language: Optional[str] = None, gender: Optional[str] = None) -> List[Dict]:
    voices = AVAILABLE_VOICES
    if language:
        voices = [voice for voice in voices if voice["language"] == language.lower()]
    if gender:
        voices = [voice for voice in voices if voice["gender"] == gender.lower()]
    return list(voices)


def replica_voice_list(language: Optional[str] = None, gender: Optional[str] = None) -> List[Dict]:
    """Return the curated Replica voices using the public custom voice IDs."""
    return list_voices(language=language, gender=gender)


class ReplicaTTS:
    """Dedicated edge_tts wrapper used by NeuralNode and Horus."""

    def __init__(
        self,
        voice_id: str = DEFAULT_VOICE_ID,
        language: str = "en-us",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        auto_install_deps: bool = True,
    ):
        global EDGE_TTS_AVAILABLE, edge_tts
        if not EDGE_TTS_AVAILABLE and auto_install_deps:
            ensure_feature_dependencies("replica_tts", auto_install=True)
            try:
                edge_tts = importlib.import_module("edge_tts")
                EDGE_TTS_AVAILABLE = True
            except ImportError:
                EDGE_TTS_AVAILABLE = False
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge_tts is required for Replica TTS. Install with: pip install edge-tts")

        voice = resolve_voice(voice_id)

        self.voice_id = voice["id"]
        self.edge_voice_id = voice["edge_voice_id"]
        self.language = voice["language"] if language == "en-us" else language.lower()
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

    async def _generate_speech(self, text: str, output_path: str) -> str:
        communicate = edge_tts.Communicate(
            text,
            voice=self.edge_voice_id,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )
        await communicate.save(output_path)
        return output_path

    def speak(self, text: str, blocking: bool = True, output_file: Optional[str] = None) -> str:
        if not output_file:
            output_file = f"replica_output_{abs(hash((self.voice_id, text))) % 100000}.mp3"

        output_path = str(Path(output_file).resolve())
        if not blocking:
            import threading

            thread = threading.Thread(
                target=lambda: asyncio.run(self._generate_speech(text, output_path)),
                daemon=True,
            )
            thread.start()
            return output_path

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, self._generate_speech(text, output_path)).result()
            return output_path

        asyncio.run(self._generate_speech(text, output_path))
        return output_path

    async def speak_async(self, text: str, output_file: Optional[str] = None) -> str:
        if not output_file:
            output_file = f"replica_output_{abs(hash((self.voice_id, text))) % 100000}.mp3"
        output_path = str(Path(output_file).resolve())
        await self._generate_speech(text, output_path)
        return output_path

    def save_to_file(self, text: str, filename: str) -> str:
        return self.speak(text, blocking=True, output_file=filename)

    def set_voice(self, voice_id: str) -> None:
        voice = resolve_voice(voice_id)
        self.voice_id = voice["id"]
        self.edge_voice_id = voice["edge_voice_id"]
        self.language = voice["language"]

    def set_language(self, language: str) -> None:
        self.language = language.lower()

    def set_rate(self, rate: str) -> None:
        self.rate = rate

    def set_pitch(self, pitch: str) -> None:
        self.pitch = pitch

    def set_volume(self, volume: str) -> None:
        self.volume = volume


def speak(text: str, voice_id: str = DEFAULT_VOICE_ID, language: str = "en-us") -> str:
    return ReplicaTTS(voice_id=voice_id, language=language).speak(text)


__all__ = [
    "ReplicaTTS",
    "list_voices",
    "replica_voice_list",
    "resolve_voice",
    "speak",
    "AVAILABLE_VOICES",
    "VOICE_MAP",
    "EDGE_VOICE_MAP",
    "DEFAULT_VOICE_ID",
]
