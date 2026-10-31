"""NeuralNode Speech Recognition - powered by the speech_recognition library.

Simple usage:
    from neuralnode.speech import NeuralSpeech

    sr = NeuralSpeech()

    # From microphone
    text = sr.listen()
    print(text)

    # From audio file
    text = sr.transcribe("audio.wav")
    print(text)

    # Specify language
    text = sr.listen(language="ar-EG")
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests

from ..utils.dependencies import ensure_feature_dependencies

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Supported languages (Google Speech Recognition)
# ──────────────────────────────────────────────

SUPPORTED_LANGUAGES: Dict[str, str] = {
    # Major languages
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "ar-EG": "Arabic (Egypt)",
    "ar-SA": "Arabic (Saudi Arabia)",
    "fr-FR": "French",
    "es-ES": "Spanish (Spain)",
    "es-MX": "Spanish (Mexico)",
    "de-DE": "German",
    "it-IT": "Italian",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)",
    "ja-JP": "Japanese",
    "ko-KR": "Korean",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "hi-IN": "Hindi",
    "tr-TR": "Turkish",
    "ru-RU": "Russian",
    "nl-NL": "Dutch",
    "pl-PL": "Polish",
    "sv-SE": "Swedish",
    "da-DK": "Danish",
    "fi-FI": "Finnish",
    "no-NO": "Norwegian",
    "el-GR": "Greek",
    "he-IL": "Hebrew",
    "th-TH": "Thai",
    "vi-VN": "Vietnamese",
    "id-ID": "Indonesian",
    "ms-MY": "Malay",
    "uk-UA": "Ukrainian",
    "ro-RO": "Romanian",
    "cs-CZ": "Czech",
    "hu-HU": "Hungarian",
    "bg-BG": "Bulgarian",
    "hr-HR": "Croatian",
    "sk-SK": "Slovak",
    "sl-SI": "Slovenian",
    "sr-RS": "Serbian",
    "ca-ES": "Catalan",
    "fil-PH": "Filipino",
    "ur-PK": "Urdu",
    "bn-BD": "Bengali",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "ml-IN": "Malayalam",
    "kn-IN": "Kannada",
    "gu-IN": "Gujarati",
    "mr-IN": "Marathi",
    "pa-IN": "Punjabi",
    "sw-KE": "Swahili",
    "af-ZA": "Afrikaans",
}
SPEECH_LANGUAGES = SUPPORTED_LANGUAGES


class NeuralSpeech:
    """NeuralNode Speech Recognition engine.

    Wraps the ``speech_recognition`` library with a simple API that
    supports 50+ languages out of the box.

    Example::

        from neuralnode.speech import NeuralSpeech

        sr = NeuralSpeech()
        text = sr.listen()          # from microphone
        text = sr.transcribe("a.wav")  # from file
        text = sr.listen(language="ar-EG")  # Arabic
    """

    def __init__(
        self,
        language: str = "en-US",
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        phrase_time_limit: Optional[float] = None,
        timeout: Optional[float] = None,
        auto_install_deps: bool = True,
    ):
        """
        Args:
            language: Default language code (e.g. ``'en-US'``, ``'ar-EG'``).
            energy_threshold: Minimum audio energy to consider for recording.
            pause_threshold: Seconds of non-speaking audio before a phrase is complete.
            phrase_time_limit: Maximum seconds for a phrase. ``None`` = no limit.
            timeout: Maximum seconds to wait for speech to start. ``None`` = no limit.
        """
        self.auto_install_deps = auto_install_deps
        try:
            if self.auto_install_deps:
                ensure_feature_dependencies("speech", auto_install=True)
            import speech_recognition as sr_lib
        except ImportError:
            raise ImportError(
                "speech_recognition is required. Install it:\n" "  pip install SpeechRecognition"
            )

        self.language = language
        self.phrase_time_limit = phrase_time_limit
        self.timeout = timeout

        self._recognizer = sr_lib.Recognizer()
        self._recognizer.energy_threshold = energy_threshold
        self._recognizer.pause_threshold = pause_threshold
        self._sr = sr_lib

    # ── Public API ────────────────────────────

    def listen(
        self,
        language: Optional[str] = None,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
    ) -> str:
        """Listen from the default microphone and return recognized text.

        Args:
            language: Override the default language for this call.
            timeout: Override the default timeout.
            phrase_time_limit: Override the default phrase time limit.

        Returns:
            Recognized text string.

        Raises:
            RuntimeError: If recognition fails or no speech detected.
        """
        lang = language or self.language
        tout = timeout or self.timeout
        plimit = phrase_time_limit or self.phrase_time_limit

        try:
            with self._sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Listening...")
                audio = self._recognizer.listen(
                    source,
                    timeout=tout,
                    phrase_time_limit=plimit,
                )
        except self._sr.WaitTimeoutError:
            raise RuntimeError("Timed out waiting for speech.")
        except Exception as e:
            raise RuntimeError(f"Microphone error: {e}")

        return self._recognize(audio, lang)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> str:
        """Transcribe an audio file to text.

        Supports WAV, AIFF, FLAC, and MP3 (if pydub is installed).

        Args:
            audio_path: Path to the audio file.
            language: Override the default language.

        Returns:
            Recognized text string.
        """
        lang = language or self.language
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Handle MP3 conversion
        if path.suffix.lower() == ".mp3":
            audio = self._load_mp3(str(path))
        else:
            with self._sr.AudioFile(str(path)) as source:
                audio = self._recognizer.record(source)

        return self._recognize(audio, lang)

    def listen_in_background(
        self,
        callback,
        language: Optional[str] = None,
    ):
        """Start background listening.

        Args:
            callback: Function called with ``(recognizer, text)`` on each phrase.
            language: Language code.

        Returns:
            Stop function – call it to stop background listening.
        """
        lang = language or self.language
        source = self._sr.Microphone()

        def _internal_callback(recognizer, audio):
            try:
                text = recognizer.recognize_google(audio, language=lang)
                callback(recognizer, text)
            except self._sr.UnknownValueError:
                callback(recognizer, "")
            except self._sr.RequestError as e:
                callback(recognizer, f"[Error: {e}]")

        stop_fn = self._recognizer.listen_in_background(source, _internal_callback)
        return stop_fn

    @staticmethod
    def list_languages() -> Dict[str, str]:
        """Return all supported language codes and their names."""
        return dict(SUPPORTED_LANGUAGES)

    @staticmethod
    def get_language_code(language_name: str) -> Optional[str]:
        """Resolve a simple language name to a supported language code."""
        query = language_name.strip().lower()
        direct = {
            "english": "en-US",
            "arabic": "ar-SA",
            "french": "fr-FR",
            "spanish": "es-ES",
            "german": "de-DE",
        }
        if query in direct:
            return direct[query]
        for code, name in SUPPORTED_LANGUAGES.items():
            if query in name.lower():
                return code
        return None

    # ── Internal ──────────────────────────────

    def _recognize(self, audio, language: str) -> str:
        """Run Google Speech Recognition on audio data."""
        try:
            text = self._recognizer.recognize_google(audio, language=language)
            return text
        except self._sr.UnknownValueError:
            raise RuntimeError("Could not understand the audio.")
        except self._sr.RequestError as e:
            raise RuntimeError(f"Speech recognition service error: {e}")

    def _load_mp3(self, path: str):
        """Load an MP3 file by converting to WAV via pydub."""
        try:
            if self.auto_install_deps:
                ensure_feature_dependencies("speech_mp3", auto_install=True)
            from pydub import AudioSegment
            import io

            sound = AudioSegment.from_mp3(path)
            wav_io = io.BytesIO()
            sound.export(wav_io, format="wav")
            wav_io.seek(0)

            with self._sr.AudioFile(wav_io) as source:
                return self._recognizer.record(source)
        except ImportError:
            raise ImportError("pydub is required to transcribe MP3 files:\n" "  pip install pydub")


class HTTPSTT:
    """Speech-to-text adapter for HTTP services accepting raw audio bytes.

    The default contract matches ``POST /transcribe`` services that return
    ``{"text": "..."}``, including the TokenAI Opet Whisper endpoint.
    """

    _CONTENT_TYPES = {
        ".wav": "audio/wav",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".mp4": "audio/mp4",
        ".m4a": "audio/mp4",
    }

    def __init__(
        self,
        base_url: str,
        token: Optional[str] = None,
        endpoint: str = "/transcribe",
        timeout: float = 90,
        token_header: str = "X-Opet-Tts-Token",
    ):
        self.url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        self.token = token
        self.timeout = timeout
        self.token_header = token_header

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        path = Path(audio_path)
        headers = {
            "Content-Type": self._CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        }
        if self.token:
            headers[self.token_header] = self.token
        response = requests.post(
            self.url, data=path.read_bytes(), headers=headers, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError("The speech service response must contain a text field.")
        return data["text"]


__all__ = [
    "NeuralSpeech",
    "HTTPSTT",
    "SUPPORTED_LANGUAGES",
    "SPEECH_LANGUAGES",
]
