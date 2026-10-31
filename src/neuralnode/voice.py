"""Composable, one-turn speech-to-agent-to-speech workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class VoiceTurn:
    """Transcript, agent response, and optional synthesized audio path."""

    transcript: str
    response: str
    audio_path: Optional[str] = None


class VoiceAgent:
    """Connect an STT engine, an agent, and a TTS engine for one voice turn.

    This is a sequential turn pipeline. It does not provide full-duplex or
    low-latency streaming audio.
    """

    def __init__(self, agent: Any, speech: Any, tts: Any):
        self.agent = agent
        self.speech = speech
        self.tts = tts

    @staticmethod
    def _response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output", response.get("content"))
        if output is None:
            raise TypeError("The agent response must be text or contain an output field.")
        return str(output)

    def process_text(self, text: str, output_audio: Optional[str] = None) -> VoiceTurn:
        """Run text through the agent and optionally synthesize its response."""
        response = self._response_text(self.agent.run(text))
        audio_path = self.tts.speak(response, output=output_audio) if output_audio else None
        return VoiceTurn(transcript=text, response=response, audio_path=audio_path)

    def process_audio_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        output_audio: Optional[str] = None,
    ) -> VoiceTurn:
        """Transcribe an audio file, run the agent, and optionally synthesize."""
        transcript = self.speech.transcribe(audio_path, language=language)
        return self.process_text(transcript, output_audio=output_audio)

    def listen_once(
        self,
        language: Optional[str] = None,
        output_audio: Optional[str] = None,
    ) -> VoiceTurn:
        """Capture one microphone utterance and run a complete voice turn."""
        transcript = self.speech.listen(language=language)
        return self.process_text(transcript, output_audio=output_audio)

    async def aprocess_text(self, text: str, output_audio: Optional[str] = None) -> VoiceTurn:
        """Async variant using agent/TTS async methods when available."""
        if hasattr(self.agent, "arun"):
            response = self._response_text(await self.agent.arun(text))
        else:
            response = self._response_text(self.agent.run(text))
        if output_audio:
            if hasattr(self.tts, "aspeak"):
                audio_path = await self.tts.aspeak(response, output=output_audio)
            else:
                audio_path = self.tts.speak(response, output=output_audio)
        else:
            audio_path = None
        return VoiceTurn(transcript=text, response=response, audio_path=audio_path)


__all__ = ["VoiceAgent", "VoiceTurn"]
