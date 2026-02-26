"""Voice and multi-modal support for NeuralNode."""

import base64
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import io


@dataclass
class AudioInput:
    """Audio input for STT."""
    content: bytes
    format: str = "mp3"  # mp3, wav, mp4, mpeg, mpga, m4a, ogg, webm
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "AudioInput":
        """Load audio from file."""
        with open(path, "rb") as f:
            return cls(content=f.read(), format=Path(path).suffix.lstrip("."))
    
    @classmethod
    def from_bytes(cls, data: bytes, format: str = "mp3") -> "AudioInput":
        """Create from bytes."""
        return cls(content=data, format=format)
    
    def to_base64(self) -> str:
        """Convert to base64."""
        return base64.b64encode(self.content).decode()


@dataclass
class ImageInput:
    """Image input for vision models."""
    content: bytes
    format: str = "png"  # png, jpeg, jpg, webp, gif
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ImageInput":
        """Load image from file."""
        with open(path, "rb") as f:
            ext = Path(path).suffix.lstrip(".").lower()
            if ext in ["jpg", "jpeg"]:
                ext = "jpeg"
            return cls(content=f.read(), format=ext)
    
    @classmethod
    def from_bytes(cls, data: bytes, format: str = "png") -> "ImageInput":
        """Create from bytes."""
        return cls(content=data, format=format)
    
    def to_base64(self) -> str:
        """Convert to base64."""
        return base64.b64encode(self.content).decode()
    
    def resize(self, max_size: int = 1024) -> "ImageInput":
        """Resize image while maintaining aspect ratio."""
        try:
            from PIL import Image
            
            img = Image.open(io.BytesIO(self.content))
            
            # Calculate new size
            width, height = img.size
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = io.BytesIO()
            img.save(buffer, format=self.format.upper() if self.format != "jpg" else "JPEG")
            return ImageInput(content=buffer.getvalue(), format=self.format)
        except ImportError:
            print("PIL not installed. Image resizing unavailable.")
            return self


class SpeechToText:
    """Speech-to-text using Whisper or cloud APIs."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "local":
            try:
                import whisper
                self._whisper = whisper.load_model("base")
            except ImportError:
                print("Whisper not installed. Install with: pip install openai-whisper")
                self._whisper = None
        else:
            self._whisper = None
    
    def transcribe(self, audio: Union[str, Path, AudioInput, bytes], language: Optional[str] = None) -> str:
        """Transcribe audio to text."""
        # Convert to AudioInput
        if isinstance(audio, (str, Path)):
            audio = AudioInput.from_file(audio)
        elif isinstance(audio, bytes):
            audio = AudioInput.from_bytes(audio)
        
        if self.provider == "local" and self._whisper:
            return self._transcribe_local(audio, language)
        elif self.provider == "openai":
            return self._transcribe_openai(audio, language)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _transcribe_local(self, audio: AudioInput, language: Optional[str]) -> str:
        """Transcribe using local Whisper."""
        if not self._whisper:
            return "Whisper not available"
        
        # Save to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.format}") as f:
            f.write(audio.content)
            temp_path = f.name
        
        try:
            result = self._whisper.transcribe(temp_path, language=language)
            return result["text"]
        finally:
            os.unlink(temp_path)
    
    def _transcribe_openai(self, audio: AudioInput, language: Optional[str]) -> str:
        """Transcribe using OpenAI Whisper API."""
        import urllib.request
        import json
        
        # Save to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio.format}") as f:
            f.write(audio.content)
            temp_path = f.name
        
        try:
            import requests
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            with open(temp_path, "rb") as f:
                files = {"file": f}
                data = {"model": "whisper-1"}
                if language:
                    data["language"] = language
                
                response = requests.post(url, headers=headers, files=files, data=data)
                result = response.json()
                return result.get("text", "")
        finally:
            os.unlink(temp_path)


class TextToSpeech:
    """Text-to-speech using various providers."""
    
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        
        if provider == "edge":
            try:
                import edge_tts
                self._edge_tts = edge_tts
            except ImportError:
                print("edge-tts not installed. Install with: pip install edge-tts")
                self._edge_tts = None
        else:
            self._edge_tts = None
    
    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[bytes, str]:
        """Convert text to speech."""
        if self.provider == "openai":
            return self._speak_openai(text, voice, output_path)
        elif self.provider == "edge":
            return self._speak_edge(text, voice, output_path)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _speak_openai(
        self,
        text: str,
        voice: Optional[str],
        output_path: Optional[Union[str, Path]]
    ) -> Union[bytes, str]:
        """Use OpenAI TTS."""
        import requests
        
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": text,
            "voice": voice or "alloy"
        }
        
        response = requests.post(url, headers=headers, json=data)
        audio_content = response.content
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(audio_content)
            return str(output_path)
        
        return audio_content
    
    async def _speak_edge(
        self,
        text: str,
        voice: Optional[str],
        output_path: Optional[Union[str, Path]]
    ) -> Union[bytes, str]:
        """Use Edge TTS (free, high quality)."""
        if not self._edge_tts:
            return "Edge TTS not available"
        
        import tempfile
        
        voice = voice or "en-US-AriaNeural"
        
        communicate = self._edge_tts.Communicate(text, voice)
        
        if output_path:
            await communicate.save(str(output_path))
            return str(output_path)
        else:
            # Save to temp file and return bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_path = f.name
            
            await communicate.save(temp_path)
            
            with open(temp_path, "rb") as f:
                content = f.read()
            
            import os
            os.unlink(temp_path)
            return content
    
    def get_voices(self) -> List[Dict[str, str]]:
        """Get available voices."""
        if self.provider == "openai":
            return [
                {"id": "alloy", "name": "Alloy", "gender": "neutral"},
                {"id": "echo", "name": "Echo", "gender": "male"},
                {"id": "fable", "name": "Fable", "gender": "male"},
                {"id": "onyx", "name": "Onyx", "gender": "male"},
                {"id": "nova", "name": "Nova", "gender": "female"},
                {"id": "shimmer", "name": "Shimmer", "gender": "female"},
            ]
        elif self.provider == "edge":
            # Common Edge voices
            return [
                {"id": "en-US-AriaNeural", "name": "Aria", "language": "en-US"},
                {"id": "en-US-GuyNeural", "name": "Guy", "language": "en-US"},
                {"id": "en-GB-SoniaNeural", "name": "Sonia", "language": "en-GB"},
                {"id": "ar-SA-ZariyahNeural", "name": "Zariyah", "language": "ar-SA"},
            ]
        return []


class VisionProcessor:
    """Process images for vision-capable LLMs."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def describe(
        self,
        image: Union[str, Path, ImageInput],
        prompt: str = "Describe this image in detail.",
    ) -> str:
        """Get LLM description of image."""
        # Convert to ImageInput
        if isinstance(image, (str, Path)):
            image = ImageInput.from_file(image)
        
        # Resize if needed
        image = image.resize(max_size=1024)
        
        # Create multimodal message
        base64_image = image.to_base64()
        
        # Check if LLM supports vision
        if hasattr(self.llm, 'supports_vision') and self.llm.supports_vision:
            return self.llm.chat(
                prompt,
                images=[f"data:image/{image.format};base64,{base64_image}"]
            )
        else:
            # Fallback: describe without vision
            return "Vision not supported by this model."
    
    def extract_text(self, image: Union[str, Path, ImageInput]) -> str:
        """Extract text from image (OCR)."""
        try:
            import pytesseract
            from PIL import Image as PILImage
            
            if isinstance(image, (str, Path)):
                img = PILImage.open(image)
            elif isinstance(image, ImageInput):
                img = PILImage.open(io.BytesIO(image.content))
            else:
                img = image
            
            text = pytesseract.image_to_string(img)
            return text
        except ImportError:
            return "OCR not available. Install pytesseract and PIL."


class MultiModalChat:
    """Chat with text, images, and audio."""
    
    def __init__(self, llm):
        self.llm = llm
        self.stt = None
        self.tts = None
        self.vision = None
    
    def enable_speech(self, provider: str = "openai", api_key: Optional[str] = None):
        """Enable speech capabilities."""
        self.stt = SpeechToText(provider, api_key)
        self.tts = TextToSpeech(provider, api_key)
    
    def enable_vision(self):
        """Enable vision capabilities."""
        self.vision = VisionProcessor(self.llm)
    
    def chat_with_image(
        self,
        text: str,
        image: Union[str, Path, ImageInput],
    ) -> str:
        """Chat with image input."""
        if not self.vision:
            self.enable_vision()
        
        return self.vision.describe(image, text)
    
    def chat_with_audio(
        self,
        audio: Union[str, Path, AudioInput],
        language: Optional[str] = None,
    ) -> str:
        """Chat with audio input (transcribe then chat)."""
        if not self.stt:
            raise ValueError("Speech not enabled. Call enable_speech() first.")
        
        # Transcribe
        text = self.stt.transcribe(audio, language)
        
        # Chat
        response = self.llm.chat(text)
        
        return response.text if hasattr(response, 'text') else str(response)
    
    def speak_response(
        self,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Union[bytes, str]:
        """Convert response to speech."""
        if not self.tts:
            raise ValueError("Speech not enabled. Call enable_speech() first.")
        
        if text is None:
            # Get last response from LLM
            raise ValueError("Text required")
        
        return self.tts.speak(text, voice, output_path)


# Convenience functions
def transcribe_audio(audio_path: str, provider: str = "openai", api_key: Optional[str] = None) -> str:
    """Quick audio transcription."""
    stt = SpeechToText(provider, api_key)
    return stt.transcribe(audio_path)


def speak_text(text: str, provider: str = "edge", voice: Optional[str] = None) -> bytes:
    """Quick text-to-speech."""
    tts = TextToSpeech(provider)
    return tts.speak(text, voice)


def describe_image(image_path: str, llm, prompt: str = "Describe this image.") -> str:
    """Quick image description."""
    vision = VisionProcessor(llm)
    return vision.describe(image_path, prompt)
