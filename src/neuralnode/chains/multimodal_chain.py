"""Multi-Modal Chain for processing voice, image, and text together."""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import base64
from io import BytesIO


class Modality(Enum):
    """Supported data modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalInput:
    """Multi-modal input data."""
    modality: Modality
    data: Any  # text string, image bytes, audio bytes
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_text(cls, text: str) -> "MultiModalInput":
        return cls(modality=Modality.TEXT, data=text)
    
    @classmethod
    def from_image(cls, image_path: str) -> "MultiModalInput":
        with open(image_path, 'rb') as f:
            return cls(modality=Modality.IMAGE, data=f.read())
    
    @classmethod
    def from_audio(cls, audio_path: str) -> "MultiModalInput":
        with open(audio_path, 'rb') as f:
            return cls(modality=Modality.AUDIO, data=f.read())


class MultiModalProcessor:
    """Process multi-modal inputs."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.processors: Dict[Modality, Callable] = {}
        self._setup_default_processors()
    
    def _setup_default_processors(self) -> None:
        """Setup default processors for each modality."""
        self.processors[Modality.TEXT] = self._process_text
        self.processors[Modality.IMAGE] = self._process_image
        self.processors[Modality.AUDIO] = self._process_audio
    
    def _process_text(self, data: str) -> str:
        """Process text input."""
        return data
    
    def _process_image(self, data: bytes) -> str:
        """Process image input - extract description."""
        if self.llm and hasattr(self.llm, 'vision'):
            # Use vision model
            return self.llm.vision.describe_image(data)
        
        # Fallback: indicate image presence
        return "[Image data]"
    
    def _process_audio(self, data: bytes) -> str:
        """Process audio input - transcribe."""
        try:
            from ..tools.multimodal import SpeechToText
            stt = SpeechToText()
            return stt.transcribe_bytes(data)
        except Exception:
            return "[Audio data]"
    
    def process(self, inputs: List[MultiModalInput]) -> List[str]:
        """Process multiple modal inputs."""
        results = []
        
        for inp in inputs:
            processor = self.processors.get(inp.modality)
            if processor:
                result = processor(inp.data)
                results.append(result)
            else:
                results.append(f"[Unsupported modality: {inp.modality}]")
        
        return results


class MultiModalChain:
    """Chain that processes multi-modal inputs."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.processor = MultiModalProcessor(llm)
        self.steps: List[Callable] = []
    
    def add_step(self, step: Callable) -> "MultiModalChain":
        """Add a processing step."""
        self.steps.append(step)
        return self
    
    def run(self, inputs: List[MultiModalInput]) -> Dict[str, Any]:
        """Run the chain on multi-modal inputs."""
        # Process each modality
        processed = self.processor.process(inputs)
        
        # Combine into context
        context = self._combine_modalities(processed, inputs)
        
        # Run through steps
        result = context
        for step in self.steps:
            result = step(result)
        
        return {
            "processed_inputs": processed,
            "context": context,
            "result": result,
        }
    
    def _combine_modalities(self, processed: List[str], inputs: List[MultiModalInput]) -> str:
        """Combine different modalities into a single context."""
        parts = []
        
        for inp, proc in zip(inputs, processed):
            if inp.modality == Modality.TEXT:
                parts.append(f"Text: {proc}")
            elif inp.modality == Modality.IMAGE:
                parts.append(f"Image shows: {proc}")
            elif inp.modality == Modality.AUDIO:
                parts.append(f"Audio says: {proc}")
        
        return "\n".join(parts)
    
    def chat_with_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        """Chat with multi-modal inputs."""
        inputs = []
        
        if text:
            inputs.append(MultiModalInput.from_text(text))
        if image_path:
            inputs.append(MultiModalInput.from_image(image_path))
        if audio_path:
            inputs.append(MultiModalInput.from_audio(audio_path))
        
        if not inputs:
            return "No input provided"
        
        result = self.run(inputs)
        context = result["context"]
        
        if self.llm:
            response = self.llm.chat(f"Based on this information:\n{context}\n\nPlease respond:")
            return response.text if hasattr(response, 'text') else str(response)
        
        return context


class VoiceImageTextChain:
    """Specialized chain for voice + image + text."""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.multi_chain = MultiModalChain(llm)
    
    def describe_image_with_audio(
        self,
        image_path: str,
        audio_path: str,
    ) -> Dict[str, str]:
        """Describe what's in image based on audio question."""
        # Transcribe audio
        from ..tools.multimodal import SpeechToText
        stt = SpeechToText()
        question = stt.transcribe(audio_path)
        
        # Describe image
        from ..tools.multimodal import VisionProcessor
        vision = VisionProcessor(llm=self.llm)
        image_desc = vision.describe_image(image_path)
        
        # Combine and answer
        if self.llm:
            prompt = f"""
Image: {image_desc}
Question: {question}

Answer the question about the image.
"""
            response = self.llm.chat(prompt)
            answer = response.text if hasattr(response, 'text') else str(response)
        else:
            answer = f"Image: {image_desc}\nQuestion: {question}"
        
        return {
            "transcribed_question": question,
            "image_description": image_desc,
            "answer": answer,
        }
    
    def generate_image_from_voice(
        self,
        audio_path: str,
        output_path: str,
    ) -> str:
        """Generate image based on voice description."""
        # Transcribe
        from ..tools.multimodal import SpeechToText
        stt = SpeechToText()
        description = stt.transcribe(audio_path)
        
        # Generate image (would use DALL-E or similar)
        # Placeholder for actual implementation
        return f"Would generate image for: {description}"


# Convenience functions
def chat_multimodal(
    text: Optional[str] = None,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    llm=None,
) -> str:
    """Quick multi-modal chat."""
    chain = MultiModalChain(llm)
    return chain.chat_with_multimodal(text, image, audio)


def process_voice_image(audio_path: str, image_path: str, llm=None) -> Dict[str, str]:
    """Process voice question about image."""
    chain = VoiceImageTextChain(llm)
    return chain.describe_image_with_audio(image_path, audio_path)
