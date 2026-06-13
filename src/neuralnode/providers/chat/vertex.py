"""Google Cloud Vertex AI Chat Provider for NeuralNode."""

import os
import logging
from typing import Any, Dict, List, Optional, Union

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType

logger = logging.getLogger(__name__)


class VertexAIChatProvider(BaseLLMProvider):
    """Google Cloud Vertex AI provider."""
    provider_name = "vertexai"
    provider_type = ProviderType.CLOUD_API
    default_model = "gemini-1.5-flash"
    
    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            vision=True,
            embeddings=True,
            context_window=1000000,
        )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import google.cloud.aiplatform as aiplatform
            project_id = kwargs.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
            location = kwargs.get("location") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            
            aiplatform.init(project=project_id, location=location)
            self.project_id = project_id
        except ImportError:
            raise ImportError("google-cloud-aiplatform required. Install: pip install google-cloud-aiplatform")
    
    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to Vertex AI."""
        try:
            from vertexai.generative_models import GenerativeModel
            
            model = model or self.model
            gemini = GenerativeModel(model)
            
            # Convert messages
            chat_history = []
            current_message = ""
            
            for msg in messages:
                if msg.get("role") == "user":
                    current_message = msg.get("content", "")
                elif msg.get("role") == "assistant":
                    chat_history.append({"role": "model", "parts": [msg.get("content", "")]})
            
            chat = gemini.start_chat(history=chat_history)
            response = chat.send_message(
                current_message,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens or 4096,
                }
            )
            
            return {
                "content": response.text,
                "model": model,
                "provider": self.provider_name,
            }
        except Exception as e:
            raise RuntimeError(f"Vertex AI error: {e}")
    
    async def achat(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))
    
    def embed(self, texts, **kwargs):
        try:
            from vertexai.language_models import TextEmbeddingModel
            
            if isinstance(texts, str):
                texts = [texts]
            
            model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
            embeddings = model.get_embeddings(texts)
            
            result = [emb.values for emb in embeddings]
            return result[0] if len(result) == 1 else result
        except Exception as e:
            raise RuntimeError(f"Vertex AI embeddings error: {e}")
    
    async def aembed(self, texts, **kwargs):
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))
