"""
Comprehensive Embedding Model Providers for NeuralNode
20+ providers for text embeddings
"""

import os
import json
import logging
import requests
from typing import Any, Dict, List, Optional, Union, Iterator
from abc import abstractmethod
import numpy as np

from .base import (
    BaseLLMProvider,
    ProviderCapabilities,
    ProviderType,
    ModelType,
    ModelInfo,
    ProviderRegistry,
)

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(BaseLLMProvider):
    """Base class for embedding providers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = self.model or self.default_model
        self.embedding_dim = kwargs.get("embedding_dim", None)

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """L2 normalize embeddings."""
        result = []
        for emb in embeddings:
            arr = np.array(emb)
            norm = np.linalg.norm(arr)
            if norm > 0:
                result.append((arr / norm).tolist())
            else:
                result.append(emb)
        return result

    def chat(self, messages, **kwargs):
        raise NotImplementedError("Embedding providers don't support chat")

    async def achat(self, messages, **kwargs):
        raise NotImplementedError()


class CohereEmbeddingProvider(BaseEmbeddingProvider):
    """Cohere embedding models."""

    provider_name = "cohere-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "embed-english-v3.0"

    AVAILABLE_MODELS = [
        "embed-english-v3.0",
        "embed-multilingual-v3.0",
        "embed-english-light-v3.0",
        "embed-multilingual-light-v3.0",
        "embed-english-v2.0",
        "embed-english-light-v2.0",
        "embed-multilingual-v2.0",
    ]

    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,
    }

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=512,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("COHERE_API_KEY")
        self.embedding_dim = self.MODEL_DIMENSIONS.get(self.model, 1024)

        if not self.api_key:
            raise ValueError("Cohere API key required")

    def embed(self, texts, model=None, input_type="search_document", **kwargs):
        """Generate embeddings using Cohere.

        Args:
            texts: Text(s) to embed
            model: Model to use
            input_type: One of "search_document", "search_query", "classification", "clustering"
        """
        if isinstance(texts, str):
            texts = [texts]

        model = model or self.model

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "texts": texts,
            "input_type": input_type,
        }

        if "truncate" in kwargs:
            payload["truncate"] = kwargs["truncate"]

        response = requests.post(
            "https://api.cohere.com/v1/embed", headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings", [])
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace Inference API for embeddings."""

    provider_name = "huggingface-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    POPULAR_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "intfloat/e5-large-v2",
        "intfloat/multilingual-e5-large",
        "thenlper/gte-large",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_API_KEY")
        self.use_inference_api = kwargs.get("use_inference_api", True)

        if self.use_inference_api and not self.api_key:
            raise ValueError("HuggingFace API key required for Inference API")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using HuggingFace."""
        if isinstance(texts, str):
            texts = [texts]

        model = model or self.model

        if self.use_inference_api:
            return self._inference_api_embed(texts, model)
        else:
            raise NotImplementedError("Local HuggingFace embeddings via separate module")

    def _inference_api_embed(self, texts, model):
        """Use HuggingFace Inference API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=self.timeout,
        )
        response.raise_for_status()

        embeddings = response.json()

        # Handle different response formats
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                return embeddings[0] if len(embeddings) == 1 else embeddings
            elif isinstance(embeddings[0], float):
                return embeddings

        return embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class GoogleEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding models."""

    provider_name = "google-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "models/text-embedding-004"

    AVAILABLE_MODELS = [
        "models/text-embedding-004",
        "models/embedding-001",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=2048,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.embedding_dim = 768

        if not self.api_key:
            raise ValueError("Google API key required")

    def embed(self, texts, model=None, task_type="RETRIEVAL_DOCUMENT", **kwargs):
        """Generate embeddings using Google Gemini.

        Args:
            texts: Text(s) to embed
            model: Model to use
            task_type: One of RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY,
                      CLASSIFICATION, CLUSTERING
        """
        if isinstance(texts, str):
            texts = [texts]

        model = model or self.model

        embeddings = []
        for text in texts:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model}:embedContent"

            payload = {
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }

            if "title" in kwargs and task_type == "RETRIEVAL_DOCUMENT":
                payload["title"] = kwargs["title"]

            params = {"key": self.api_key}

            response = requests.post(url, params=params, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", {}).get("values", [])
            embeddings.append(embedding)

        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class MistralEmbeddingProvider(BaseEmbeddingProvider):
    """Mistral AI embedding models."""

    provider_name = "mistral-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "mistral-embed"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=8192,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        self.embedding_dim = 1024

        if not self.api_key:
            raise ValueError("Mistral API key required")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Mistral."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "input": texts,
        }

        response = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item.get("embedding", []) for item in data.get("data", [])]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class VoyageEmbeddingProvider(BaseEmbeddingProvider):
    """Voyage AI embedding models (high quality)."""

    provider_name = "voyage-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "voyage-2"

    AVAILABLE_MODELS = [
        "voyage-3",
        "voyage-3-lite",
        "voyage-2",
        "voyage-large-2",
        "voyage-code-2",
        "voyage-law-2",
        "voyage-finance-2",
        "voyage-multilingual-2",
    ]

    MODEL_DIMENSIONS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-2": 1024,
        "voyage-large-2": 1536,
        "voyage-code-2": 1536,
        "voyage-law-2": 1024,
        "voyage-finance-2": 1024,
        "voyage-multilingual-2": 1024,
    }

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=32000,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("VOYAGE_API_KEY")
        self.embedding_dim = self.MODEL_DIMENSIONS.get(self.model, 1024)

        if not self.api_key:
            raise ValueError("Voyage API key required")

    def embed(self, texts, model=None, input_type="document", **kwargs):
        """Generate embeddings using Voyage AI.

        Args:
            texts: Text(s) to embed
            model: Model to use
            input_type: "query" or "document"
        """
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "input": texts,
            "input_type": input_type,
        }

        if "truncation" in kwargs:
            payload["truncation"] = kwargs["truncation"]

        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class JinaEmbeddingProvider(BaseEmbeddingProvider):
    """Jina AI embedding models."""

    provider_name = "jina-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "jina-embeddings-v3"

    AVAILABLE_MODELS = [
        "jina-embeddings-v3",
        "jina-embeddings-v2-base-en",
        "jina-embeddings-v2-small-en",
        "jina-embeddings-v2-base-de",
        "jina-embeddings-v2-base-es",
        "jina-embeddings-v2-base-zh",
        "jina-embeddings-v2-base-code",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=8192,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("JINA_API_KEY")
        self.embedding_dim = 1024

        if not self.api_key:
            raise ValueError("Jina API key required")

    def embed(self, texts, model=None, task="text-matching", **kwargs):
        """Generate embeddings using Jina AI.

        Args:
            texts: Text(s) to embed
            model: Model to use
            task: task type for v3 model (text-matching, classification, etc.)
        """
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "input": texts,
        }

        if "jina-embeddings-v3" in (model or self.model):
            payload["task"] = task

        if "dimensions" in kwargs:
            payload["dimensions"] = kwargs["dimensions"]

        response = requests.post(
            "https://api.jina.ai/v1/embeddings", headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class MixedbreadEmbeddingProvider(BaseEmbeddingProvider):
    """mixedbread.ai embedding models."""

    provider_name = "mixedbread-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "mxbai-embed-large-v1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=512,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("MIXEDBREAD_API_KEY")
        self.embedding_dim = 1024

        if not self.api_key:
            raise ValueError("Mixedbread API key required")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using mixedbread.ai."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "input": texts,
        }

        if "dimensions" in kwargs:
            payload["dimensions"] = kwargs["dimensions"]
        if "normalized" in kwargs:
            payload["normalized"] = kwargs["normalized"]

        response = requests.post(
            "https://api.mixedbread.ai/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class NomicEmbeddingProvider(BaseEmbeddingProvider):
    """Nomic AI embedding models."""

    provider_name = "nomic-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "nomic-embed-text-v1.5"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=8192,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("NOMIC_API_KEY")
        self.embedding_dim = 768

        if not self.api_key:
            raise ValueError("Nomic API key required")

    def embed(self, texts, model=None, task_type="search_document", **kwargs):
        """Generate embeddings using Nomic."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "texts": texts,
            "task_type": task_type,
        }

        if "dimensionality" in kwargs:
            payload["dimensionality"] = kwargs["dimensionality"]

        response = requests.post(
            "https://api-atlas.nomic.ai/v1/embedding/text",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings", [])
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama local embedding models."""

    provider_name = "ollama-embeddings"
    provider_type = ProviderType.LOCAL
    default_model = "nomic-embed-text"

    AVAILABLE_MODELS = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
    ]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            context_window=8192,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = self.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.embedding_dim = kwargs.get("embedding_dim", 768)

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Ollama."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model or self.model,
                    "prompt": text,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data.get("embedding", []))

        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class TransformersEmbeddingProvider(BaseEmbeddingProvider):
    """Local HuggingFace Transformers embeddings."""

    provider_name = "transformers-embeddings"
    provider_type = ProviderType.LOCAL
    default_model = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self.device = kwargs.get("device", "cpu")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            context_window=512,
            async_support=False,
        )

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(self.model)
                self._model = AutoModel.from_pretrained(self.model)
                self._model.to(self.device)
                self._model.eval()

                # Get embedding dimension from model config
                self.embedding_dim = self._model.config.hidden_size
            except ImportError:
                raise ImportError(
                    "transformers and torch required. Install: pip install transformers torch"
                )

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using local transformers."""
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        try:
            import torch
            import torch.nn.functional as F

            # Tokenize
            encoded = self._tokenizer(
                texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Generate embeddings
            with torch.no_grad():
                model_output = self._model(**encoded)

            # Mean pooling
            attention_mask = encoded["attention_mask"]
            token_embeddings = model_output[0]
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

            result = embeddings.cpu().numpy().tolist()
            return result[0] if len(result) == 1 else result

        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {e}")

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Sentence-Transformers library embeddings."""

    provider_name = "sentence-transformers"
    provider_type = ProviderType.LOCAL
    default_model = "all-MiniLM-L6-v2"

    POPULAR_MODELS = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2",
        "all-mpnet-base-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self.device = kwargs.get("device", None)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=512,
            async_support=False,
        )

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model, device=self.device)
                self.embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. Install: pip install sentence-transformers"
                )

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using sentence-transformers."""
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=kwargs.get("normalize", True),
            show_progress_bar=kwargs.get("show_progress", False),
            batch_size=kwargs.get("batch_size", 32),
        )

        result = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        return result[0] if len(result) == 1 else result

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class InstructorEmbeddingProvider(BaseEmbeddingProvider):
    """INSTRUCTOR embeddings (instruction-finetuned)."""

    provider_name = "instructor-embeddings"
    provider_type = ProviderType.LOCAL
    default_model = "hkunlp/instructor-large"

    AVAILABLE_MODELS = [
        "hkunlp/instructor-xl",
        "hkunlp/instructor-large",
        "hkunlp/instructor-base",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self.instruction = kwargs.get("instruction", "Represent the sentence for retrieval: ")

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=512,
            async_support=False,
        )

    def _load_model(self):
        """Lazy load the INSTRUCTOR model."""
        if self._model is None:
            try:
                from InstructorEmbedding import INSTRUCTOR

                self._model = INSTRUCTOR(self.model)
            except ImportError:
                raise ImportError("INSTRUCTOR required. Install: pip install InstructorEmbedding")

    def embed(self, texts, model=None, instruction=None, **kwargs):
        """Generate embeddings using INSTRUCTOR."""
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        instruction = instruction or self.instruction

        # INSTRUCTOR expects list of [instruction, text] pairs
        texts_with_instructions = [[instruction, text] for text in texts]

        embeddings = self._model.encode(texts_with_instructions)

        result = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        return result[0] if len(result) == 1 else result

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class FastEmbedProvider(BaseEmbeddingProvider):
    """Qdrant FastEmbed (fast local embeddings)."""

    provider_name = "fastembed"
    provider_type = ProviderType.LOCAL
    default_model = "BAAI/bge-small-en-v1.5"

    AVAILABLE_MODELS = [
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-large",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None
        self.cache_dir = kwargs.get("cache_dir", None)
        self.threads = kwargs.get("threads", None)

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=512,
            async_support=False,
        )

    def _load_model(self):
        """Lazy load the FastEmbed model."""
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(
                    model_name=self.model,
                    cache_dir=self.cache_dir,
                    threads=self.threads,
                )
            except ImportError:
                raise ImportError("fastembed required. Install: pip install fastembed")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using FastEmbed."""
        if isinstance(texts, str):
            texts = [texts]

        self._load_model()

        embeddings = list(self._model.embed(texts, batch_size=kwargs.get("batch_size", 256)))

        # Convert numpy arrays to lists
        result = [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]
        return result[0] if len(result) == 1 else result

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class EmbeddingStudioProvider(BaseEmbeddingProvider):
    """Embedding Studio (hosted embeddings service)."""

    provider_name = "embedding-studio"
    provider_type = ProviderType.CLOUD_API
    default_model = "default"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("EMBEDDING_STUDIO_API_KEY")
        self.base_url = self.base_url or "https://api.embeddingstudio.com"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            async_support=True,
        )

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Embedding Studio."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "texts": texts,
        }

        response = requests.post(
            f"{self.base_url}/v1/embeddings", headers=headers, json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings", [])
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class UpstageEmbeddingProvider(BaseEmbeddingProvider):
    """Upstage Solar embedding models."""

    provider_name = "upstage-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "solar-embedding-1-large"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=4096,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("UPSTAGE_API_KEY")
        self.embedding_dim = 4096

        if not self.api_key:
            raise ValueError("Upstage API key required")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Upstage."""
        if isinstance(texts, str):
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model or self.model,
            "input": texts,
        }

        response = requests.post(
            "https://api.upstage.ai/v1/solar/embeddings",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


class GeckoEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gecko embedding models (via Vertex AI)."""

    provider_name = "gecko-embeddings"
    provider_type = ProviderType.CLOUD_API
    default_model = "textembedding-gecko@latest"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            embeddings=True,
            batch_inference=True,
            context_window=3072,
            async_support=True,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project_id = kwargs.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = kwargs.get("location") or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self.project_id:
            raise ValueError("Google Cloud project_id required")

    def embed(self, texts, model=None, **kwargs):
        """Generate embeddings using Gecko via Vertex AI."""
        try:
            from vertexai.language_models import TextEmbeddingModel

            if isinstance(texts, str):
                texts = [texts]

            model = TextEmbeddingModel.from_pretrained(model or self.model)
            embeddings = model.get_embeddings(texts)

            result = [emb.values for emb in embeddings]
            return result[0] if len(result) == 1 else result

        except ImportError:
            raise ImportError("google-cloud-aiplatform required")

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))


# Register all embedding providers
for provider_class in [
    CohereEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    GoogleEmbeddingProvider,
    MistralEmbeddingProvider,
    VoyageEmbeddingProvider,
    JinaEmbeddingProvider,
    MixedbreadEmbeddingProvider,
    NomicEmbeddingProvider,
    OllamaEmbeddingProvider,
    TransformersEmbeddingProvider,
    SentenceTransformersProvider,
    InstructorEmbeddingProvider,
    FastEmbedProvider,
    EmbeddingStudioProvider,
    UpstageEmbeddingProvider,
    GeckoEmbeddingProvider,
]:
    ProviderRegistry.register(provider_class.provider_name, provider_class)


__all__ = [
    "BaseEmbeddingProvider",
    "CohereEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "GoogleEmbeddingProvider",
    "MistralEmbeddingProvider",
    "VoyageEmbeddingProvider",
    "JinaEmbeddingProvider",
    "MixedbreadEmbeddingProvider",
    "NomicEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "TransformersEmbeddingProvider",
    "SentenceTransformersProvider",
    "InstructorEmbeddingProvider",
    "FastEmbedProvider",
    "EmbeddingStudioProvider",
    "UpstageEmbeddingProvider",
    "GeckoEmbeddingProvider",
]
