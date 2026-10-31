"""AWS Bedrock Chat Provider for NeuralNode."""

import os
import logging
from typing import Any, Dict, List, Optional, Union
import json

from ..base import BaseLLMProvider, ProviderCapabilities, ProviderType

logger = logging.getLogger(__name__)


class BedrockChatProvider(BaseLLMProvider):
    """AWS Bedrock provider."""

    provider_name = "bedrock"
    provider_type = ProviderType.CLOUD_API
    default_model = "anthropic.claude-3-sonnet-20240229-v1:0"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            chat=True,
            streaming=True,
            function_calling=True,
            embeddings=True,
            context_window=200000,
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import boto3

            self.session = boto3.Session(
                aws_access_key_id=kwargs.get("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=kwargs.get("aws_secret_access_key")
                or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=kwargs.get("region_name") or os.getenv("AWS_REGION", "us-east-1"),
            )
            self.client = self.session.client("bedrock-runtime")
        except ImportError:
            raise ImportError("boto3 required for AWS Bedrock. Install: pip install boto3")
        except Exception as e:
            raise ValueError(f"Failed to initialize AWS Bedrock client: {e}")

    def chat(self, messages, model=None, temperature=0.7, max_tokens=None, **kwargs):
        """Send chat request to AWS Bedrock."""
        model = model or self.model

        # Format messages for Bedrock
        bedrock_messages = []
        system_prompt = None

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            else:
                bedrock_messages.append(
                    {
                        "role": msg.get("role"),
                        "content": [{"type": "text", "text": msg.get("content", "")}],
                    }
                )

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or 4096,
            "messages": bedrock_messages,
            "temperature": temperature,
        }

        if system_prompt:
            body["system"] = system_prompt

        response = self.client.invoke_model(modelId=model, body=json.dumps(body))

        result = json.loads(response["body"].read())

        content = ""
        if result.get("content"):
            content = result["content"][0].get("text", "")

        return {
            "content": content,
            "model": model,
            "provider": self.provider_name,
            "usage": result.get("usage", {}),
            "finish_reason": result.get("stop_reason"),
        }

    async def achat(self, messages, **kwargs):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.chat(messages, **kwargs))

    def embed(self, texts, model=None, **kwargs):
        model = model or "amazon.titan-embed-text-v1"
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            body = {"inputText": text}
            response = self.client.invoke_model(modelId=model, body=json.dumps(body))
            result = json.loads(response["body"].read())
            embeddings.append(result.get("embedding", []))

        return embeddings[0] if len(embeddings) == 1 else embeddings

    async def aembed(self, texts, **kwargs):
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.embed(texts, **kwargs))
