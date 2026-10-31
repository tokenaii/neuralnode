import unittest
import sys

sys.path.insert(0, "src")
from neuralnode.providers import get_available_providers, is_local_provider, ProviderRegistry
from neuralnode.providers.base import BaseLLMProvider


class DummyProvider(BaseLLMProvider):
    provider_name = "dummy"
    provider_type = None
    default_model = "dummy-1"

    @property
    def capabilities(self):
        return {"tools": False, "streaming": False, "embeddings": True}

    def chat(self, messages, **kwargs):
        return {"content": "ok", "model": self.model, "provider": "dummy"}

    def achat(self, messages, **kwargs):
        return {"content": "aok", "model": self.model, "provider": "dummy"}

    def embed(self, texts, **kwargs):
        return [[0.0, 0.1]]

    async def aembed(self, texts, **kwargs):
        return [[0.0, 0.1]]


class TestProvidersMocked(unittest.TestCase):
    def test_registry_has_core_providers(self):
        providers = get_available_providers()
        for name in ["openai", "anthropic", "google", "groq", "ollama", "horus"]:
            self.assertIn(name, providers)

    def test_register_and_create_dummy(self):
        ProviderRegistry.register("dummy-provider", DummyProvider)
        inst = ProviderRegistry.create_provider("dummy-provider", cache=False)
        self.assertEqual(inst.chat([{"role": "user", "content": "x"}]).get("content"), "ok")

    def test_lazy_info_no_instance(self):
        info = ProviderRegistry.get_provider_info("openai")
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)

    def test_is_local(self):
        self.assertTrue(is_local_provider("ollama"))
        self.assertFalse(is_local_provider("openai"))

    def test_openai_compatible_provider_serializes_image_messages(self):
        from neuralnode.core import Message, MessageRole
        from neuralnode.providers.chat.openai import OpenAICompatibleProvider

        payload = OpenAICompatibleProvider._payload_messages(
            [
                Message(
                    role=MessageRole.USER,
                    content="Describe it",
                    metadata={"images": ["YWJj"]},
                )
            ]
        )

        self.assertEqual(payload[0]["role"], "user")
        self.assertEqual(payload[0]["content"][0]["text"], "Describe it")
        self.assertEqual(
            payload[0]["content"][1]["image_url"]["url"], "data:image/jpeg;base64,YWJj"
        )

    def test_google_provider_formats_image_metadata(self):
        from neuralnode.core import Message, MessageRole
        from neuralnode.providers.chat.google import GoogleChatProvider

        formatted = GoogleChatProvider._format_messages(
            None,
            [
                Message(
                    role=MessageRole.USER,
                    content="Describe",
                    metadata={"images": ["YWJj"], "image_media_type": "image/png"},
                )
            ],
        )

        self.assertEqual(formatted[0]["parts"][1]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(formatted[0]["parts"][1]["inline_data"]["data"], "YWJj")

    def test_anthropic_provider_formats_image_metadata(self):
        from neuralnode.core import Message, MessageRole
        from neuralnode.providers.chat.anthropic import AnthropicChatProvider

        _, formatted = AnthropicChatProvider._prepare_messages(
            None,
            [
                Message(
                    role=MessageRole.USER,
                    content="Describe",
                    metadata={"images": ["YWJj"], "image_media_type": "image/png"},
                )
            ],
        )

        self.assertEqual(formatted[0]["content"][1]["source"]["type"], "base64")
        self.assertEqual(formatted[0]["content"][1]["source"]["media_type"], "image/png")


if __name__ == "__main__":
    unittest.main()
