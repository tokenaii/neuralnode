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

if __name__ == "__main__":
    unittest.main()
