import os
import unittest
import neuralnode as nn

class TestRefactoring(unittest.TestCase):
    
    def test_provider_imports(self):
        """Test that new providers register correctly and can be imported."""
        providers = nn.get_available_providers()
        print("Available providers:", providers)
        self.assertIn("openai", providers)
        self.assertIn("openrouter", providers)
        self.assertIn("agentrouter", providers)
        self.assertIn("bedrock", providers)
        self.assertIn("vertexai", providers)

    def test_blocker_removed(self):
        """Test that creating an OpenAI provider does not raise blocker errors."""
        try:
            # Should not raise any ValueError about OpenAI blocker
            # Will raise ValueError because of invalid key prefix or missing key,
            # but NOT "OpenAI is blocked in NeuralNode" or similar.
            provider = nn.create_provider("openai", api_key="sk-testkey123")
            self.assertEqual(provider.provider_name, "openai")
        except ValueError as e:
            # Verify it's not the blocker exception
            self.assertNotIn("does not support OpenAI", str(e))
            self.assertNotIn("variable is blocked", str(e))

    def test_whatsapp_evolution_import(self):
        """Test that EvolutionWhatsAppBot can be imported and initialized."""
        from neuralnode.integrations.whatsapp import EvolutionWhatsAppBot
        bot = EvolutionWhatsAppBot(
            api_url="http://localhost:8080",
            global_apikey="my-secret-key"
        )
        self.assertEqual(bot.api_url, "http://localhost:8080")
        self.assertEqual(bot.global_apikey, "my-secret-key")

    def test_rag_loaders(self):
        """Test that CSV and JSON loaders exist and can be instantiated."""
        from neuralnode.rag import CSVLoader, JSONLoader, DocumentLoader
        loader = DocumentLoader()
        self.assertTrue(any(isinstance(l, CSVLoader) for l in loader.loaders))
        self.assertTrue(any(isinstance(l, JSONLoader) for l in loader.loaders))

if __name__ == "__main__":
    unittest.main()
