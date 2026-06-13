import unittest
import os
from pathlib import Path
import neuralnode as nn
import nn as nn_short

class TestReleaseReadiness(unittest.TestCase):
    
    def test_version_bump(self):
        """Verify version is updated to 2.2.0 in the packages."""
        self.assertEqual(nn.__version__, "2.2.0")
        self.assertEqual(nn_short.__version__, "2.2.0")

    def test_provider_registration(self):
        """Verify key providers are registered in the registry."""
        providers = nn.get_available_providers()
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)
        self.assertIn("google", providers)
        self.assertIn("groq", providers)
        self.assertIn("ollama", providers)
        self.assertIn("horus", providers)

    def test_integration_imports(self):
        """Verify integrations can be successfully imported."""
        from neuralnode.integrations.telegram import TelegramBot
        from neuralnode.integrations.discord import DiscordBot
        from neuralnode.integrations.slack import SlackBot
        from neuralnode.integrations.whatsapp import EvolutionWhatsAppBot, TwilioWhatsAppBot
        from neuralnode.integrations.openclaw import OpenClawManager, OpenClawServer, OpenClawClient

        self.assertIsNotNone(TelegramBot)
        self.assertIsNotNone(DiscordBot)
        self.assertIsNotNone(SlackBot)
        self.assertIsNotNone(EvolutionWhatsAppBot)
        self.assertIsNotNone(TwilioWhatsAppBot)
        self.assertIsNotNone(OpenClawManager)

    def test_aembed_implementation(self):
        """Verify aembed is defined in the OpenAI Compatible provider class."""
        from neuralnode.providers.chat.openai import OpenAICompatibleProvider
        self.assertTrue(hasattr(OpenAICompatibleProvider, "aembed"))
        self.assertTrue(callable(getattr(OpenAICompatibleProvider, "aembed")))

    def test_whatsapp_gateway_path(self):
        """Verify that the WhatsApp evolution-api directory exists in the correct subfolder."""
        integrations_path = Path(nn.__file__).parent / "integrations"
        gateway_path = integrations_path / "whatsapp" / "evolution-api"
        self.assertTrue(gateway_path.exists(), f"WhatsApp Gateway path not found at: {gateway_path}")
        self.assertTrue(gateway_path.is_dir())

    def test_alias_package_proxies(self):
        """Verify that the short package 'nn' correctly exports NeuralNode main classes."""
        self.assertTrue(hasattr(nn_short, "NeuralNode"))
        self.assertTrue(hasattr(nn_short, "Message"))
        self.assertTrue(hasattr(nn_short, "EvolutionWhatsAppBot"))
        self.assertTrue(hasattr(nn_short, "OpenClawManager"))

if __name__ == "__main__":
    unittest.main()
