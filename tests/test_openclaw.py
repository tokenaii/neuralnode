import unittest
from neuralnode.integrations.openclaw import OpenClawClient, OpenClawServer
import neuralnode as nn

class TestOpenClaw(unittest.TestCase):
    
    def test_client_init(self):
        """Test that OpenClawClient initializes with correct parameters."""
        client = OpenClawClient(gateway_url="http://127.0.0.1:18789", api_key="test-key")
        self.assertEqual(client.gateway_url, "http://127.0.0.1:18789")
        self.assertEqual(client.headers.get("Authorization"), "Bearer test-key")

    def test_server_init(self):
        """Test that OpenClawServer can be initialized with a mock node."""
        class MockNode:
            model = "test-model"
        
        node = MockNode()
        server = OpenClawServer(node=node, port=18788)
        self.assertEqual(server.port, 18788)
        self.assertEqual(server.node.model, "test-model")

if __name__ == "__main__":
    unittest.main()
