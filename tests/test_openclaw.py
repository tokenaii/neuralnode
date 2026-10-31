import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, "src")

from neuralnode.integrations.openclaw import OpenClawClient, OpenClawServer
import neuralnode as nn


class TestOpenClaw(unittest.TestCase):

    def test_client_init(self):
        """Test that OpenClawClient initializes with correct parameters."""
        client = OpenClawClient(gateway_url="http://127.0.0.1:18789", api_key="test-key")
        self.assertEqual(client.gateway_url, "http://127.0.0.1:18789")
        self.assertEqual(client.headers.get("Authorization"), "Bearer test-key")
        self.assertEqual(client.api_url, "http://127.0.0.1:18789/v1")

    def test_create_session_documents_implicit_gateway_session(self):
        client = OpenClawClient()

        session = client.create_session("conversation-123")

        self.assertEqual(session["sessionId"], "conversation-123")
        self.assertEqual(session["status"], "pending_first_message")

    @patch("neuralnode.integrations.openclaw.requests.post")
    def test_send_task_uses_openai_compatible_gateway_endpoint(self, post):
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
        post.return_value = response
        client = OpenClawClient("http://127.0.0.1:18789", api_key="gateway-token")

        result = client.send_task("conversation-123", "Say hello")

        self.assertEqual(result["choices"][0]["message"]["content"], "hello")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:18789/v1/chat/completions")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "openclaw/default",
                "user": "conversation-123",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer gateway-token")

    def test_session_status_explains_gateway_ws_requirement(self):
        client = OpenClawClient()

        with self.assertRaisesRegex(NotImplementedError, "WebSocket"):
            client.get_session_status("conversation-123")

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
