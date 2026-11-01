import asyncio
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, "src")

from neuralnode.integrations.openclaw import OpenClawClient, OpenClawServer


def _request_asgi(app, method, path, *, headers=None, body=None):
    request_body = json.dumps(body).encode() if body is not None else b""
    normalized_headers = dict(headers or {})
    if body is not None and not any(key.lower() == "content-type" for key in normalized_headers):
        normalized_headers["Content-Type"] = "application/json"
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in normalized_headers.items()
    ]
    sent = []
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": request_body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": request_headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    start = next(message for message in sent if message["type"] == "http.response.start")
    response_headers = {key.decode(): value.decode() for key, value in start.get("headers", [])}
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    ).decode()
    return SimpleNamespace(
        status_code=start["status"],
        headers=response_headers,
        text=response_body,
        json=lambda: json.loads(response_body),
    )


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

    def test_non_loopback_server_requires_api_key(self):
        class MockNode:
            model = "test-model"

        with self.assertRaisesRegex(ValueError, "api_key is required"):
            OpenClawServer(node=MockNode(), host="0.0.0.0")

    def test_server_rejects_empty_api_keys_and_wildcard_cors(self):
        class MockNode:
            model = "test-model"

        with self.assertRaisesRegex(ValueError, "api_key must be a non-empty string"):
            OpenClawServer(node=MockNode(), api_key="  ")
        with self.assertRaisesRegex(ValueError, "explicit origins"):
            OpenClawServer(node=MockNode(), cors_origins=["*"])

    def test_manager_passes_its_generated_token_to_server(self):
        from neuralnode.integrations.openclaw import OpenClawManager

        class MockNode:
            model = "test-model"

        manager = OpenClawManager(node=MockNode())
        self.assertEqual(manager.server.api_key, manager.api_key)

    def test_public_server_authenticates_models_and_chat(self):
        class MockNode:
            model = "test-model"

            def chat(self, **kwargs):
                return "safe response"

        server = OpenClawServer(node=MockNode(), host="0.0.0.0", api_key="server-token")
        self.assertEqual(_request_asgi(server.app, "GET", "/v1/models").status_code, 401)
        self.assertEqual(
            _request_asgi(
                server.app,
                "POST",
                "/v1/chat/completions",
                body={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
            ).status_code,
            401,
        )
        headers = {"Authorization": "Bearer server-token"}
        self.assertEqual(
            _request_asgi(server.app, "GET", "/v1/models", headers=headers).status_code, 200
        )
        response = _request_asgi(
            server.app,
            "POST",
            "/v1/chat/completions",
            headers=headers,
            body={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "safe response")

    def test_openclaw_server_does_not_reflect_arbitrary_cors_origins(self):
        class MockNode:
            model = "test-model"

        response = _request_asgi(
            OpenClawServer(node=MockNode()).app,
            "OPTIONS",
            "/v1/models",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertNotEqual(
            response.headers.get("access-control-allow-origin"), "https://attacker.example"
        )

    def test_chat_errors_do_not_disclose_exception_details(self):
        class MockNode:
            model = "test-model"

            def chat(self, **kwargs):
                raise RuntimeError("provider secret response")

        response = _request_asgi(
            OpenClawServer(node=MockNode()).app,
            "POST",
            "/v1/chat/completions",
            body={"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]},
        )

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("provider secret response", response.text)

    def test_streaming_errors_do_not_disclose_exception_details(self):
        class MockNode:
            model = "test-model"

            def chat(self, **kwargs):
                raise RuntimeError("private provider detail")

        server = OpenClawServer(node=MockNode())
        result = "".join(server._stream_completion("Hi", None, [], "test-model"))
        self.assertIn("The request could not be completed.", result)
        self.assertNotIn("private provider detail", result)


if __name__ == "__main__":
    unittest.main()
