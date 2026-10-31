import json
import unittest
from unittest.mock import Mock, patch

import neuralnode as nn
from neuralnode.core import Message


class TestHorusCyberNanoProvider(unittest.TestCase):
    @staticmethod
    def _successful_response():
        response = Mock()
        response.ok = True
        response.headers = {
            "X-Request-ID": "req-test",
            "X-RateLimit-Limit": "20",
            "X-RateLimit-Remaining": "19",
        }
        response.json.return_value = {
            "id": "chatcmpl-test",
            "request_id": "req-test",
            "conversation_id": "8ddedee4-594d-48f2-a72e-a0d942ce27fe",
            "model": "horus-cyber-nano-1.0-beta",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Defensive review complete."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        }
        return response

    def test_provider_is_registered(self):
        self.assertIn("horus-cyber-nano", nn.get_available_providers())
        self.assertIn("horus-cyber-nano-beta", nn.get_available_providers())

    def test_provider_reports_hosted_context_window(self):
        provider = nn.HorusCyberNanoProvider(api_key="horusNN-ABCDEFGHIJKLMNOP")
        self.assertEqual(provider.capabilities.context_window, 32768)
        self.assertEqual(provider.model_type, "text-to-text")
        self.assertFalse(provider.capabilities.vision)

    def test_access_token_is_required(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "Access Token is required"),
        ):
            nn.HorusCyberNanoProvider()

    def test_provider_reads_canonical_environment_variables(self):
        with patch.dict(
            "os.environ",
            {
                "HORUS_CYBER_NANO_API_KEY": "canonical-token",
                "HORUS_CYBER_NANO_BASE_URL": "https://canonical.example/v1",
            },
            clear=True,
        ):
            provider = nn.HorusCyberNanoProvider()

        self.assertEqual(provider.api_key, "canonical-token")
        self.assertEqual(provider.base_url, "https://canonical.example/v1")

    @patch("requests.Session.post")
    def test_chat_uses_early_access_gateway(self, post):
        post.return_value = self._successful_response()

        provider = nn.HorusCyberNanoProvider(
            api_key="horusNN-ABCDEFGHIJKLMNOP",
            base_url="https://example.test/v1",
        )
        result = provider.chat([Message.user("Review this flow")])

        self.assertEqual(result["content"], "Defensive review complete.")
        self.assertEqual(result["metadata"]["request_id"], "req-test")
        request = post.call_args
        self.assertEqual(request.args[0], "https://example.test/v1/chat/completions")
        self.assertEqual(request.kwargs["json"]["messages"][0]["role"], "user")
        self.assertEqual(
            request.kwargs["headers"]["Authorization"],
            "Bearer horusNN-ABCDEFGHIJKLMNOP",
        )
        self.assertFalse(request.kwargs["stream"])
        self.assertEqual(result["metadata"]["attempts"], 1)

    @patch("time.sleep")
    @patch("requests.Session.post")
    def test_chat_retries_temporary_service_errors(self, post, sleep):
        unavailable = Mock()
        unavailable.ok = False
        unavailable.status_code = 503
        unavailable.headers = {"Retry-After": "0", "X-Request-ID": "req-retry"}
        unavailable.json.return_value = {
            "error": {"code": "inference_queue_full", "message": "Try again."}
        }
        post.side_effect = [unavailable, self._successful_response()]

        provider = nn.HorusCyberNanoProvider(
            api_key="horusNN-ABCDEFGHIJKLMNOP",
            base_url="https://example.test/v1",
        )
        result = provider.chat([Message.user("Review this flow")])

        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["metadata"]["attempts"], 2)
        unavailable.close.assert_called_once()
        sleep.assert_called_once_with(0.0)

    @patch("requests.Session.post")
    def test_chat_does_not_retry_disabled_access(self, post):
        forbidden = Mock()
        forbidden.ok = False
        forbidden.status_code = 403
        forbidden.headers = {"X-Request-ID": "req-forbidden"}
        forbidden.json.return_value = {"error": {"message": "API access is disabled."}}
        post.return_value = forbidden
        provider = nn.HorusCyberNanoProvider(api_key="horusNN-ABCDEFGHIJKLMNOP")

        with self.assertRaises(nn.HorusCyberNanoAPIError) as raised:
            provider.chat([Message.user("Review this flow")])

        self.assertEqual(raised.exception.status_code, 403)
        self.assertEqual(post.call_count, 1)

    @patch("requests.Session.post")
    def test_chat_passes_tools_and_returns_tool_calls(self, post):
        response = self._successful_response()
        response.json.return_value["choices"][0] = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {"name": "get_status", "arguments": '{"target":"api"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
        post.return_value = response
        provider = nn.HorusCyberNanoProvider(
            api_key="horusNN-ABCDEFGHIJKLMNOP",
            base_url="https://example.test/v1",
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_status",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        result = provider.chat([Message.user("Check status")], tools=tools)

        self.assertEqual(result["finish_reason"], "tool_calls")
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "get_status")
        self.assertEqual(post.call_args.kwargs["json"]["tools"], tools)
        self.assertEqual(post.call_args.kwargs["json"]["tool_choice"], "auto")

    @patch("requests.Session.post")
    def test_streaming_sse_chunks(self, post):
        response = Mock()
        response.ok = True
        response.headers = {"X-Request-ID": "req-stream"}
        response.iter_lines.return_value = [
            "data: "
            + json.dumps(
                {
                    "model": "horus-cyber-nano-1.0-beta",
                    "choices": [
                        {
                            "delta": {"role": "assistant", "content": "Ready"},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            "data: "
            + json.dumps(
                {
                    "conversation_id": "8ddedee4-594d-48f2-a72e-a0d942ce27fe",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                }
            ),
            "data: [DONE]",
        ]
        post.return_value = response
        provider = nn.HorusCyberNanoProvider(
            api_key="horusNN-ABCDEFGHIJKLMNOP",
            base_url="https://example.test/v1",
        )

        chunks = list(provider.chat([Message.user("Stream")], stream=True))

        self.assertEqual(chunks[0]["content"], "Ready")
        self.assertEqual(chunks[-1]["finish_reason"], "stop")
        self.assertTrue(post.call_args.kwargs["stream"])
        response.close.assert_called_once()

    @patch("requests.Session.post")
    def test_neuralnode_chat_uses_provider(self, post):
        post.return_value = self._successful_response()
        client = nn.NeuralNode(
            provider="horus-cyber-nano",
            model="horus-cyber-nano-1.0-beta",
            api_key="horusNN-ABCDEFGHIJKLMNOP",
            base_url="https://example.test/v1",
        )

        answer = client.chat("Review this authentication flow")

        self.assertEqual(answer, "Defensive review complete.")
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
