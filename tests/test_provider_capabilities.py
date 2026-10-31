import unittest

from neuralnode.providers.chat.openai import OpenAICompatibleProvider
from neuralnode.core import BaseTool, Message, MessageRole, ParameterSpec, ToolResult


class EchoTool(BaseTool):
    name = "echo"
    description = "Return input text."
    parameters = {"text": ParameterSpec("string", "Text to return")}

    def execute(self, text):
        return ToolResult.ok(text)


class TestProviderCapabilities(unittest.TestCase):
    def test_dataclass_capabilities_are_used_by_base_helpers(self):
        provider = OpenAICompatibleProvider(
            api_key="test-key",
            base_url="http://127.0.0.1:8001/v1",
            model="test-model",
        )

        self.assertTrue(provider.supports_tools())
        self.assertTrue(provider.supports_embeddings())
        self.assertFalse(provider.supports_streaming())

    def test_tools_are_serialized_to_function_schemas(self):
        schema = OpenAICompatibleProvider._payload_tools([EchoTool()])[0]

        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "echo")

    def test_existing_tool_schemas_pass_through(self):
        schema = {"type": "function", "function": {"name": "echo"}}

        self.assertEqual(OpenAICompatibleProvider._payload_tools([schema]), [schema])

    def test_openai_tool_calls_are_normalized(self):
        tool_calls = [
            {
                "id": "call-1",
                "function": {"name": "echo", "arguments": '{"text":"hello"}'},
            }
        ]

        self.assertEqual(
            OpenAICompatibleProvider._normalize_tool_calls(tool_calls),
            [{"name": "echo", "arguments": {"text": "hello"}, "call_id": "call-1"}],
        )

    def test_messages_preserve_native_tool_call_protocol(self):
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                content="",
                metadata={
                    "tool_calls": [
                        {"name": "echo", "arguments": {"text": "hello"}, "call_id": "call-1"}
                    ]
                },
            ),
            Message(
                role=MessageRole.TOOL,
                content="hello",
                metadata={"tool_name": "echo", "tool_call_id": "call-1"},
            ),
        ]

        self.assertEqual(
            OpenAICompatibleProvider._payload_messages(messages),
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "echo", "arguments": '{"text": "hello"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "hello", "tool_call_id": "call-1", "name": "echo"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
