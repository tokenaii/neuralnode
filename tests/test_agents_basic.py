import unittest
import sys
sys.path.insert(0, "src")
from neuralnode.agents import SimpleAgent, ReActAgent, ThinkingAgent
from neuralnode.core import BaseTool, Message, ParameterSpec, ToolResult

class DummyLLM:
    provider_name = "dummy"
    provider_type = None
    @property
    def capabilities(self):
        return {"tools": False, "streaming": False, "embeddings": False}
    def chat(self, messages, tools=None, stream=False, **kw):
        return "hello from dummy"
    def supports_tools(self):
        return False
    def supports_streaming(self):
        return False
    def supports_embeddings(self):
        return False
    async def achat(self, *a, **k):
        return "async"
    def embed(self, texts):
        return [[0.0]]
    async def aembed(self, texts):
        return [[0.0]]


class AddTool(BaseTool):
    name = "add_numbers"
    description = "Add two numbers."
    parameters = {
        "a": ParameterSpec("number", "First number"),
        "b": ParameterSpec("number", "Second number"),
    }

    def execute(self, a, b):
        return ToolResult.ok(str(a + b))


class TextToolCallLLM(DummyLLM):
    def __init__(self):
        self.calls = 0

    def supports_tools(self):
        return True

    def chat(self, messages, tools=None, stream=False, **kw):
        self.calls += 1
        if self.calls == 1:
            return 'ACTION: add_numbers\nPARAMETERS: {"a": 19, "b": 23}'
        assert any(message.role.value == "tool" and message.content == "42" for message in messages)
        return "42"

class TestAgentsBasic(unittest.TestCase):
    def test_simple_agent_run(self):
        llm = DummyLLM()
        agent = SimpleAgent(llm, tools=[], system_prompt="sys")
        out = agent.run("hi")
        self.assertIsInstance(out, str)
        self.assertIn("hello", out)

    def test_thinking_agent_wraps(self):
        llm = DummyLLM()
        base = SimpleAgent(llm, tools=[])
        ta = ThinkingAgent(base)
        out = ta.run("hi")
        self.assertIsInstance(out, str)

    def test_native_tool_provider_text_action_fallback_executes_tool(self):
        llm = TextToolCallLLM()
        agent = SimpleAgent(llm, tools=[AddTool()])

        self.assertEqual(agent.run("What is 19 plus 23?"), "42")
        self.assertEqual(llm.calls, 2)

if __name__ == "__main__":
    unittest.main()
