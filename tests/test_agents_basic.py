import unittest
import sys
sys.path.insert(0, "src")
from neuralnode.agents import SimpleAgent, ReActAgent, ThinkingAgent
from neuralnode.core import Message

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

if __name__ == "__main__":
    unittest.main()
