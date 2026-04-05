"""Test agent system."""

import pytest
from unittest.mock import MagicMock, patch

import neuralnode as nn
from neuralnode.core import Message, LLMResponse, ToolResult, ToolCall


class TestSimpleAgent:
    """Test SimpleAgent."""
    
    def test_init(self, mock_provider):
        """Test agent initialization."""
        agent = nn.SimpleAgent(llm=mock_provider)
        assert agent.llm == mock_provider
        assert agent.max_iterations == 10
    
    def test_init_with_tools(self, mock_provider, mock_tool):
        """Test initialization with tools."""
        agent = nn.SimpleAgent(llm=mock_provider, tools=[mock_tool])
        assert len(agent.tools) == 1
        assert "mock_tool" in agent.tools
    
    def test_register_tool(self, mock_provider):
        """Test tool registration."""
        agent = nn.SimpleAgent(llm=mock_provider)
        tool = MagicMock()
        tool.name = "new_tool"
        
        agent.register_tool(tool)
        assert "new_tool" in agent.tools
    
    def test_run_without_tools(self, mock_provider):
        """Test running without tool calls."""
        mock_provider.chat.return_value = LLMResponse(
            content="Direct response",
            model="mock",
            provider="mock"
        )
        
        agent = nn.SimpleAgent(llm=mock_provider)
        result = agent.run("Hello")
        
        assert result == "Direct response"
    
    def test_run_with_tool_call(self, mock_provider):
        """Test running with tool execution."""
        # First response triggers tool
        mock_provider.chat.side_effect = [
            LLMResponse(
                content="ACTION: search_tool\nPARAMETERS: {\"query\": \"test\"}",
                model="mock",
                provider="mock"
            ),
            # Second response after tool execution
            LLMResponse(
                content="Final answer",
                model="mock",
                provider="mock"
            )
        ]
        
        tool = MagicMock()
        tool.name = "search_tool"
        tool.execute.return_value = ToolResult.ok(content="Tool result")
        
        agent = nn.SimpleAgent(llm=mock_provider, tools=[tool])
        result = agent.run("Search for something")
        
        assert "Final answer" in result
        tool.execute.assert_called_once()


class TestReActAgent:
    """Test ReActAgent."""
    
    def test_init(self, mock_provider):
        """Test ReActAgent initialization."""
        agent = nn.ReActAgent(llm=mock_provider)
        assert isinstance(agent, nn.BaseAgent)
    
    def test_run_basic(self, mock_provider):
        """Test basic ReAct run."""
        mock_provider.chat.return_value = LLMResponse(
            content="Thought: I need to answer\nAction: None\nFinal Answer: Hello!",
            model="mock",
            provider="mock"
        )
        
        agent = nn.ReActAgent(llm=mock_provider)
        result = agent.run("Say hello")
        
        assert "Hello!" in result
    
    def test_run_with_tool(self, mock_provider):
        """Test ReAct with tool use."""
        responses = [
            # First: decide to use tool
            LLMResponse(
                content="Thought: Need to search\nAction: search\nParameters: {\"q\": \"test\"}",
                model="mock",
                provider="mock"
            ),
            # Second: final answer
            LLMResponse(
                content="Thought: Got results\nFinal Answer: Here is the answer",
                model="mock",
                provider="mock"
            )
        ]
        mock_provider.chat.side_effect = responses
        
        tool = MagicMock()
        tool.name = "search"
        tool.execute.return_value = ToolResult.ok(content="Search results")
        
        agent = nn.ReActAgent(llm=mock_provider, tools=[tool])
        result = agent.run("Find information")
        
        assert "answer" in result.lower()


class TestChainOfThoughtAgent:
    """Test ChainOfThoughtAgent."""
    
    def test_init(self, mock_provider):
        """Test initialization."""
        agent = nn.ChainOfThoughtAgent(llm=mock_provider)
        assert agent.show_reasoning is True
    
    def test_run(self, mock_provider):
        """Test chain of thought reasoning."""
        mock_provider.chat.return_value = LLMResponse(
            content="""REASONING: Step 1: Understand problem. Step 2: Analyze.
            PLAN: I will break this down.
            FINAL ANSWER: The answer is 42.
            """,
            model="mock",
            provider="mock"
        )
        
        agent = nn.ChainOfThoughtAgent(llm=mock_provider)
        result = agent.run("What is the answer?")
        
        assert "42" in result
    
    def test_get_reasoning_trace(self, mock_provider):
        """Test getting reasoning trace."""
        mock_provider.chat.return_value = LLMResponse(
            content="REASONING: Step 1\nFINAL ANSWER: Done",
            model="mock",
            provider="mock"
        )
        
        agent = nn.ChainOfThoughtAgent(llm=mock_provider)
        agent.run("Test")
        trace = agent.get_reasoning_trace()
        
        assert isinstance(trace, list)
        assert len(trace) > 0


class TestReflectiveAgent:
    """Test ReflectiveAgent."""
    
    def test_init(self, mock_provider):
        """Test initialization."""
        agent = nn.ReflectiveAgent(llm=mock_provider, reflection_rounds=2)
        assert agent.reflection_rounds == 2
    
    def test_run(self, mock_provider):
        """Test reflective reasoning."""
        mock_provider.chat.return_value = LLMResponse(
            content="Improved answer after reflection",
            model="mock",
            provider="mock"
        )
        
        agent = nn.ReflectiveAgent(llm=mock_provider, reflection_rounds=1)
        result = agent.run("Solve this")
        
        assert "answer" in result.lower()
    
    def test_get_reflections(self, mock_provider):
        """Test getting reflection history."""
        mock_provider.chat.return_value = LLMResponse(
            content="Answer",
            model="mock",
            provider="mock"
        )
        
        agent = nn.ReflectiveAgent(llm=mock_provider)
        agent.run("Test")
        reflections = agent.get_reflections()
        
        assert isinstance(reflections, list)


class TestAgentBuilder:
    """Test AgentBuilder."""
    
    def test_build_simple_agent(self, mock_provider, mock_tool):
        """Test building simple agent."""
        agent = nn.AgentBuilder() \
            .with_llm(mock_provider) \
            .with_tools([mock_tool]) \
            .with_system_prompt("You are helpful") \
            .build_simple()
        
        assert isinstance(agent, nn.SimpleAgent)
        assert agent.system_prompt == "You are helpful"
    
    def test_build_react_agent(self, mock_provider):
        """Test building ReAct agent."""
        agent = nn.AgentBuilder() \
            .with_llm(mock_provider) \
            .build_react()
        
        assert isinstance(agent, nn.ReActAgent)
    
    def test_builder_missing_llm(self):
        """Test builder fails without LLM."""
        with pytest.raises(ValueError):
            nn.AgentBuilder().build_simple()


class TestUniversalReasoner:
    """Test UniversalReasoner."""
    
    def test_init(self, mock_provider):
        """Test initialization."""
        reasoner = nn.UniversalReasoner(llm=mock_provider)
        assert reasoner.llm == mock_provider
        assert reasoner.enable_self_correction is True
    
    def test_reason(self, mock_provider):
        """Test basic reasoning."""
        mock_provider.chat.return_value = LLMResponse(
            content="REASONING: Step 1\nFINAL ANSWER: Answer is 5",
            model="mock",
            provider="mock"
        )
        
        reasoner = nn.UniversalReasoner(llm=mock_provider)
        result = reasoner.reason("What is 2+3?")
        
        assert isinstance(result, nn.ReasoningResult)
        assert result.conclusion is not None
    
    def test_compare_strategies(self, mock_provider):
        """Test comparing reasoning strategies."""
        mock_provider.chat.return_value = LLMResponse(
            content="FINAL ANSWER: Test",
            model="mock",
            provider="mock"
        )
        
        reasoner = nn.UniversalReasoner(llm=mock_provider)
        results = reasoner.compare_strategies("Query")
        
        assert isinstance(results, dict)
        assert len(results) > 0


class TestNeuralNodeMain:
    """Test main NeuralNode interface."""
    
    def test_init_auto_provider(self):
        """Test initialization with auto provider selection."""
        with patch('neuralnode.auto_select_provider') as mock_auto:
            mock_provider = MagicMock()
            mock_auto.return_value = mock_provider
            
            nn_interface = nn.NeuralNode(provider="auto")
            assert nn_interface._llm is not None
    
    def test_chat(self, mock_provider):
        """Test chat method."""
        nn_interface = nn.NeuralNode.__new__(nn.NeuralNode)
        nn_interface._llm = mock_provider
        nn_interface._memory = None
        
        mock_provider.chat.return_value = LLMResponse(
            content="Hello!",
            model="mock",
            provider="mock"
        )
        
        response = nn_interface.chat("Hi")
        assert response == "Hello!"
    
    def test_create_agent(self, mock_provider):
        """Test creating agent from interface."""
        nn_interface = nn.NeuralNode.__new__(nn.NeuralNode)
        nn_interface._llm = mock_provider
        nn_interface._tools = []
        nn_interface._system_tools = []
        
        agent = nn_interface.agent()
        assert isinstance(agent, nn.BaseAgent)
    
    def test_get_capabilities(self, mock_provider):
        """Test getting capabilities."""
        nn_interface = nn.NeuralNode.__new__(nn.NeuralNode)
        nn_interface._llm = mock_provider
        mock_provider.capabilities = {"tools": True, "streaming": True}
        
        caps = nn_interface.get_capabilities()
        assert caps["tools"] is True
