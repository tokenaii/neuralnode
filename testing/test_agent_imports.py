"""
Quick test to verify all agent imports work correctly
"""

def test_agent_imports():
    """Test that all agent classes and functions can be imported."""
    try:
        # Test imports from agents module
        from neuralnode.agents import (
            # Dataclasses
            AgentAction,
            AgentFinish,
            AgentStep,
            AgentExecutorOutput,
            ReasoningStep,
            ReasoningStepType,
            # Core classes
            AgentExecutor,
            BaseAgent,
            # Agent types
            SimpleAgent,
            ReActAgent,
            ChainOfThoughtAgent,
            ReflectiveAgent,
            ToolCallingAgent,
            OpenAIFunctionsAgent,
            StructuredChatAgent,
            # Builder
            AgentBuilder,
            # Factory functions
            create_agent,
            create_react_agent,
            create_openai_functions_agent,
            create_structured_chat_agent,
            create_tool_calling_agent,
        )
        print("✅ All agent imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_main_imports():
    """Test that main neuralnode imports work."""
    try:
        import neuralnode as nn
        print(f"✅ NeuralNode imported successfully (version: {nn.__version__})")
        
        # Test agent classes from main module
        agent_classes = [
            'AgentAction', 'AgentFinish', 'AgentStep', 'AgentExecutorOutput',
            'BaseAgent', 'AgentExecutor',
            'ReActAgent', 'OpenAIFunctionsAgent', 'StructuredChatAgent', 'ToolCallingAgent',
            'SimpleAgent', 'ChainOfThoughtAgent', 'ReflectiveAgent',
            'AgentBuilder', 'create_agent',
            'create_react_agent', 'create_openai_functions_agent',
            'create_structured_chat_agent', 'create_tool_calling_agent',
        ]
        
        missing = []
        for cls_name in agent_classes:
            if not hasattr(nn, cls_name):
                missing.append(cls_name)
        
        if missing:
            print(f"❌ Missing from neuralnode namespace: {missing}")
            return False
        else:
            print("✅ All agent classes available in neuralnode namespace")
            return True
            
    except Exception as e:
        print(f"❌ NeuralNode import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NeuralNode Agent Imports")
    print("=" * 60)
    
    success1 = test_agent_imports()
    print()
    success2 = test_main_imports()
    
    print()
    print("=" * 60)
    if success1 and success2:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)
