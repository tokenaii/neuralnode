"""
Example: AI Agent with Tools

This example demonstrates how to create an AI agent
that can use tools to accomplish tasks.
"""

from neuralnode import NeuralNode
from neuralnode.agents import Agent, Tool, create_tool
from neuralnode.tools import Calculator, WebSearch


def main():
    # Initialize LLM
    ai = NeuralNode(provider="auto")
    
    # Example 1: Agent with calculator tool
    print("=== Example 1: Agent with Calculator ===")
    
    calculator = Tool(
        name="calculator",
        description="Perform mathematical calculations",
        parameters={
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
            }
        },
        handler=lambda expression: Calculator().calculate(expression),
    )
    
    agent1 = Agent(
        llm=ai,
        tools=[calculator],
        verbose=True,
    )
    
    result = agent1.run("What is 1234 * 5678 + 90?")
    print(f"Result: {result}")
    
    # Example 2: Agent with multiple tools
    print("\n=== Example 2: Agent with Multiple Tools ===")
    
    # Create search tool
    search_tool = Tool(
        name="web_search",
        description="Search the web for information",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
            }
        },
        handler=lambda query: WebSearch().search(query, num_results=3),
    )
    
    # Create custom tool using decorator
    @create_tool(
        name="get_weather",
        description="Get weather for a city",
        parameters={
            "city": {"type": "string", "description": "City name"},
        },
    )
    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny and 72°F (mock data)"
    
    agent2 = Agent(
        llm=ai,
        tools=[calculator, search_tool, get_weather],
        system="You are a helpful assistant with access to tools.",
        verbose=True,
    )
    
    result = agent2.run(
        "Calculate 15% of $850 and also tell me the weather in New York"
    )
    print(f"Result: {result}")
    
    # Example 3: Tool using decorator pattern
    print("\n=== Example 3: Tool Decorator Pattern ===")
    
    @create_tool(
        name="count_words",
        description="Count words in text",
        parameters={
            "text": {"type": "string", "description": "Text to analyze"},
        },
    )
    def count_words(text: str) -> int:
        return len(text.split())
    
    @create_tool(
        name="reverse_string",
        description="Reverse a string",
        parameters={
            "text": {"type": "string", "description": "Text to reverse"},
        },
    )
    def reverse_string(text: str) -> str:
        return text[::-1]
    
    agent3 = Agent(
        llm=ai,
        tools=[count_words, reverse_string],
        verbose=True,
    )
    
    result = agent3.run(
        "Count the words in 'Hello world this is NeuralNode' and reverse 'Python'"
    )
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
