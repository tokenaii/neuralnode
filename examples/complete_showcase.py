"""
Example: Complete NeuralNode Showcase

This example demonstrates all major features of NeuralNode in one place.
"""

import asyncio
from pydantic import BaseModel, Field
from typing import List

from neuralnode import NeuralNode
from neuralnode.agents import Agent, Tool, create_tool
from neuralnode.rag import RAG, Document
from neuralnode.local import HardwareDetector
from neuralnode.tools import Calculator


class MovieRecommendation(BaseModel):
    """Movie recommendation with details."""
    title: str
    year: int
    genre: List[str]
    rating: float = Field(ge=0, le=10)
    reason: str


def showcase_basic_usage():
    """Showcase 1: Basic chat with all providers."""
    print("=" * 60)
    print("SHOWCASE 1: Basic Usage")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    print(f"Provider: {ai.provider.__class__.__name__}")
    print(f"Model: {ai.model}")
    print(f"Supports streaming: {ai.supports_streaming}")
    print(f"Supports tools: {ai.supports_tools}")
    
    # Simple chat
    response = ai.chat("What is the capital of France?")
    print(f"\nQ: What is the capital of France?")
    print(f"A: {response.text}")
    
    # With streaming
    print("\nStreaming example:")
    for chunk in ai.chat("Count to 5", stream=True):
        print(chunk.text, end="", flush=True)
    print()


def showcase_memory():
    """Showcase 2: Conversation memory."""
    print("\n" + "=" * 60)
    print("SHOWCASE 2: Conversation Memory")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    
    r1 = ai.chat("My favorite color is blue.")
    print(f"User: My favorite color is blue.")
    print(f"AI: {r1.text}")
    
    r2 = ai.chat("What's my favorite color?")
    print(f"\nUser: What's my favorite color?")
    print(f"AI: {r2.text}")
    
    ai.clear_memory()
    print("\n[Memory cleared]")


def showcase_structured_output():
    """Showcase 3: Structured output."""
    print("\n" + "=" * 60)
    print("SHOWCASE 3: Structured Output")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    
    try:
        movie = ai.structured_output(
            "Recommend a sci-fi movie from the 1980s-1990s",
            output_schema=MovieRecommendation,
        )
        print(f"Title: {movie.title} ({movie.year})")
        print(f"Genre: {', '.join(movie.genre)}")
        print(f"Rating: {movie.rating}/10")
        print(f"Why: {movie.reason}")
    except Exception as e:
        print(f"Error: {e}")


def showcase_agent():
    """Showcase 4: AI Agent with tools."""
    print("\n" + "=" * 60)
    print("SHOWCASE 4: AI Agent")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    
    # Create tools
    calc = Tool(
        name="calculator",
        description="Perform calculations",
        parameters={"expression": {"type": "string"}},
        handler=lambda expression: Calculator().calculate(expression),
    )
    
    @create_tool(
        name="get_time",
        description="Get current time in a city",
        parameters={"city": {"type": "string"}},
    )
    def get_time(city: str) -> str:
        return f"It's 3:00 PM in {city} (mock)"
    
    # Create agent
    agent = Agent(
        llm=ai,
        tools=[calc, get_time],
        verbose=False,
    )
    
    result = agent.run("What is 15 * 23 + 100?")
    print(f"Agent result: {result}")


def showcase_rag():
    """Showcase 5: RAG."""
    print("\n" + "=" * 60)
    print("SHOWCASE 5: RAG (Retrieval-Augmented Generation)")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    
    # Mock embedding function
    def mock_embed(text: str) -> list[float]:
        return [hash(text) % 100 / 100.0] * 10
    
    rag = RAG(llm=ai, embedding_function=mock_embed)
    
    # Add documents
    documents = [
        Document(content="Python is a high-level programming language."),
        Document(content="JavaScript runs in web browsers."),
        Document(content="Rust is a systems programming language."),
    ]
    rag.add_documents(documents)
    
    # Query
    answer = rag.query("What is Python used for?")
    print(f"Q: What is Python used for?")
    print(f"A: {answer}")


def showcase_hardware():
    """Showcase 6: Hardware detection."""
    print("\n" + "=" * 60)
    print("SHOWCASE 6: Hardware Detection")
    print("=" * 60)
    
    hw = HardwareDetector()
    print(hw.get_summary())
    
    config = hw.recommend_model_config()
    print(f"\nRecommended config:")
    print(f"  Device: {config['device']}")
    print(f"  Quantization: {config['quantization']}")
    print(f"  Context length: {config['context_length']}")


async def showcase_async():
    """Showcase 7: Async operations."""
    print("\n" + "=" * 60)
    print("SHOWCASE 7: Async Operations")
    print("=" * 60)
    
    ai = NeuralNode(provider="auto")
    
    # Concurrent requests
    questions = ["What is AI?", "What is ML?", "What is DL?"]
    
    print("Running 3 concurrent requests...")
    tasks = [ai.achat(q) for q in questions]
    responses = await asyncio.gather(*tasks)
    
    for q, r in zip(questions, responses):
        print(f"Q: {q}")
        print(f"A: {r.text[:50]}...")
        print()


def main():
    """Run all showcases."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "NeuralNode Framework Showcase" + " " * 14 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    # Run synchronous showcases
    showcase_basic_usage()
    showcase_memory()
    showcase_structured_output()
    showcase_agent()
    showcase_rag()
    showcase_hardware()
    
    # Run async showcase
    asyncio.run(showcase_async())
    
    print("\n" + "=" * 60)
    print("Showcase complete! NeuralNode is ready to use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
