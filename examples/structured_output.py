"""
Example: Structured Output with Pydantic

This example demonstrates how to get structured,
type-safe responses from LLMs using Pydantic models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from neuralnode import NeuralNode


# Define output schemas
class Person(BaseModel):
    """Person information."""
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years")
    occupation: str = Field(description="Person's job or profession")
    skills: List[str] = Field(description="List of skills")


class ProductReview(BaseModel):
    """Product review analysis."""
    product_name: str
    rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
    pros: List[str]
    cons: List[str]
    summary: str


class CodeAnalysis(BaseModel):
    """Code analysis result."""
    language: str
    complexity: str = Field(description="low, medium, or high")
    issues: List[str]
    suggestions: List[str]


class MeetingNotes(BaseModel):
    """Meeting notes extraction."""
    attendees: List[str]
    date: str
    topics: List[str]
    action_items: List[dict] = Field(
        default_factory=list,
        description="Action items with assignee and deadline"
    )
    decisions: List[str]


def main():
    # Initialize LLM
    ai = NeuralNode(provider="auto")
    
    # Example 1: Extract person information
    print("=== Example 1: Extract Person Info ===")
    
    text = """
    John Smith is a 35-year-old software engineer based in San Francisco.
    He has expertise in Python, machine learning, and cloud architecture.
    Previously, he worked as a data scientist at Google.
    """
    
    try:
        person = ai.structured_output(
            f"Extract person information from this text: {text}",
            output_schema=Person,
        )
        print(f"Name: {person.name}")
        print(f"Age: {person.age}")
        print(f"Occupation: {person.occupation}")
        print(f"Skills: {', '.join(person.skills)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Analyze product review
    print("\n=== Example 2: Product Review Analysis ===")
    
    review = """
    The iPhone 15 Pro is an excellent smartphone with a stunning camera
    and great battery life. However, it's quite expensive and the
    learning curve for new features can be steep. Overall, very satisfied!
    """
    
    try:
        analysis = ai.structured_output(
            f"Analyze this product review: {review}",
            output_schema=ProductReview,
        )
        print(f"Product: {analysis.product_name}")
        print(f"Rating: {analysis.rating}/5")
        print(f"Pros: {', '.join(analysis.pros)}")
        print(f"Cons: {', '.join(analysis.cons)}")
        print(f"Summary: {analysis.summary}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Code analysis
    print("\n=== Example 3: Code Analysis ===")
    
    code = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    
    print(fibonacci(30))
    """
    
    try:
        analysis = ai.structured_output(
            f"Analyze this code:\n```\n{code}\n```",
            output_schema=CodeAnalysis,
        )
        print(f"Language: {analysis.language}")
        print(f"Complexity: {analysis.complexity}")
        print(f"Issues: {analysis.issues}")
        print(f"Suggestions: {analysis.suggestions}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Async structured output
    print("\n=== Example 4: Async Structured Output ===")
    print("""
# To use async:
import asyncio

async def main():
    result = await ai.astructured_output(
        "Extract info from: Alice is 28, works as a designer",
        output_schema=Person,
    )
    print(result)

asyncio.run(main())
""")


if __name__ == "__main__":
    main()
