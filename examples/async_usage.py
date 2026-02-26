"""
Example: Async Usage

This example demonstrates how to use NeuralNode
with async/await for concurrent operations.
"""

import asyncio
from neuralnode import NeuralNode


async def main():
    # Initialize LLM
    ai = NeuralNode(provider="auto")
    
    # Example 1: Basic async chat
    print("=== Example 1: Async Chat ===")
    
    response = await ai.achat("What is machine learning?")
    print(f"Response: {response.text}")
    
    # Example 2: Concurrent chat requests
    print("\n=== Example 2: Concurrent Requests ===")
    
    questions = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]
    
    # Run all requests concurrently
    tasks = [ai.achat(q) for q in questions]
    responses = await asyncio.gather(*tasks)
    
    for question, response in zip(questions, responses):
        print(f"\nQ: {question}")
        print(f"A: {response.text[:100]}...")
    
    # Example 3: Async with streaming
    print("\n=== Example 3: Async Streaming ===")
    
    stream = await ai.achat("Count from 1 to 10", stream=True)
    print("Response: ", end="", flush=True)
    
    async for chunk in stream:
        print(chunk.text, end="", flush=True)
    print()
    
    # Example 4: Async embeddings
    print("\n=== Example 4: Async Embeddings ===")
    
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "NLP deals with human language",
    ]
    
    embeddings = await ai.aembed(texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    
    # Example 5: Mixed sync/async workflow
    print("\n=== Example 5: Mixed Workflow ===")
    
    # Sync operation
    print("Starting sync operation...")
    sync_response = ai.chat("Quick question: 2+2?", remember=False)
    print(f"Sync result: {sync_response.text}")
    
    # Async operation
    print("Starting async operation...")
    async_response = await ai.achat("Another quick question: 3+3?")
    print(f"Async result: {async_response.text}")
    
    # Example 6: Async with error handling
    print("\n=== Example 6: Error Handling ===")
    
    try:
        response = await ai.achat("Hello!")
        print(f"Success: {response.text}")
    except Exception as e:
        print(f"Error occurred: {e}")
    
    # Example 7: Batch processing with semaphores
    print("\n=== Example 7: Controlled Concurrency ===")
    
    async def process_with_limit(semaphore, text):
        async with semaphore:
            response = await ai.achat(f"Summarize: {text}", remember=False)
            return response.text
    
    # Limit to 2 concurrent requests
    semaphore = asyncio.Semaphore(2)
    texts = [f"Article {i} about technology" for i in range(5)]
    
    tasks = [process_with_limit(semaphore, t) for t in texts]
    results = await asyncio.gather(*tasks)
    
    print(f"Processed {len(results)} items")


if __name__ == "__main__":
    asyncio.run(main())
