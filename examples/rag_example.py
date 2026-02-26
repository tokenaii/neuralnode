"""
Example: RAG (Retrieval-Augmented Generation)

This example demonstrates how to use RAG for
question answering over documents.
"""

from neuralnode import NeuralNode
from neuralnode.rag import RAG, Document, FileLoader, TextSplitter


def main():
    # Initialize LLM
    ai = NeuralNode(provider="auto")
    
    # Example 1: Basic RAG with in-memory documents
    print("=== Example 1: Basic RAG ===")
    
    # Create a simple embedding function (in production, use a proper embedding model)
    def simple_embed(text: str) -> list[float]:
        # This is a mock embedding - replace with real embeddings
        import hashlib
        hash_val = hashlib.md5(text.encode()).hexdigest()
        return [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
    
    # Create RAG system
    rag = RAG(llm=ai, embedding_function=simple_embed)
    
    # Add documents
    documents = [
        Document(
            content="NeuralNode is a Python framework for building AI applications. "
                    "It supports multiple LLM providers and provides a unified API.",
            metadata={"source": "docs/intro.md"},
        ),
        Document(
            content="The NeuralNode Agent system allows you to create autonomous AI agents "
                    "that can use tools and make decisions.",
            metadata={"source": "docs/agents.md"},
        ),
        Document(
            content="RAG (Retrieval-Augmented Generation) enables LLMs to answer questions "
                    "based on your documents and knowledge base.",
            metadata={"source": "docs/rag.md"},
        ),
    ]
    
    rag.add_documents(documents, split=True)
    
    # Query the RAG system
    questions = [
        "What is NeuralNode?",
        "How do agents work?",
        "What is RAG used for?",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.query(question)
        print(f"A: {answer}")
    
    # Example 2: RAG with text files
    print("\n=== Example 2: Loading Text Files ===")
    print("""
# To load text files:
# doc = FileLoader.load_text("path/to/file.txt")
# rag.add_documents([doc])
""")
    
    # Example 3: RAG with custom text splitting
    print("\n=== Example 3: Custom Text Splitting ===")
    
    splitter = TextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separator=". ",
    )
    
    long_doc = Document(
        content=". ".join([
            f"This is paragraph {i} of a long document about machine learning. "
            f"Machine learning is a subset of artificial intelligence. "
            f"It enables computers to learn from data without explicit programming."
            for i in range(10)
        ]),
        metadata={"source": "long_doc.txt"},
    )
    
    chunks = splitter.split_documents([long_doc])
    print(f"Split document into {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i}: {chunk.content[:100]}...")
    
    # Example 4: Advanced RAG with metadata filtering
    print("\n=== Example 4: Working with Metadata ===")
    
    doc_with_meta = Document(
        content="Important information about the project roadmap.",
        metadata={
            "source": "roadmap.md",
            "author": "Team Lead",
            "date": "2024-01-15",
            "category": "planning",
        },
    )
    
    print(f"Document metadata: {doc_with_meta.metadata}")


if __name__ == "__main__":
    main()
