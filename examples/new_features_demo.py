"""
Example: All New NeuralNode Features

This example demonstrates all the new features added to NeuralNode:
- Chat Templates (Llama3, Qwen, Mistral, etc.)
- Caching Layer
- Output Parsers
- Chains / Pipelines
- Document Loaders (CSV, Markdown, HTML, JSON)
- Rate Limiting & Retries
- The 'nn' shortcut alias
"""

# NEW: Use 'nn' as shortcut alias
import nn
from nn import NeuralNode, Agent, Chain, Pipeline, PromptTemplate
from pydantic import BaseModel, Field
from typing import List


def demo_chat_templates():
    """Demo 1: Chat Templates for different models."""
    print("=" * 60)
    print("DEMO 1: Chat Templates")
    print("=" * 60)
    
    from nn.utils.chat_templates import (
        apply_chat_template,
        get_stop_sequences,
        LLAMA3_TEMPLATE,
        QWEN_TEMPLATE,
    )
    from nn.core.types import Message, MessageRole
    
    # Create sample messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello, how are you?"),
    ]
    
    # Apply Llama 3 template
    llama3_prompt = apply_chat_template(messages, "llama3")
    print("Llama 3 Template:")
    print(llama3_prompt[:200] + "...")
    print()
    
    # Apply Qwen template
    qwen_prompt = apply_chat_template(messages, "qwen")
    print("Qwen Template:")
    print(qwen_prompt[:200] + "...")
    print()
    
    # Get stop sequences
    stop_seqs = get_stop_sequences("llama3")
    print(f"Llama 3 stop sequences: {stop_seqs}")


def demo_caching():
    """Demo 2: Caching Layer."""
    print("\n" + "=" * 60)
    print("DEMO 2: Caching Layer")
    print("=" * 60)
    
    from nn.utils.cache import InMemoryCache, DiskCache, cached, get_default_cache
    from nn.core.types import LLMResponse, ProviderType, TokenUsage
    
    # In-memory cache
    cache = InMemoryCache(max_size=100)
    
    # Create a mock response
    response = LLMResponse(
        text="Hello world",
        model="gpt-4",
        provider=ProviderType.OPENAI,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    
    # Store in cache
    cache.set("test_key", response, ttl=3600)  # 1 hour TTL
    print("Stored response in cache")
    
    # Retrieve from cache
    cached_response = cache.get("test_key")
    if cached_response:
        print(f"Retrieved from cache: {cached_response.text}")
    
    # Check cache stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Disk cache example
    print("\nDiskCache can persist across sessions")
    print("cache_dir = '.nn_cache'  # Stores in this folder")


def demo_output_parsers():
    """Demo 3: Output Parsers."""
    print("\n" + "=" * 60)
    print("DEMO 3: Output Parsers")
    print("=" * 60)
    
    from nn.utils.output_parsers import (
        JSONOutputParser,
        ListOutputParser,
        CommaSeparatedListOutputParser,
        BooleanOutputParser,
        MarkdownOutputParser,
        parse_json,
        parse_list,
    )
    
    # JSON parser
    json_text = '{"name": "Alice", "age": 28}'
    parser = JSONOutputParser()
    result = parser.parse(json_text)
    print(f"JSON parsed: {result}")
    
    # List parser
    list_text = """1. Apple
2. Banana
3. Cherry"""
    list_parser = ListOutputParser()
    items = list_parser.parse(list_text)
    print(f"List parsed: {items}")
    
    # Comma-separated list
    csv_text = "apple, banana, cherry"
    csv_parser = CommaSeparatedListOutputParser()
    fruits = csv_parser.parse(csv_text)
    print(f"CSV parsed: {fruits}")
    
    # Boolean parser
    bool_parser = BooleanOutputParser()
    print(f"'yes' -> {bool_parser.parse('yes')}")
    print(f"'no' -> {bool_parser.parse('no')}")
    
    # Markdown parser
    md_text = """# Title
This is content.

## Section 1
More content here.
"""
    md_parser = MarkdownOutputParser()
    sections = md_parser.parse(md_text)
    print(f"Markdown sections: {list(sections.keys())}")
    
    # Convenience functions
    data = parse_json('{"key": "value"}')
    print(f"parse_json convenience: {data}")


def demo_chains():
    """Demo 4: Chains and Pipelines."""
    print("\n" + "=" * 60)
    print("DEMO 4: Chains and Pipelines")
    print("=" * 60)
    
    from nn import Chain, Pipeline, PromptTemplate, LLMChain, TransformChain, prompt_chain
    
    # Create a simple chain manually
    template = PromptTemplate.from_template("Translate this to French: {text}")
    print(f"Template: {template.format(text='Hello')}")
    
    # Chain using | operator
    ai = nn.NeuralNode(provider="auto")
    
    # Create a chain: prompt -> transform
    chain = (
        PromptTemplate.from_template("Summarize in 3 words: {text}")
        | TransformChain(lambda x: f"Result: {x}" if isinstance(x, str) else str(x))
    )
    
    result = chain.run("Artificial Intelligence is changing the world rapidly")
    print(f"Chain result: {result}")
    
    # Pipeline builder
    print("\nPipeline example:")
    pipe = Pipeline(llm=ai)
    pipe.add_prompt("Analyze: {topic}")
    pipe.add_llm(system_prompt="You are an analyst.")
    pipe.add_transform(lambda x: x.text.upper() if hasattr(x, 'text') else str(x).upper())
    
    # Note: Won't actually run without API key
    print("Pipeline built with 3 steps: prompt -> llm -> transform")
    
    # Convenience function
    print("\nConvenience: prompt_chain(template, llm)")


def demo_document_loaders():
    """Demo 5: Extended Document Loaders."""
    print("\n" + "=" * 60)
    print("DEMO 5: Extended Document Loaders")
    print("=" * 60)
    
    from nn.rag import CSVLoader, MarkdownLoader, HTMLLoader, JSONLoader, DirectoryLoader
    
    # CSV Loader
    print("CSVLoader:")
    print("  loader = CSVLoader('data.csv', content_columns=['text'])")
    print("  docs = loader.load()")
    
    # Markdown Loader
    print("\nMarkdownLoader:")
    print("  loader = MarkdownLoader('readme.md', split_headers=True)")
    print("  docs = loader.load()  # Splits by headers")
    
    # HTML Loader
    print("\nHTMLLoader:")
    print("  loader = HTMLLoader('page.html', extract_links=True)")
    print("  docs = loader.load()  # Extracts text from HTML")
    
    # JSON Loader
    print("\nJSONLoader:")
    print("  loader = JSONLoader('data.json', content_key='text')")
    print("  docs = loader.load()  # One doc per JSON item")
    
    # Directory Loader
    print("\nDirectoryLoader:")
    print("  loader = DirectoryLoader('./docs', glob_pattern='**/*.md')")
    print("  docs = loader.load()  # Loads all matching files")


def demo_rate_limiting():
    """Demo 6: Rate Limiting & Retries."""
    print("\n" + "=" * 60)
    print("DEMO 6: Rate Limiting & Retries")
    print("=" * 60)
    
    from nn.utils.retry import (
        with_retry,
        with_rate_limit,
        RetryConfig,
        RateLimiter,
        AdaptiveRateLimiter,
    )
    
    # Retry decorator
    print("Retry decorator:")
    print("""
    @with_retry(max_attempts=3, min_wait=1, max_wait=10)
    def call_api():
        return requests.get("...")
""")
    
    # Rate limit decorator
    print("Rate limit decorator:")
    print("""
    @with_rate_limit(max_requests=60, time_window=60)
    def call_api():
        return requests.get("...")
""")
    
    # Adaptive rate limiter
    limiter = AdaptiveRateLimiter(initial_rate=100)
    print(f"\nAdaptiveRateLimiter: current_rate={limiter.current_rate}")
    print("  - Increases rate on success")
    print("  - Decreases rate on 429 errors")


def demo_nn_alias():
    """Demo 7: The 'nn' Shortcut Alias."""
    print("\n" + "=" * 60)
    print("DEMO 7: The 'nn' Shortcut Alias")
    print("=" * 60)
    
    print("Instead of:")
    print("  from neuralnode import NeuralNode, Agent, RAG")
    print()
    print("You can now use:")
    print("  import nn")
    print("  ai = nn.NeuralNode(provider='auto')")
    print()
    print("Or:")
    print("  from nn import NeuralNode, Agent, RAG, Chain")
    print()
    print("All exports are identical between 'neuralnode' and 'nn'")


def main():
    """Run all demos."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + " " * 15 + "NeuralNode New Features" + " " * 24 + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    demo_chat_templates()
    demo_caching()
    demo_output_parsers()
    demo_chains()
    demo_document_loaders()
    demo_rate_limiting()
    demo_nn_alias()
    
    print("\n" + "=" * 60)
    print("All demos complete! NeuralNode is feature-complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
