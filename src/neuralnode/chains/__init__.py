"""
Advanced Chains Module for NeuralNode
Supports LLMChain, SequentialChain, RouterChain, RetrievalQA
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Iterator, Tuple
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChainOutput:
    """Output from a chain run."""

    output: Any
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseChain(ABC):
    """Abstract base class for all chains."""

    def __init__(
        self,
        input_keys: List[str],
        output_keys: List[str],
        verbose: bool = False,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.memory = None

    @abstractmethod
    def run(self, **inputs) -> ChainOutput:
        """Execute the chain with inputs."""
        pass

    @abstractmethod
    async def arun(self, **inputs) -> ChainOutput:
        """Async execute the chain."""
        pass

    def __call__(self, **inputs) -> ChainOutput:
        """Convenience call method."""
        return self.run(**inputs)

    def _validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that all required inputs are provided."""
        missing = set(self.input_keys) - set(inputs.keys())
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        return True

    def _log(self, message: str) -> None:
        """Log message if verbose mode."""
        if self.verbose:
            logger.info(message)

    def _call_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """Call registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class LLMChain(BaseChain):
    """Basic chain that calls an LLM with a prompt."""

    def __init__(
        self,
        llm: Any,  # LLM provider
        prompt: Any,  # Prompt template
        output_parser: Optional[Callable] = None,
        **kwargs,
    ):
        self.llm = llm
        self.prompt = prompt
        self.output_parser = output_parser or (lambda x: x)

        # Extract input/output keys from prompt
        input_keys = getattr(prompt, "input_variables", [])
        output_keys = ["output"]

        super().__init__(input_keys, output_keys, **kwargs)

    def run(self, **inputs) -> ChainOutput:
        """Execute LLM chain."""
        self._validate_inputs(inputs)
        self._call_callbacks("chain_start", {"chain": "LLMChain", "inputs": inputs})

        # Format prompt
        if hasattr(self.prompt, "format_prompt"):
            prompt_value = self.prompt.format_prompt(**inputs)
            messages = prompt_value.to_messages()
        else:
            prompt_text = self.prompt.format(**inputs)
            messages = [{"role": "user", "content": prompt_text}]

        self._log(f"Prompt: {messages}")

        # Call LLM
        response = self.llm.chat(messages)

        if isinstance(response, dict):
            output_text = response.get("content", "")
        else:
            output_text = str(response)

        self._log(f"Response: {output_text[:100]}...")

        # Parse output
        parsed_output = self.output_parser(output_text)

        self._call_callbacks("chain_end", {"output": parsed_output})

        return ChainOutput(
            output=parsed_output,
            intermediate_steps=[{"prompt": messages, "response": output_text}],
            metadata={"model": getattr(self.llm, "model", "unknown")},
        )

    async def arun(self, **inputs) -> ChainOutput:
        """Async execute LLM chain."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**inputs))

    def stream(self, **inputs) -> Iterator[str]:
        """Stream LLM output."""
        self._validate_inputs(inputs)

        # Format prompt
        if hasattr(self.prompt, "format_prompt"):
            prompt_value = self.prompt.format_prompt(**inputs)
            messages = prompt_value.to_messages()
        else:
            prompt_text = self.prompt.format(**inputs)
            messages = [{"role": "user", "content": prompt_text}]

        # Stream from LLM
        if hasattr(self.llm, "chat"):
            response = self.llm.chat(messages, stream=True)
            for chunk in response:
                content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
                yield content
        else:
            result = self.run(**inputs)
            yield result.output


class SequentialChain(BaseChain):
    """Chain that executes multiple chains in sequence."""

    def __init__(
        self,
        chains: List[BaseChain],
        input_variables: List[str],
        output_variables: Optional[List[str]] = None,
        return_intermediate: bool = False,
        **kwargs,
    ):
        self.chains = chains
        self.return_intermediate = return_intermediate

        # Validate chain compatibility
        self._validate_chains()

        output_keys = output_variables or chains[-1].output_keys if chains else []

        super().__init__(input_variables, output_keys, **kwargs)

    def _validate_chains(self) -> bool:
        """Validate that chains can be connected."""
        for i in range(len(self.chains) - 1):
            current_outputs = set(self.chains[i].output_keys)
            next_inputs = set(self.chains[i + 1].input_keys)

            # Check that next chain can get its inputs
            if not next_inputs.issubset(current_outputs | self.input_keys):
                missing = next_inputs - current_outputs - set(self.input_keys)
                raise ValueError(f"Chain {i+1} missing inputs from chain {i}: {missing}")

        return True

    def run(self, **inputs) -> ChainOutput:
        """Execute chains in sequence."""
        self._validate_inputs(inputs)
        self._call_callbacks("chain_start", {"chain": "SequentialChain", "inputs": inputs})

        current_inputs = inputs.copy()
        intermediate_steps = []
        all_outputs = {}

        for i, chain in enumerate(self.chains):
            self._log(f"Running chain {i+1}/{len(self.chains)}")

            # Run chain
            result = chain.run(**current_inputs)

            # Store intermediate results
            intermediate_steps.append(
                {
                    "chain_index": i,
                    "chain_type": type(chain).__name__,
                    "inputs": current_inputs.copy(),
                    "output": result.output,
                }
            )

            # Update inputs for next chain
            if isinstance(result.output, dict):
                current_inputs.update(result.output)
                all_outputs.update(result.output)
            else:
                # Single output - use first output key
                if chain.output_keys:
                    current_inputs[chain.output_keys[0]] = result.output
                    all_outputs[chain.output_keys[0]] = result.output

            self._call_callbacks("chain_progress", {"chain_index": i, "output": result.output})

        # Determine final output
        if self.return_intermediate:
            final_output = all_outputs
        else:
            final_output = {k: all_outputs.get(k) for k in self.output_keys}

        self._call_callbacks("chain_end", {"output": final_output})

        return ChainOutput(output=final_output, intermediate_steps=intermediate_steps)

    async def arun(self, **inputs) -> ChainOutput:
        """Async execute sequential chain."""
        import asyncio

        self._validate_inputs(inputs)
        current_inputs = inputs.copy()
        intermediate_steps = []

        for i, chain in enumerate(self.chains):
            if hasattr(chain, "arun"):
                result = await chain.arun(**current_inputs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: chain.run(**current_inputs))

            intermediate_steps.append(
                {"chain_index": i, "chain_type": type(chain).__name__, "output": result.output}
            )

            if isinstance(result.output, dict):
                current_inputs.update(result.output)
            else:
                if chain.output_keys:
                    current_inputs[chain.output_keys[0]] = result.output

        return ChainOutput(output=current_inputs, intermediate_steps=intermediate_steps)


class RouterChain(BaseChain):
    """Chain that routes to different destination chains based on input."""

    def __init__(
        self,
        router_chain: BaseChain,
        destination_chains: Dict[str, BaseChain],
        default_chain: Optional[BaseChain] = None,
        **kwargs,
    ):
        self.router_chain = router_chain
        self.destination_chains = destination_chains
        self.default_chain = default_chain

        # Collect all possible input keys
        all_inputs = set(router_chain.input_keys)
        for chain in destination_chains.values():
            all_inputs.update(chain.input_keys)

        # Output keys from all destinations
        all_outputs = set()
        for chain in destination_chains.values():
            all_outputs.update(chain.output_keys)

        super().__init__(list(all_inputs), list(all_outputs), **kwargs)

    def run(self, **inputs) -> ChainOutput:
        """Route to appropriate chain."""
        self._call_callbacks("chain_start", {"chain": "RouterChain", "inputs": inputs})

        # Get routing decision
        route_result = self.router_chain.run(**inputs)
        destination = route_result.output

        self._log(f"Routing to: {destination}")

        # Select chain
        if isinstance(destination, str) and destination in self.destination_chains:
            selected_chain = self.destination_chains[destination]
        elif self.default_chain:
            selected_chain = self.default_chain
            destination = "default"
        else:
            raise ValueError(f"No destination chain for: {destination}")

        # Execute selected chain
        chain_result = selected_chain.run(**inputs)

        self._call_callbacks(
            "chain_end", {"destination": destination, "output": chain_result.output}
        )

        return ChainOutput(
            output=chain_result.output,
            intermediate_steps=[
                {"route": destination, "router_output": route_result.output},
                {"chain_result": chain_result.output},
            ],
            metadata={"destination": destination},
        )

    async def arun(self, **inputs) -> ChainOutput:
        """Async execute router chain."""
        import asyncio

        # Get routing decision
        if hasattr(self.router_chain, "arun"):
            route_result = await self.router_chain.arun(**inputs)
        else:
            loop = asyncio.get_event_loop()
            route_result = await loop.run_in_executor(None, lambda: self.router_chain.run(**inputs))

        destination = route_result.output

        # Select and execute chain
        if isinstance(destination, str) and destination in self.destination_chains:
            selected_chain = self.destination_chains[destination]
        elif self.default_chain:
            selected_chain = self.default_chain
        else:
            raise ValueError(f"No destination chain for: {destination}")

        if hasattr(selected_chain, "arun"):
            chain_result = await selected_chain.arun(**inputs)
        else:
            loop = asyncio.get_event_loop()
            chain_result = await loop.run_in_executor(None, lambda: selected_chain.run(**inputs))

        return ChainOutput(
            output=chain_result.output,
            intermediate_steps=[{"route": destination}, {"chain_result": chain_result.output}],
        )


class RetrievalQAChain(BaseChain):
    """Chain for Retrieval-Augmented Generation (RAG)."""

    def __init__(
        self,
        llm: Any,
        retriever: Any,  # Vector store retriever
        prompt: Optional[Any] = None,
        return_source_documents: bool = False,
        top_k: int = 4,
        **kwargs,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.return_source_documents = return_source_documents
        self.top_k = top_k

        input_keys = ["query"]
        output_keys = ["answer"]
        if return_source_documents:
            output_keys.append("source_documents")

        super().__init__(input_keys, output_keys, **kwargs)

    def run(self, **inputs) -> ChainOutput:
        """Execute RAG chain."""
        query = inputs.get("query") or inputs.get("question")
        if not query:
            raise ValueError("Query required (use 'query' or 'question' key)")

        self._call_callbacks("chain_start", {"chain": "RetrievalQA", "query": query})
        self._log(f"Retrieving documents for: {query}")

        # Retrieve documents
        try:
            if hasattr(self.retriever, "search"):
                docs = self.retriever.search(query, k=self.top_k)
            elif hasattr(self.retriever, "similarity_search"):
                docs = self.retriever.similarity_search(query, k=self.top_k)
            elif hasattr(self.retriever, "retrieve"):
                docs = self.retriever.retrieve(query, top_k=self.top_k)
            else:
                docs = []
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            docs = []

        self._log(f"Retrieved {len(docs)} documents")

        # Build context from documents
        context = self._build_context(docs)

        # Build prompt with context
        if self.prompt:
            if hasattr(self.prompt, "format_prompt"):
                prompt_value = self.prompt.format_prompt(context=context, question=query)
                messages = prompt_value.to_messages()
            else:
                prompt_text = self.prompt.format(context=context, question=query)
                messages = [{"role": "user", "content": prompt_text}]
        else:
            # Default RAG prompt
            messages = [
                {
                    "role": "user",
                    "content": f"Based on the following context:\n\n{context}\n\nAnswer this question: {query}",
                }
            ]

        # Call LLM
        self._log("Generating answer...")
        response = self.llm.chat(messages)

        if isinstance(response, dict):
            answer = response.get("content", "")
        else:
            answer = str(response)

        self._call_callbacks("chain_end", {"answer": answer})

        # Build output
        output = {"answer": answer}
        if self.return_source_documents:
            output["source_documents"] = docs

        return ChainOutput(
            output=output,
            intermediate_steps=[
                {
                    "retrieved_docs": len(docs),
                    "context": context[:500] + "..." if len(context) > 500 else context,
                }
            ],
            metadata={"retrieved_count": len(docs)},
        )

    def _build_context(self, docs: List[Any]) -> str:
        """Build context string from documents."""
        context_parts = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                content = doc.get("content", doc.get("text", ""))
            elif hasattr(doc, "content"):
                content = doc.content
            elif hasattr(doc, "page_content"):
                content = doc.page_content
            else:
                content = str(doc)

            context_parts.append(f"[Document {i+1}]\n{content}")

        return "\n\n".join(context_parts)

    async def arun(self, **inputs) -> ChainOutput:
        """Async execute RAG chain."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**inputs))


class TransformChain(BaseChain):
    """Chain that applies a transformation function."""

    def __init__(
        self,
        transform_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        input_variables: List[str],
        output_variables: List[str],
        **kwargs,
    ):
        self.transform_func = transform_func

        super().__init__(input_variables, output_variables, **kwargs)

    def run(self, **inputs) -> ChainOutput:
        """Apply transformation."""
        self._validate_inputs(inputs)

        result = self.transform_func(inputs)

        return ChainOutput(output=result, intermediate_steps=[{"input": inputs, "output": result}])

    async def arun(self, **inputs) -> ChainOutput:
        """Async apply transformation."""
        return self.run(**inputs)


class APIChain(BaseChain):
    """Chain that calls an external API."""

    def __init__(
        self,
        api_url: str,
        api_method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params_template: Optional[Dict[str, str]] = None,
        body_template: Optional[Dict[str, str]] = None,
        response_key: str = "api_response",
        **kwargs,
    ):
        self.api_url = api_url
        self.api_method = api_method
        self.headers = headers or {}
        self.params_template = params_template
        self.body_template = body_template
        self.response_key = response_key

        # Extract input keys from templates
        all_inputs = set()
        if params_template:
            for v in params_template.values():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    all_inputs.add(v[1:-1])
        if body_template:
            for v in body_template.values():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    all_inputs.add(v[1:-1])

        super().__init__(list(all_inputs), [response_key], **kwargs)

    def run(self, **inputs) -> ChainOutput:
        """Call API."""
        import requests

        self._validate_inputs(inputs)

        # Build URL with params
        url = self.api_url
        params = {}
        if self.params_template:
            for k, v in self.params_template.items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    params[k] = inputs.get(v[1:-1], "")
                else:
                    params[k] = v

        # Build body
        body = None
        if self.body_template:
            body = {}
            for k, v in self.body_template.items():
                if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
                    body[k] = inputs.get(v[1:-1], "")
                else:
                    body[k] = v

        # Make request
        if self.api_method.upper() == "GET":
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
        elif self.api_method.upper() == "POST":
            response = requests.post(url, headers=self.headers, json=body, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {self.api_method}")

        response.raise_for_status()
        data = response.json()

        return ChainOutput(
            output={self.response_key: data},
            intermediate_steps=[{"url": url, "params": params, "status": response.status_code}],
        )

    async def arun(self, **inputs) -> ChainOutput:
        """Async call API."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.run(**inputs))


# Convenience functions
def create_llm_chain(llm: Any, prompt: Any, **kwargs) -> LLMChain:
    """Create an LLM chain."""
    return LLMChain(llm=llm, prompt=prompt, **kwargs)


def create_sequential_chain(
    chains: List[BaseChain], input_variables: List[str], **kwargs
) -> SequentialChain:
    """Create a sequential chain."""
    return SequentialChain(chains=chains, input_variables=input_variables, **kwargs)


def create_retrieval_qa_chain(llm: Any, retriever: Any, **kwargs) -> RetrievalQAChain:
    """Create a Retrieval QA chain."""
    return RetrievalQAChain(llm=llm, retriever=retriever, **kwargs)


__all__ = [
    # Base
    "BaseChain",
    "ChainOutput",
    # Chains
    "LLMChain",
    "SequentialChain",
    "RouterChain",
    "RetrievalQAChain",
    "TransformChain",
    "APIChain",
    # Convenience functions
    "create_llm_chain",
    "create_sequential_chain",
    "create_retrieval_qa_chain",
]
