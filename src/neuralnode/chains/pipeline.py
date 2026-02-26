"""Chains and Pipelines for composable LLM workflows."""

from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

from ..core.ai import NeuralNode
from ..core.types import LLMResponse

T = TypeVar("T")
U = TypeVar("U")


class ChainComponent(ABC, Generic[T, U]):
    """Abstract base for chain components."""
    
    @abstractmethod
    def run(self, input_data: T) -> U:
        """Execute this component."""
        pass
    
    @abstractmethod
    async def arun(self, input_data: T) -> U:
        """Async execute this component."""
        pass
    
    def __or__(self, other: "ChainComponent[U, V]") -> "Chain[T, V]":
        """Allow chaining with | operator."""
        return Chain([self, other])


@dataclass
class ChainContext:
    """Context passed through chain execution."""
    data: Any
    metadata: Dict[str, Any]
    history: List[Dict[str, Any]]
    
    def __init__(self, data: Any = None):
        self.data = data
        self.metadata = {}
        self.history = []
    
    def add_step(self, name: str, input_data: Any, output_data: Any):
        """Record a step in history."""
        self.history.append({
            "step": name,
            "input": input_data,
            "output": output_data,
        })


class PromptTemplate(ChainComponent[str, str]):
    """Template for generating prompts."""
    
    def __init__(
        self,
        template: str,
        input_variables: Optional[List[str]] = None,
    ):
        self.template = template
        self.input_variables = input_variables or []
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} in template")
    
    def run(self, input_data: Union[str, Dict[str, Any]]) -> str:
        """Apply template to input."""
        if isinstance(input_data, dict):
            return self.format(**input_data)
        elif isinstance(input_data, str):
            # If single string, use as default variable or directly
            if len(self.input_variables) == 1:
                return self.format(**{self.input_variables[0]: input_data})
            return self.format(text=input_data)
        return str(input_data)
    
    async def arun(self, input_data: Union[str, Dict[str, Any]]) -> str:
        return self.run(input_data)
    
    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        """Create from template string, auto-extract variables."""
        import re
        variables = re.findall(r'\{(\w+)\}', template)
        return cls(template=template, input_variables=variables)


class LLMChain(ChainComponent[str, LLMResponse]):
    """Chain component that calls an LLM."""
    
    def __init__(
        self,
        llm: NeuralNode,
        prompt_template: Optional[PromptTemplate] = None,
        system_prompt: Optional[str] = None,
        **llm_kwargs: Any,
    ):
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt
        self.llm_kwargs = llm_kwargs
    
    def run(self, input_data: str) -> LLMResponse:
        """Run LLM on input."""
        if self.prompt_template:
            prompt = self.prompt_template.run(input_data)
        else:
            prompt = input_data
        
        return self.llm.chat(
            prompt,
            system=self.system_prompt,
            **self.llm_kwargs,
        )
    
    async def arun(self, input_data: str) -> LLMResponse:
        if self.prompt_template:
            prompt = self.prompt_template.run(input_data)
        else:
            prompt = input_data
        
        return await self.llm.achat(
            prompt,
            system=self.system_prompt,
            **self.llm_kwargs,
        )


class TransformChain(ChainComponent[T, U]):
    """Chain component that transforms data."""
    
    def __init__(
        self,
        transform: Callable[[T], U],
        name: str = "transform",
    ):
        self.transform = transform
        self.name = name
    
    def run(self, input_data: T) -> U:
        return self.transform(input_data)
    
    async def arun(self, input_data: T) -> U:
        # Check if transform is async
        if asyncio.iscoroutinefunction(self.transform):
            return await self.transform(input_data)
        return self.transform(input_data)


class ConditionalChain(ChainComponent[T, U]):
    """Branching chain based on condition."""
    
    def __init__(
        self,
        condition: Callable[[T], bool],
        if_true: ChainComponent[T, U],
        if_false: ChainComponent[T, U],
    ):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
    
    def run(self, input_data: T) -> U:
        if self.condition(input_data):
            return self.if_true.run(input_data)
        return self.if_false.run(input_data)
    
    async def arun(self, input_data: T) -> U:
        if self.condition(input_data):
            return await self.if_true.arun(input_data)
        return await self.if_false.arun(input_data)


class ParallelChain(ChainComponent[T, List[U]]):
    """Run multiple chains in parallel."""
    
    def __init__(self, chains: List[ChainComponent[T, U]]):
        self.chains = chains
    
    def run(self, input_data: T) -> List[U]:
        """Run chains sequentially (use arun for parallel)."""
        return [chain.run(input_data) for chain in self.chains]
    
    async def arun(self, input_data: T) -> List[U]:
        """Run chains in parallel."""
        tasks = [chain.arun(input_data) for chain in self.chains]
        return await asyncio.gather(*tasks)


class Chain(Generic[T, U]):
    """A chain of components."""
    
    def __init__(self, components: Optional[List[ChainComponent]] = None):
        self.components = components or []
        self.context = ChainContext()
    
    def add(self, component: ChainComponent) -> "Chain[T, U]":
        """Add a component to the chain."""
        self.components.append(component)
        return self
    
    def run(self, input_data: T) -> U:
        """Execute the chain."""
        result: Any = input_data
        
        for i, component in enumerate(self.components):
            step_name = getattr(component, "name", f"step_{i}")
            input_snapshot = result
            
            result = component.run(result)
            
            self.context.add_step(step_name, input_snapshot, result)
        
        return result
    
    async def arun(self, input_data: T) -> U:
        """Execute the chain asynchronously."""
        result: Any = input_data
        
        for i, component in enumerate(self.components):
            step_name = getattr(component, "name", f"step_{i}")
            input_snapshot = result
            
            result = await component.arun(result)
            
            self.context.add_step(step_name, input_snapshot, result)
        
        return result
    
    def __or__(self, other: ChainComponent) -> "Chain[T, U]":
        """Allow chaining with | operator."""
        self.components.append(other)
        return self


class Pipeline:
    """High-level pipeline for complex workflows."""
    
    def __init__(self, llm: Optional[NeuralNode] = None):
        self.llm = llm
        self.steps: List[Dict[str, Any]] = []
    
    def add_prompt(
        self,
        template: str,
        name: str = "prompt",
    ) -> "Pipeline":
        """Add a prompt template step."""
        self.steps.append({
            "name": name,
            "type": "prompt",
            "component": PromptTemplate.from_template(template),
        })
        return self
    
    def add_llm(
        self,
        name: str = "llm",
        system_prompt: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> "Pipeline":
        """Add an LLM call step."""
        if not self.llm:
            raise ValueError("Pipeline needs an LLM to add LLM steps")
        
        self.steps.append({
            "name": name,
            "type": "llm",
            "component": LLMChain(
                llm=self.llm,
                system_prompt=system_prompt,
                **llm_kwargs,
            ),
        })
        return self
    
    def add_transform(
        self,
        transform: Callable[[Any], Any],
        name: str = "transform",
    ) -> "Pipeline":
        """Add a transformation step."""
        self.steps.append({
            "name": name,
            "type": "transform",
            "component": TransformChain(transform, name=name),
        })
        return self
    
    def add_parser(
        self,
        parser: Any,  # OutputParser
        name: str = "parse",
    ) -> "Pipeline":
        """Add an output parsing step."""
        def parse_fn(data):
            if isinstance(data, LLMResponse):
                return parser.parse(data.text)
            return parser.parse(str(data))
        
        return self.add_transform(parse_fn, name=name)
    
    def build(self) -> Chain:
        """Build the chain from pipeline steps."""
        components = [step["component"] for step in self.steps]
        return Chain(components)
    
    def run(self, input_data: Any) -> Any:
        """Run the pipeline."""
        chain = self.build()
        return chain.run(input_data)
    
    async def arun(self, input_data: Any) -> Any:
        """Run the pipeline asynchronously."""
        chain = self.build()
        return await chain.arun(input_data)


# Convenience functions
def create_chain(*components: ChainComponent) -> Chain:
    """Create a chain from components."""
    return Chain(list(components))


def prompt_chain(template: str, llm: NeuralNode, system: Optional[str] = None) -> Chain:
    """Create a simple prompt -> LLM chain."""
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(llm=llm, prompt_template=prompt, system_prompt=system)
    return Chain([prompt, llm_chain])
