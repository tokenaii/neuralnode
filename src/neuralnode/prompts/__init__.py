"""
Advanced Prompt System for NeuralNode
Supports Prompt Templates, Few-shot Templates, and Dynamic Prompt Building
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Type, Iterator, Tuple
from dataclasses import dataclass, field
from string import Formatter
import json
import re
import logging

logger = logging.getLogger(__name__)


class PromptValue(ABC):
    """Abstract base class for prompt values."""

    @abstractmethod
    def to_string(self) -> str:
        """Convert to string representation."""
        pass

    @abstractmethod
    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to list of message dicts."""
        pass


@dataclass
class StringPromptValue(PromptValue):
    """Simple string prompt value."""

    text: str

    def to_string(self) -> str:
        return self.text

    def to_messages(self) -> List[Dict[str, str]]:
        return [{"role": "user", "content": self.text}]


@dataclass
class ChatPromptValue(PromptValue):
    """Chat message prompt value."""

    messages: List[Dict[str, str]]

    def to_string(self) -> str:
        """Convert messages to a single string."""
        parts = []
        for msg in self.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(parts)

    def to_messages(self) -> List[Dict[str, str]]:
        return self.messages


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    def __init__(self, input_variables: List[str], **kwargs):
        self.input_variables = input_variables
        self.partial_variables = kwargs.get("partial_variables", {})
        self.output_parser = kwargs.get("output_parser", None)
        self.validate_template = kwargs.get("validate_template", True)

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt with variables."""
        pass

    @abstractmethod
    def format_prompt(self, **kwargs) -> PromptValue:
        """Format the prompt and return a PromptValue."""
        pass

    def partial(self, **kwargs) -> "BasePromptTemplate":
        """Create a new template with partial variables filled."""
        new_partial = {**self.partial_variables, **kwargs}
        new_template = self._copy()
        new_template.partial_variables = new_partial
        return new_template

    @abstractmethod
    def _copy(self) -> "BasePromptTemplate":
        """Create a copy of this template."""
        pass

    def validate(self, input_variables: Optional[List[str]] = None) -> bool:
        """Validate that all required variables are provided."""
        if not self.validate_template:
            return True

        required = set(self.input_variables) - set(self.partial_variables.keys())
        if input_variables:
            provided = set(input_variables)
        else:
            provided = set()

        missing = required - provided
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return True


class PromptTemplate(BasePromptTemplate):
    """A simple prompt template with variable substitution."""

    def __init__(self, template: str, input_variables: Optional[List[str]] = None, **kwargs):
        self.template = template

        # Auto-extract input variables if not provided
        if input_variables is None:
            input_variables = self._extract_variables(template)

        super().__init__(input_variables, **kwargs)

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template string."""
        formatter = Formatter()
        variables = []
        for _, field_name, _, _ in formatter.parse(template):
            if field_name:
                variables.append(field_name)
        return variables

    def format(self, **kwargs) -> str:
        """Format the prompt with variables."""
        # Merge with partial variables
        all_vars = {**self.partial_variables, **kwargs}

        # Validate
        self.validate(list(all_vars.keys()))

        return self.template.format(**all_vars)

    def format_prompt(self, **kwargs) -> PromptValue:
        """Format and return as PromptValue."""
        text = self.format(**kwargs)
        return StringPromptValue(text=text)

    def _copy(self) -> "PromptTemplate":
        """Create a copy of this template."""
        return PromptTemplate(
            template=self.template,
            input_variables=self.input_variables.copy(),
            partial_variables=self.partial_variables.copy(),
            output_parser=self.output_parser,
            validate_template=self.validate_template,
        )

    @classmethod
    def from_file(cls, file_path: str, **kwargs) -> "PromptTemplate":
        """Load template from file."""
        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()
        return cls(template=template, **kwargs)

    @classmethod
    def from_template(cls, template: str, **kwargs) -> "PromptTemplate":
        """Create from template string."""
        return cls(template=template, **kwargs)


class ChatPromptTemplate(BasePromptTemplate):
    """Template for chat-style prompts with multiple messages."""

    def __init__(
        self,
        messages: List[Union[str, Dict[str, Any], BasePromptTemplate]],
        input_variables: Optional[List[str]] = None,
        **kwargs,
    ):
        self.messages = messages

        # Auto-extract input variables
        if input_variables is None:
            input_variables = []
            for msg in messages:
                if isinstance(msg, BasePromptTemplate):
                    input_variables.extend(msg.input_variables)
                elif isinstance(msg, dict) and "template" in msg:
                    input_variables.extend(PromptTemplate._extract_variables(msg["template"]))

        super().__init__(input_variables, **kwargs)

    def format(self, **kwargs) -> str:
        """Format and return as single string."""
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs) -> PromptValue:
        """Format and return as ChatPromptValue."""
        all_vars = {**self.partial_variables, **kwargs}
        self.validate(list(all_vars.keys()))

        formatted_messages = []
        for msg in self.messages:
            if isinstance(msg, BasePromptTemplate):
                formatted = msg.format(**all_vars)
                formatted_messages.append({"role": "user", "content": formatted})
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content_template = msg.get("content", msg.get("template", ""))
                if isinstance(content_template, str):
                    content = content_template.format(**all_vars)
                else:
                    content = content_template.format(**all_vars)
                formatted_messages.append({"role": role, "content": content})
            elif isinstance(msg, str):
                formatted_messages.append({"role": "user", "content": msg.format(**all_vars)})

        return ChatPromptValue(messages=formatted_messages)

    def format_messages(self, **kwargs) -> List[Dict[str, str]]:
        """Format and return list of messages."""
        return self.format_prompt(**kwargs).to_messages()

    def _copy(self) -> "ChatPromptTemplate":
        """Create a copy."""
        return ChatPromptTemplate(
            messages=self.messages.copy(),
            input_variables=self.input_variables.copy(),
            partial_variables=self.partial_variables.copy(),
            output_parser=self.output_parser,
            validate_template=self.validate_template,
        )

    @classmethod
    def from_messages(
        cls, messages: List[Union[str, Dict[str, Any], BasePromptTemplate]], **kwargs
    ) -> "ChatPromptTemplate":
        """Create from list of messages."""
        return cls(messages=messages, **kwargs)

    @classmethod
    def from_system_template(
        cls, system_template: str, human_template: str, **kwargs
    ) -> "ChatPromptTemplate":
        """Create with system and human templates."""
        messages = [
            {"role": "system", "content": system_template},
            {"role": "user", "content": human_template},
        ]
        return cls(messages=messages, **kwargs)


class FewShotPromptTemplate(BasePromptTemplate):
    """Few-shot learning prompt template with examples."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        example_prompt: BasePromptTemplate,
        prefix: str = "",
        suffix: str = "",
        input_variables: Optional[List[str]] = None,
        example_separator: str = "\n\n",
        prefix_variables: Optional[List[str]] = None,
        **kwargs,
    ):
        self.examples = examples
        self.example_prompt = example_prompt
        self.prefix = prefix
        self.suffix = suffix
        self.example_separator = example_separator
        self.prefix_variables = prefix_variables or []

        # Extract variables
        if input_variables is None:
            # Variables needed for suffix (excluding those in examples)
            suffix_vars = PromptTemplate._extract_variables(suffix)
            input_variables = list(set(suffix_vars) - set(example_prompt.input_variables))

        super().__init__(input_variables, **kwargs)

    def format(self, **kwargs) -> str:
        """Format the few-shot prompt."""
        all_vars = {**self.partial_variables, **kwargs}
        self.validate(list(all_vars.keys()))

        # Format prefix
        formatted_prefix = self.prefix.format(
            **{k: v for k, v in all_vars.items() if k in self.prefix_variables}
        )

        # Format examples
        formatted_examples = []
        for example in self.examples:
            formatted_example = self.example_prompt.format(**example)
            formatted_examples.append(formatted_example)

        # Format suffix
        suffix_vars = {k: v for k, v in all_vars.items() if k not in self.prefix_variables}
        formatted_suffix = self.suffix.format(**suffix_vars)

        # Combine
        parts = []
        if formatted_prefix:
            parts.append(formatted_prefix)
        parts.extend(formatted_examples)
        if formatted_suffix:
            parts.append(formatted_suffix)

        return self.example_separator.join(parts)

    def format_prompt(self, **kwargs) -> PromptValue:
        """Format and return as PromptValue."""
        text = self.format(**kwargs)
        return StringPromptValue(text=text)

    def _copy(self) -> "FewShotPromptTemplate":
        """Create a copy."""
        return FewShotPromptTemplate(
            examples=self.examples.copy(),
            example_prompt=self.example_prompt._copy(),
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables.copy(),
            example_separator=self.example_separator,
            prefix_variables=self.prefix_variables.copy(),
            partial_variables=self.partial_variables.copy(),
            output_parser=self.output_parser,
            validate_template=self.validate_template,
        )

    def select_examples(self, input_variables: Dict[str, Any], k: int = 4) -> List[Dict[str, str]]:
        """Select k examples based on input (can be overridden for dynamic selection)."""
        return self.examples[:k]

    @classmethod
    def from_examples(
        cls,
        examples: List[Dict[str, str]],
        suffix: str,
        input_variables: List[str],
        example_prompt: Optional[BasePromptTemplate] = None,
        prefix: str = "",
        **kwargs,
    ) -> "FewShotPromptTemplate":
        """Create from examples."""
        if example_prompt is None:
            # Create default example prompt
            example_keys = list(examples[0].keys()) if examples else []
            template = "\n".join([f"{k}: {{{k}}}" for k in example_keys])
            example_prompt = PromptTemplate(template=template, input_variables=example_keys)

        return cls(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            **kwargs,
        )


class DynamicFewShotPromptTemplate(FewShotPromptTemplate):
    """Dynamic few-shot template with example selection."""

    def __init__(
        self,
        example_selector: "BaseExampleSelector",
        example_prompt: BasePromptTemplate,
        prefix: str = "",
        suffix: str = "",
        input_variables: Optional[List[str]] = None,
        example_separator: str = "\n\n",
        **kwargs,
    ):
        self.example_selector = example_selector

        # Examples are selected dynamically, so we pass empty list
        super().__init__(
            examples=[],
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            **kwargs,
        )

    def format(self, **kwargs) -> str:
        """Format with dynamically selected examples."""
        # Select examples based on input
        selected_examples = self.example_selector.select_examples(kwargs)

        # Temporarily set examples
        original_examples = self.examples
        self.examples = selected_examples

        try:
            result = super().format(**kwargs)
        finally:
            self.examples = original_examples

        return result


class BaseExampleSelector(ABC):
    """Abstract base class for example selection."""

    @abstractmethod
    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select examples based on input."""
        pass

    @abstractmethod
    def add_example(self, example: Dict[str, str]) -> None:
        """Add an example."""
        pass


class LengthBasedExampleSelector(BaseExampleSelector):
    """Select examples based on length constraints."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        example_prompt: BasePromptTemplate,
        max_length: int = 2048,
        get_text_length: Optional[Callable[[str], int]] = None,
    ):
        self.examples = examples
        self.example_prompt = example_prompt
        self.max_length = max_length
        self.get_text_length = get_text_length or len

    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select examples that fit within max length."""
        selected = []
        total_length = 0

        # Calculate input length
        input_str = json.dumps(input_variables)
        input_length = self.get_text_length(input_str)

        remaining = self.max_length - input_length

        for example in self.examples:
            formatted = self.example_prompt.format(**example)
            example_length = self.get_text_length(formatted)

            if total_length + example_length <= remaining:
                selected.append(example)
                total_length += example_length
            else:
                break

        return selected

    def add_example(self, example: Dict[str, str]) -> None:
        """Add an example."""
        self.examples.append(example)


class SemanticSimilarityExampleSelector(BaseExampleSelector):
    """Select examples based on semantic similarity."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        embeddings: Any,  # Embedding provider
        vector_store: Optional[Any] = None,
        k: int = 4,
        example_keys: Optional[List[str]] = None,
    ):
        self.examples = examples
        self.embeddings = embeddings
        self.k = k
        self.example_keys = example_keys or ["input"]

        # Build vector store if provided
        if vector_store:
            self.vector_store = vector_store
            self._index_examples()
        else:
            self.vector_store = None

    def _index_examples(self):
        """Index examples in vector store."""
        texts = []
        for example in self.examples:
            # Combine specified keys
            text = " ".join([str(example.get(k, "")) for k in self.example_keys])
            texts.append(text)

        embeddings = self.embeddings.embed(texts)
        # Add to vector store...

    def select_examples(self, input_variables: Dict[str, Any]) -> List[Dict[str, str]]:
        """Select k most similar examples."""
        # Get embedding for input
        input_text = " ".join([str(input_variables.get(k, "")) for k in self.example_keys])
        input_embedding = self.embeddings.embed([input_text])[0]

        # Find similar examples
        # This would use the vector store for similarity search
        # For now, return first k
        return self.examples[: self.k]

    def add_example(self, example: Dict[str, str]) -> None:
        """Add an example and index it."""
        self.examples.append(example)
        if self.vector_store:
            # Index the new example
            pass


class PipelinePromptTemplate(BasePromptTemplate):
    """Compose multiple prompt templates in a pipeline."""

    def __init__(
        self,
        pipeline_prompts: List[Tuple[str, BasePromptTemplate]],
        final_prompt: BasePromptTemplate,
        **kwargs,
    ):
        self.pipeline_prompts = pipeline_prompts
        self.final_prompt = final_prompt

        # Collect all input variables
        all_vars = set()
        for _, prompt in pipeline_prompts:
            all_vars.update(prompt.input_variables)
        all_vars.update(final_prompt.input_variables)

        super().__init__(list(all_vars), **kwargs)

    def format(self, **kwargs) -> str:
        """Format through the pipeline."""
        all_vars = {**self.partial_variables, **kwargs}

        # Process each step
        intermediate_results = {}
        for name, prompt in self.pipeline_prompts:
            # Get inputs for this step
            step_inputs = {k: v for k, v in all_vars.items() if k in prompt.input_variables}
            # Add intermediate results
            step_inputs.update(
                {k: v for k, v in intermediate_results.items() if k in prompt.input_variables}
            )

            # Format
            result = prompt.format(**step_inputs)
            intermediate_results[name] = result

        # Final prompt
        final_inputs = {**all_vars, **intermediate_results}
        final_inputs = {
            k: v for k, v in final_inputs.items() if k in self.final_prompt.input_variables
        }

        return self.final_prompt.format(**final_inputs)

    def format_prompt(self, **kwargs) -> PromptValue:
        """Format and return as PromptValue."""
        text = self.format(**kwargs)
        return StringPromptValue(text=text)

    def _copy(self) -> "PipelinePromptTemplate":
        """Create a copy."""
        return PipelinePromptTemplate(
            pipeline_prompts=[(n, p._copy()) for n, p in self.pipeline_prompts],
            final_prompt=self.final_prompt._copy(),
            partial_variables=self.partial_variables.copy(),
            output_parser=self.output_parser,
            validate_template=self.validate_template,
        )


class StructuredPromptTemplate(BasePromptTemplate):
    """Template for structured output (JSON, XML, etc.)."""

    def __init__(
        self,
        template: str,
        output_schema: Dict[str, Any],
        input_variables: Optional[List[str]] = None,
        format_instructions_key: str = "format_instructions",
        **kwargs,
    ):
        self.output_schema = output_schema
        self.format_instructions_key = format_instructions_key

        # Generate format instructions
        format_instructions = self._generate_format_instructions()

        # Add format instructions to template
        if f"{{{format_instructions_key}}}" not in template:
            template = f"{template}\n\n{{{format_instructions_key}}}"

        super().__init__(
            template=template,
            input_variables=input_variables or [],
            partial_variables={
                format_instructions_key: format_instructions,
                **kwargs.get("partial_variables", {}),
            },
            **kwargs,
        )

    def _generate_format_instructions(self) -> str:
        """Generate format instructions from schema."""
        return json.dumps(self.output_schema, indent=2)

    def format(self, **kwargs) -> str:
        """Format with schema instructions."""
        return super().format(**kwargs)

    def format_prompt(self, **kwargs) -> PromptValue:
        """Format and return as PromptValue."""
        text = self.format(**kwargs)
        return StringPromptValue(text=text)

    def _copy(self) -> "StructuredPromptTemplate":
        """Create a copy."""
        return StructuredPromptTemplate(
            template=self.template,
            output_schema=self.output_schema.copy(),
            input_variables=self.input_variables.copy(),
            format_instructions_key=self.format_instructions_key,
            partial_variables=self.partial_variables.copy(),
            output_parser=self.output_parser,
            validate_template=self.validate_template,
        )


# Convenience functions
def create_prompt_template(template: str, **kwargs) -> PromptTemplate:
    """Create a simple prompt template."""
    return PromptTemplate.from_template(template, **kwargs)


def create_chat_template(
    system_message: Optional[str] = None,
    human_message: str = "{input}",
    ai_message: Optional[str] = None,
    **kwargs,
) -> ChatPromptTemplate:
    """Create a chat prompt template."""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": human_message})

    if ai_message:
        messages.append({"role": "assistant", "content": ai_message})

    return ChatPromptTemplate.from_messages(messages, **kwargs)


def create_few_shot_template(
    examples: List[Dict[str, str]],
    example_template: str,
    prefix: str = "",
    suffix: str = "",
    input_variables: Optional[List[str]] = None,
    **kwargs,
) -> FewShotPromptTemplate:
    """Create a few-shot prompt template."""
    example_prompt = PromptTemplate.from_template(example_template)

    return FewShotPromptTemplate.from_examples(
        examples=examples,
        suffix=suffix,
        input_variables=input_variables or ["input"],
        example_prompt=example_prompt,
        prefix=prefix,
        **kwargs,
    )


__all__ = [
    # Base classes
    "PromptValue",
    "StringPromptValue",
    "ChatPromptValue",
    "BasePromptTemplate",
    # Templates
    "PromptTemplate",
    "ChatPromptTemplate",
    "FewShotPromptTemplate",
    "DynamicFewShotPromptTemplate",
    "PipelinePromptTemplate",
    "StructuredPromptTemplate",
    # Example Selectors
    "BaseExampleSelector",
    "LengthBasedExampleSelector",
    "SemanticSimilarityExampleSelector",
    # Convenience functions
    "create_prompt_template",
    "create_chat_template",
    "create_few_shot_template",
]
