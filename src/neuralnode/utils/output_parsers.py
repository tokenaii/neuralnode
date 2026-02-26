"""Output parsers for converting LLM text to structured data."""

import re
import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError

T = TypeVar("T")


class OutputParser(ABC, Generic[T]):
    """Base class for output parsers."""
    
    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse the LLM output text."""
        pass
    
    @abstractmethod
    def get_format_instructions(self) -> str:
        """Get instructions for the LLM on how to format output."""
        pass


class StrOutputParser(OutputParser[str]):
    """Simple string output parser."""
    
    def parse(self, text: str) -> str:
        return text.strip()
    
    def get_format_instructions(self) -> str:
        return "Respond with plain text."


class JSONOutputParser(OutputParser[Dict[str, Any]]):
    """Parse JSON output from LLM."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling code blocks."""
        # Try to extract JSON from code blocks
        code_block_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        
        if matches:
            text = matches[0]
        
        # Try to find JSON object in text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from text
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from: {text[:200]}")
    
    def get_format_instructions(self) -> str:
        if self.schema:
            return f"""Respond with a JSON object matching this schema:
```json
{json.dumps(self.schema, indent=2)}
```

Respond ONLY with valid JSON."""
        return "Respond with valid JSON."


class PydanticOutputParser(OutputParser[BaseModel]):
    """Parse output into a Pydantic model."""
    
    def __init__(self, model_class: Type[BaseModel]):
        self.model_class = model_class
    
    def parse(self, text: str) -> BaseModel:
        """Parse text into Pydantic model."""
        # First try JSON parsing
        json_parser = JSONOutputParser()
        try:
            data = json_parser.parse(text)
            return self.model_class(**data)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Try to extract key-value pairs from text
        data = self._extract_key_values(text)
        try:
            return self.model_class(**data)
        except ValidationError as e:
            raise ValueError(f"Failed to parse into {self.model_class.__name__}: {e}")
    
    def _extract_key_values(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text."""
        data = {}
        
        # Look for patterns like "Key: Value" or "Key = Value"
        patterns = [
            r'(\w+):\s*(.+?)(?=\n\w+:|$)',
            r'(\w+)\s*=\s*(.+?)(?=\n\w+\s*=|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            for key, value in matches:
                key = key.strip().lower()
                value = value.strip()
                
                # Try to convert value types
                if value.lower() in ('true', 'yes'):
                    data[key] = True
                elif value.lower() in ('false', 'no'):
                    data[key] = False
                else:
                    try:
                        data[key] = int(value)
                    except ValueError:
                        try:
                            data[key] = float(value)
                        except ValueError:
                            data[key] = value
        
        return data
    
    def get_format_instructions(self) -> str:
        schema = self.model_class.model_json_schema()
        return f"""Respond with a JSON object matching this schema:
```json
{json.dumps(schema, indent=2)}
```

Respond ONLY with valid JSON."""


class ListOutputParser(OutputParser[List[str]]):
    """Parse a list of items from LLM output."""
    
    def __init__(self, separator: str = r'\n', numbered: bool = True):
        self.separator = separator
        self.numbered = numbered
    
    def parse(self, text: str) -> List[str]:
        """Parse list from text."""
        items = []
        
        # Split by separator
        raw_items = re.split(self.separator, text)
        
        for item in raw_items:
            item = item.strip()
            if not item:
                continue
            
            # Remove numbering if present
            if self.numbered:
                item = re.sub(r'^\d+[.\-)]\s*', '', item)
            
            # Remove bullet points
            item = re.sub(r'^[\-\*•]\s*', '', item)
            
            if item:
                items.append(item)
        
        return items
    
    def get_format_instructions(self) -> str:
        return "Respond with a list of items, one per line. You may use numbers or bullet points."


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse comma-separated list."""
    
    def __init__(self):
        super().__init__(separator=r',', numbered=False)
    
    def parse(self, text: str) -> List[str]:
        """Parse comma-separated values."""
        items = []
        for item in text.split(','):
            item = item.strip()
            if item:
                items.append(item)
        return items
    
    def get_format_instructions(self) -> str:
        return "Respond with items separated by commas."


class BooleanOutputParser(OutputParser[bool]):
    """Parse yes/no or true/false output."""
    
    def __init__(
        self,
        affirmative_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ):
        self.affirmative = affirmative_words or ['yes', 'true', '1', 'correct', 'affirmative']
        self.negative = negative_words or ['no', 'false', '0', 'incorrect', 'negative']
    
    def parse(self, text: str) -> bool:
        """Parse boolean from text."""
        text_lower = text.strip().lower()
        
        # Check first word
        first_word = text_lower.split()[0] if text_lower else ''
        
        if first_word in [w.lower() for w in self.affirmative]:
            return True
        if first_word in [w.lower() for w in self.negative]:
            return False
        
        # Check contains
        for word in self.affirmative:
            if word.lower() in text_lower:
                return True
        for word in self.negative:
            if word.lower() in text_lower:
                return False
        
        # Default to parsing as bool
        try:
            return bool(int(text_lower))
        except ValueError:
            raise ValueError(f"Could not determine boolean from: {text}")
    
    def get_format_instructions(self) -> str:
        return f"Respond with {'/'.join(self.affirmative[:2])} or {'/'.join(self.negative[:2])}."


class MarkdownOutputParser(OutputParser[Dict[str, str]]):
    """Parse markdown formatted output into sections."""
    
    def parse(self, text: str) -> Dict[str, str]:
        """Parse markdown headers and content."""
        sections = {}
        current_header = "content"
        current_content = []
        
        for line in text.split('\n'):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_content:
                    sections[current_header] = '\n'.join(current_content).strip()
                    current_content = []
                
                current_header = header_match.group(2).strip().lower()
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_header] = '\n'.join(current_content).strip()
        
        return sections
    
    def get_format_instructions(self) -> str:
        return "Respond with markdown formatted text using headers (##) for sections."


class RegexOutputParser(OutputParser[Dict[str, str]]):
    """Parse output using regex patterns."""
    
    def __init__(self, patterns: Dict[str, str]):
        """
        Args:
            patterns: Dict of field name to regex pattern
        """
        self.patterns = {k: re.compile(v) for k, v in patterns.items()}
    
    def parse(self, text: str) -> Dict[str, str]:
        """Extract fields using regex."""
        result = {}
        for field, pattern in self.patterns.items():
            match = pattern.search(text)
            if match:
                result[field] = match.group(1) if match.groups() else match.group(0)
        return result
    
    def get_format_instructions(self) -> str:
        return "Respond with text that matches the expected format."


class EnumOutputParser(OutputParser[str]):
    """Parse output that must be one of a set of values."""
    
    def __init__(self, values: List[str], case_sensitive: bool = False):
        self.values = values
        self.case_sensitive = case_sensitive
    
    def parse(self, text: str) -> str:
        """Parse and validate enum value."""
        text_stripped = text.strip()
        
        if self.case_sensitive:
            if text_stripped in self.values:
                return text_stripped
        else:
            text_lower = text_stripped.lower()
            for value in self.values:
                if value.lower() == text_lower:
                    return value
        
        # Try to find closest match
        raise ValueError(
            f"Value '{text_stripped}' not in allowed values: {self.values}"
        )
    
    def get_format_instructions(self) -> str:
        return f"Respond with one of: {', '.join(self.values)}"


# Convenience functions
def parse_json(text: str) -> Dict[str, Any]:
    """Convenience function to parse JSON."""
    return JSONOutputParser().parse(text)


def parse_list(text: str, separator: str = '\n') -> List[str]:
    """Convenience function to parse list."""
    return ListOutputParser(separator=separator).parse(text)


def parse_pydantic(text: str, model_class: Type[T]) -> T:
    """Convenience function to parse into Pydantic model."""
    parser = PydanticOutputParser(model_class)
    return parser.parse(text)
