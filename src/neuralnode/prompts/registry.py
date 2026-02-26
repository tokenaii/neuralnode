"""Prompt Registry for managing and sharing prompts."""

import json
import hashlib
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import yaml


@dataclass
class Prompt:
    """A registered prompt."""
    id: str
    name: str
    description: str
    template: str
    variables: List[str]
    tags: List[str]
    author: str
    created_at: datetime
    updated_at: datetime
    version: str = "1.0"
    usage_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def format(self, **kwargs) -> str:
        """Format the prompt with variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template": self.template,
            "variables": self.variables,
            "tags": self.tags,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "usage_count": self.usage_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prompt":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            template=data["template"],
            variables=data["variables"],
            tags=data.get("tags", []),
            author=data.get("author", "unknown"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            version=data.get("version", "1.0"),
            usage_count=data.get("usage_count", 0),
            metadata=data.get("metadata", {}),
        )


class PromptRegistry:
    """Central registry for managing prompts."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path.home() / ".neuralnode" / "prompts"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.prompts: Dict[str, Prompt] = {}
        self._load_all()
    
    def register(
        self,
        name: str,
        template: str,
        description: str = "",
        variables: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        author: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Prompt:
        """Register a new prompt."""
        # Extract variables from template if not provided
        if variables is None:
            variables = self._extract_variables(template)
        
        # Generate ID
        prompt_id = self._generate_id(name, template)
        
        # Create prompt
        now = datetime.now()
        prompt = Prompt(
            id=prompt_id,
            name=name,
            description=description,
            template=template,
            variables=variables,
            tags=tags or [],
            author=author,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
        
        self.prompts[prompt_id] = prompt
        self._save_prompt(prompt)
        
        return prompt
    
    def get(self, prompt_id: str) -> Optional[Prompt]:
        """Get a prompt by ID."""
        prompt = self.prompts.get(prompt_id)
        if prompt:
            prompt.usage_count += 1
            self._save_prompt(prompt)
        return prompt
    
    def get_by_name(self, name: str) -> Optional[Prompt]:
        """Get a prompt by name."""
        for prompt in self.prompts.values():
            if prompt.name.lower() == name.lower():
                return prompt
        return None
    
    def list_prompts(
        self,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
    ) -> List[Prompt]:
        """List prompts with optional filtering."""
        prompts = list(self.prompts.values())
        
        if tags:
            prompts = [p for p in prompts if any(t in p.tags for t in tags)]
        
        if author:
            prompts = [p for p in prompts if p.author == author]
        
        # Sort by usage (most used first)
        prompts.sort(key=lambda p: p.usage_count, reverse=True)
        
        return prompts
    
    def update(
        self,
        prompt_id: str,
        template: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Prompt]:
        """Update an existing prompt."""
        prompt = self.prompts.get(prompt_id)
        if not prompt:
            return None
        
        if template:
            prompt.template = template
            prompt.variables = self._extract_variables(template)
        
        if description:
            prompt.description = description
        
        if tags:
            prompt.tags = tags
        
        prompt.updated_at = datetime.now()
        prompt.version = self._increment_version(prompt.version)
        
        self._save_prompt(prompt)
        return prompt
    
    def delete(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        if prompt_id not in self.prompts:
            return False
        
        del self.prompts[prompt_id]
        
        # Delete file
        prompt_file = self.storage_path / f"{prompt_id}.json"
        if prompt_file.exists():
            prompt_file.unlink()
        
        return True
    
    def search(self, query: str) -> List[Prompt]:
        """Search prompts."""
        query_lower = query.lower()
        results = []
        
        for prompt in self.prompts.values():
            if (query_lower in prompt.name.lower() or
                query_lower in prompt.description.lower() or
                query_lower in prompt.template.lower() or
                any(query_lower in tag.lower() for tag in prompt.tags)):
                results.append(prompt)
        
        return results
    
    def export(self, filepath: str, format: str = "json") -> None:
        """Export all prompts to file."""
        data = [p.to_dict() for p in self.prompts.values()]
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
    
    def import_prompts(self, filepath: str, format: str = "json") -> int:
        """Import prompts from file."""
        with open(filepath, 'r') as f:
            if format == "json":
                data = json.load(f)
            elif format == "yaml":
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        count = 0
        for item in data:
            try:
                prompt = Prompt.from_dict(item)
                self.prompts[prompt.id] = prompt
                self._save_prompt(prompt)
                count += 1
            except Exception as e:
                print(f"Error importing prompt: {e}")
        
        return count
    
    def _generate_id(self, name: str, template: str) -> str:
        """Generate unique ID for prompt."""
        content = f"{name}:{template}"
        hash_obj = hashlib.md5(content.encode())
        return hash_obj.hexdigest()[:12]
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template."""
        import re
        # Find {variable} patterns
        matches = re.findall(r'\{(\w+)\}', template)
        return list(set(matches))  # Remove duplicates
    
    def _increment_version(self, version: str) -> str:
        """Increment version number."""
        try:
            parts = version.split('.')
            parts[-1] = str(int(parts[-1]) + 1)
            return '.'.join(parts)
        except ValueError:
            return version
    
    def _save_prompt(self, prompt: Prompt) -> None:
        """Save prompt to disk."""
        prompt_file = self.storage_path / f"{prompt.id}.json"
        with open(prompt_file, 'w') as f:
            json.dump(prompt.to_dict(), f, indent=2)
    
    def _load_all(self) -> None:
        """Load all prompts from disk."""
        for prompt_file in self.storage_path.glob("*.json"):
            try:
                with open(prompt_file, 'r') as f:
                    data = json.load(f)
                    prompt = Prompt.from_dict(data)
                    self.prompts[prompt.id] = prompt
            except Exception as e:
                print(f"Error loading prompt {prompt_file}: {e}")


class PromptHub:
    """Community prompt hub for sharing prompts."""
    
    def __init__(self, registry: Optional[PromptRegistry] = None):
        self.registry = registry or PromptRegistry()
        self.remote_url = "https://hub.neuralnode.ai"
    
    def share(self, prompt_id: str) -> Optional[str]:
        """Share a prompt to the hub."""
        prompt = self.registry.get(prompt_id)
        if not prompt:
            return None
        
        # In a real implementation, this would upload to a server
        # For now, we just simulate it
        share_id = f"hub_{prompt_id}"
        print(f"Prompt shared! ID: {share_id}")
        return share_id
    
    def load_from_hub(self, hub_id: str) -> Optional[Prompt]:
        """Load a prompt from the hub."""
        # In a real implementation, this would download from a server
        # For now, return None
        return None
    
    def list_community_prompts(
        self,
        category: Optional[str] = None,
        sort_by: str = "popular",
    ) -> List[Dict[str, Any]]:
        """List prompts from the community hub."""
        # This would fetch from remote API
        return []


# Built-in prompts
BUILT_IN_PROMPTS = {
    "zero-shot-react": Prompt(
        id="builtin_001",
        name="Zero-Shot ReAct",
        description="Zero-shot reasoning and acting agent prompt",
        template="""You are an AI assistant that can use tools to help answer questions.

You have access to the following tools:
{tools}

Use this format:
Question: the input question
Thought: think about what to do
Action: the action to take (one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer

Begin!

Question: {question}
Thought:""",
        variables=["tools", "tool_names", "question"],
        tags=["agent", "reasoning", "tools"],
        author="NeuralNode",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    ),
    
    "summarization": Prompt(
        id="builtin_002",
        name="Text Summarization",
        description="Summarize long text into key points",
        template="""Summarize the following text into {num_points} key points:

{text}

Provide a concise summary that captures the main ideas:""",
        variables=["text", "num_points"],
        tags=["summarization", "nlp"],
        author="NeuralNode",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    ),
    
    "code-review": Prompt(
        id="builtin_003",
        name="Code Review",
        description="Review code for issues and improvements",
        template="""Review the following {language} code and provide feedback:

```{language}
{code}
```

Check for:
1. Bugs and errors
2. Performance issues
3. Security vulnerabilities
4. Code style improvements
5. Best practices

Provide specific suggestions:""",
        variables=["language", "code"],
        tags=["coding", "review", "development"],
        author="NeuralNode",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    ),
}


# Convenience functions
def load_prompt(prompt_id: str, registry: Optional[PromptRegistry] = None) -> Optional[Prompt]:
    """Quick load prompt by ID."""
    reg = registry or PromptRegistry()
    return reg.get(prompt_id)


def create_prompt(
    name: str,
    template: str,
    **kwargs,
) -> Prompt:
    """Quick prompt creation."""
    registry = PromptRegistry()
    return registry.register(name, template, **kwargs)


def get_builtin_prompt(name: str) -> Optional[Prompt]:
    """Get a built-in prompt."""
    return BUILT_IN_PROMPTS.get(name)
