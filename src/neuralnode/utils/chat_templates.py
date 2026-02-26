"""Chat templates for different LLM models.

Proper prompt formatting for various models like Llama3, Qwen, Mistral, etc.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..core.types import Message, MessageRole


@dataclass
class ChatTemplate:
    """A chat template for a specific model family."""
    
    name: str
    system_format: str = "{content}"
    user_format: str = "{content}"
    assistant_format: str = "{content}"
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


# Llama 3 Template
LLAMA3_TEMPLATE = ChatTemplate(
    name="llama3",
    system_format="<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
    assistant_format="<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>",
    stop_sequences=["<|eot_id|>"],
)

# Qwen Template
QWEN_TEMPLATE = ChatTemplate(
    name="qwen",
    system_format="<|im_start|>system\n{content}<|im_end|>",
    user_format="<|im_start|>user\n{content}<|im_end|>",
    assistant_format="<|im_start|>assistant\n{content}<|im_end|>",
    stop_sequences=["<|im_end|>"],
)

# Mistral Template (no system message support)
MISTRAL_TEMPLATE = ChatTemplate(
    name="mistral",
    user_format="[INST] {content} [/INST]",
    assistant_format=" {content}</s>",
    stop_sequences=["</s>"],
)

# Zephyr Template
ZEPHYR_TEMPLATE = ChatTemplate(
    name="zephyr",
    system_format="<|system|>\n{content}</s>\n",
    user_format="<|user|>\n{content}</s>\n",
    assistant_format="<|assistant|>\n{content}</s>\n",
    stop_sequences=["</s>"],
)

# Phi-3 Template
PHI3_TEMPLATE = ChatTemplate(
    name="phi3",
    system_format="<|system|>\n{content}<|end|>\n",
    user_format="<|user|>\n{content}<|end|>\n",
    assistant_format="<|assistant|>\n{content}<|end|>\n",
    stop_sequences=["<|end|>"],
)

# Gemma Template
GEMMA_TEMPLATE = ChatTemplate(
    name="gemma",
    user_format="<start_of_turn>user\n{content}<end_of_turn>\n",
    assistant_format="<start_of_turn>model\n{content}<end_of_turn>\n",
    stop_sequences=["<end_of_turn>"],
)

# ChatML Template (used by many models)
CHATML_TEMPLATE = ChatTemplate(
    name="chatml",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n",
    assistant_format="<|im_start|>assistant\n{content}<|im_end|>\n",
    stop_sequences=["<|im_end|>"],
)

# Default/OpenAI format
DEFAULT_TEMPLATE = ChatTemplate(
    name="default",
    system_format="{content}",
    user_format="{content}",
    assistant_format="{content}",
)

# Model to template mapping
MODEL_TEMPLATES = {
    # Llama 3 models
    "llama3": LLAMA3_TEMPLATE,
    "llama-3": LLAMA3_TEMPLATE,
    "llama-3.1": LLAMA3_TEMPLATE,
    "llama3.1": LLAMA3_TEMPLATE,
    "llama3:": LLAMA3_TEMPLATE,
    
    # Qwen models
    "qwen": QWEN_TEMPLATE,
    "qwen2": QWEN_TEMPLATE,
    "qwen2.5": QWEN_TEMPLATE,
    
    # Mistral models
    "mistral": MISTRAL_TEMPLATE,
    "mixtral": MISTRAL_TEMPLATE,
    "codestral": MISTRAL_TEMPLATE,
    
    # Zephyr models
    "zephyr": ZEPHYR_TEMPLATE,
    
    # Phi-3 models
    "phi": PHI3_TEMPLATE,
    "phi3": PHI3_TEMPLATE,
    "phi-3": PHI3_TEMPLATE,
    
    # Gemma models
    "gemma": GEMMA_TEMPLATE,
    "gemma2": GEMMA_TEMPLATE,
    
    # ChatML models
    "chatml": CHATML_TEMPLATE,
    "openchat": CHATML_TEMPLATE,
    "starling": CHATML_TEMPLATE,
    "neural-chat": CHATML_TEMPLATE,
    "snorkel": CHATML_TEMPLATE,
    "dolphin": CHATML_TEMPLATE,
}


class ChatTemplateManager:
    """Manages chat templates for different models."""
    
    def __init__(self):
        self._templates = MODEL_TEMPLATES.copy()
        self._default = DEFAULT_TEMPLATE
    
    def get_template(self, model_name: str) -> ChatTemplate:
        """Get the appropriate template for a model."""
        model_lower = model_name.lower()
        
        # Direct match
        if model_lower in self._templates:
            return self._templates[model_lower]
        
        # Partial match (e.g., "llama3:70b" should match "llama3")
        for template_name, template in self._templates.items():
            if model_lower.startswith(template_name.lower()):
                return template
        
        return self._default
    
    def register_template(self, model_name: str, template: ChatTemplate) -> None:
        """Register a custom template for a model."""
        self._templates[model_name.lower()] = template
    
    def apply_template(
        self,
        messages: List[Message],
        model_name: str,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of messages
            model_name: Model name to determine template
            add_generation_prompt: Whether to add assistant prefix at end
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(model_name)
        result_parts = []
        
        for msg in messages:
            content = msg.content
            
            if msg.role == MessageRole.SYSTEM:
                formatted = template.system_format.format(content=content)
            elif msg.role == MessageRole.USER:
                formatted = template.user_format.format(content=content)
            elif msg.role == MessageRole.ASSISTANT:
                formatted = template.assistant_format.format(content=content)
            else:
                formatted = content
            
            result_parts.append(formatted)
        
        # Add generation prompt
        if add_generation_prompt:
            result_parts.append(template.assistant_format.format(content="").rstrip())
        
        return "".join(result_parts)
    
    def get_stop_sequences(self, model_name: str) -> List[str]:
        """Get stop sequences for a model."""
        template = self.get_template(model_name)
        return template.stop_sequences.copy()


# Global template manager instance
template_manager = ChatTemplateManager()


def apply_chat_template(
    messages: List[Message],
    model_name: str,
    add_generation_prompt: bool = True,
) -> str:
    """
    Convenience function to apply chat template.
    
    Example:
        from neuralnode.utils.chat_templates import apply_chat_template
        
        prompt = apply_chat_template(messages, "llama3")
    """
    return template_manager.apply_template(messages, model_name, add_generation_prompt)


def get_stop_sequences(model_name: str) -> List[str]:
    """Get stop sequences for a model."""
    return template_manager.get_stop_sequences(model_name)
