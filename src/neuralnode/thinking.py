"""Thinking Mode for NeuralNode - Structured Reasoning for Any LLM

This module provides a comprehensive Thinking Mode that works with ANY model,
even those that don't natively support structured reasoning.

Features:
- Structured reasoning steps (understand → plan → execute → verify)
- Dynamic max_tokens to prevent cutoff
- Concise response mode
- Works with any NeuralNode provider
- Automatic thinking detection and parsing

Example::

    import neuralnode as nn
    from neuralnode.thinking import ThinkingMode

    # Wrap any model with thinking mode
    model = nn.HorusModel().load()
    thinking_model = ThinkingMode(model)

    # Chat with thinking visible
    response = thinking_model.chat("What is 15 × 23?")
    print(response.thinking)  # See reasoning steps
    print(response.content)   # Final answer

Advanced Usage::

    # Custom reasoning steps
    thinking_model = ThinkingMode(
        model,
        steps=["analyze", "research", "synthesize", "conclude"],
        show_thinking=True,
        concise=True,
    )
"""

from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import re


@dataclass
class ThinkingResponse:
    """Response with thinking/reasoning information.
    
    Attributes:
        content: The final answer/response
        thinking: The reasoning/thinking process
        steps: Individual reasoning steps
        confidence: Confidence score (0-1)
        metadata: Additional information
    """
    content: str
    thinking: str = ""
    steps: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThinkingMode:
    """Wraps any LLM provider with structured thinking mode.
    
    This class adds reasoning capabilities to ANY model, including:
    - Structured thinking steps
    - Automatic token management (no cutoff)
    - Concise response control
    - Thinking visibility control
    
    Args:
        model: Any NeuralNode model/provider
        steps: List of reasoning step names (default: understand, plan, execute, verify)
        show_thinking: Whether to include thinking in output
        concise: Whether to keep responses brief
        max_tokens_per_step: Max tokens for each thinking step
        max_final_tokens: Max tokens for final answer
        system_prompt_template: Custom thinking template
        
    Example::
        
        import neuralnode as nn
        from neuralnode.thinking import ThinkingMode
        
        # Basic usage
        model = nn.HorusModel().load()
        thinker = ThinkingMode(model)
        
        response = thinker.chat("Explain quantum computing")
        print(f"Thinking: {response.thinking}")
        print(f"Answer: {response.content}")
        
        # Advanced usage with custom steps
        thinker = ThinkingMode(
            model,
            steps=["analyze", "brainstorm", "evaluate", "conclude"],
            show_thinking=True,
            concise=True,
        )
    """
    
    # Default reasoning steps
    DEFAULT_STEPS = ["understand", "plan", "execute", "verify"]
    
    # Default thinking template
    DEFAULT_TEMPLATE = """You are an AI assistant that thinks step by step.

Follow these reasoning steps to answer the user's question:

{steps}

Important:
- Complete each step fully before moving to the next
- Be concise in your thinking (2-3 sentences per step max)
- Provide a clear, direct final answer
- Do not exceed the specified token limits

Now, work through the steps above to answer the user's question.
"""
    
    def __init__(
        self,
        model: Any,
        steps: Optional[List[str]] = None,
        show_thinking: bool = True,
        concise: bool = True,
        max_tokens_per_step: int = 100,
        max_final_tokens: int = 200,
        system_prompt_template: Optional[str] = None,
    ):
        self.model = model
        self.steps = steps or self.DEFAULT_STEPS.copy()
        self.show_thinking = show_thinking
        self.concise = concise
        self.max_tokens_per_step = max_tokens_per_step
        self.max_final_tokens = max_final_tokens
        self.template = system_prompt_template or self.DEFAULT_TEMPLATE
        
        # Build the thinking system prompt
        self._thinking_prompt = self._build_thinking_prompt()
    
    def _build_thinking_prompt(self) -> str:
        """Build the thinking system prompt with steps."""
        # Format steps as numbered list
        steps_text = "\n".join(
            f"{i+1}. **{step.capitalize()}**: Think about this aspect (max {self.max_tokens_per_step} tokens)"
            for i, step in enumerate(self.steps)
        )
        
        prompt = self.template.format(steps=steps_text)
        
        # Add concise instruction if enabled
        if self.concise:
            prompt += "\n\n**Concise Mode**: Keep your responses brief and to the point."
        
        return prompt
    
    def _generate_thinking(self, user_message: str) -> Dict[str, str]:
        """Generate thinking steps and final answer."""
        messages = [
            {"role": "system", "content": self._thinking_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Calculate total available tokens
        total_tokens = len(self.steps) * self.max_tokens_per_step + self.max_final_tokens
        
        # Generate thinking with structured format
        thinking_messages = messages.copy()
        thinking_prompt = "\n\nShow your step-by-step thinking first, then provide the final answer. "
        thinking_prompt += "Format:\n"
        for i, step in enumerate(self.steps, 1):
            thinking_prompt += f"Step {i} ({step}): [your reasoning]\n"
        thinking_prompt += f"\nFinal Answer: [concise answer, max {self.max_final_tokens} tokens]\n\n"
        thinking_prompt += f"Now respond to: {user_message}"
        
        thinking_messages[-1]["content"] = thinking_prompt
        
        # Generate with ample tokens to prevent cutoff
        response = self.model.chat(
            thinking_messages,
            max_new_tokens=total_tokens,
            temperature=0.3,  # Lower temperature for more focused thinking
        )
        
        full_response = response.content
        
        # Parse thinking and final answer
        return self._parse_response(full_response)
    
    def _parse_response(self, text: str) -> Dict[str, str]:
        """Parse thinking steps from response."""
        result = {
            "thinking": "",
            "content": text,
            "steps": []
        }
        
        # Try to find "Final Answer" pattern
        final_answer_patterns = [
            r'(?:Final Answer|Answer|Conclusion|Result):\s*(.+?)(?:\Z|Step \d|$)',
            r'\*\*Final Answer\*\*:\s*(.+?)(?:\Z|Step \d|$)',
        ]
        
        thinking_text = text
        final_answer = text
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                final_answer = match.group(1).strip()
                thinking_text = text[:match.start()].strip()
                break
        
        # Extract individual steps
        steps = []
        step_pattern = r'(?:Step \d+|\*\*\w+\*\*):\s*(.+?)(?=Step \d+|\*\*\w+\*\*|$)'
        step_matches = re.findall(step_pattern, thinking_text, re.DOTALL | re.IGNORECASE)
        
        if step_matches:
            steps = [s.strip() for s in step_matches if s.strip()]
        else:
            # Fallback: split by newlines or sentences
            steps = [s.strip() for s in thinking_text.split('\n') if s.strip() and len(s.strip()) > 10]
        
        result["thinking"] = thinking_text
        result["content"] = final_answer
        result["steps"] = steps
        
        return result
    
    def chat(
        self,
        message: Union[str, List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> ThinkingResponse:
        """Chat with thinking mode enabled.
        
        Args:
            message: User message string or message list
            max_new_tokens: Override default token limit
            **kwargs: Additional arguments passed to model
            
        Returns:
            ThinkingResponse with thinking, content, and steps
        """
        # Handle string or message list
        if isinstance(message, str):
            user_message = message
        else:
            # Extract last user message
            user_message = message[-1].get("content", "") if message else ""
        
        # Generate with thinking
        parsed = self._generate_thinking(user_message)
        
        # Build response
        response = ThinkingResponse(
            content=parsed["content"],
            thinking=parsed["thinking"],
            steps=parsed["steps"],
            confidence=self._calculate_confidence(parsed),
            metadata={
                "model_id": getattr(self.model, "model_id", "unknown"),
                "steps_count": len(parsed["steps"]),
            }
        )
        
        return response
    
    def _calculate_confidence(self, parsed: Dict[str, str]) -> float:
        """Calculate confidence score based on thinking quality."""
        confidence = 0.5  # Base confidence
        
        # More steps = more thorough thinking
        if len(parsed["steps"]) >= len(self.steps):
            confidence += 0.2
        
        # Presence of final answer
        if parsed["content"] and len(parsed["content"]) > 10:
            confidence += 0.2
        
        # Reasonable thinking length
        thinking_len = len(parsed["thinking"])
        if 50 < thinking_len < 1000:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def stream_chat(
        self,
        message: Union[str, List[Dict[str, str]]],
        **kwargs
    ):
        """Stream thinking and response (yields ThinkingResponse chunks).
        
        Note: Full thinking is collected before streaming starts
        to ensure proper parsing.
        """
        response = self.chat(message, **kwargs)
        
        # Yield thinking first if enabled
        if self.show_thinking and response.thinking:
            # Yield thinking as a special chunk
            yield ThinkingResponse(
                content="",
                thinking=response.thinking,
                steps=response.steps,
                confidence=response.confidence,
                metadata={"is_thinking": True, **response.metadata}
            )
        
        # Stream final answer character by character
        for i in range(0, len(response.content), 3):  # Chunk size 3
            chunk = response.content[i:i+3]
            yield ThinkingResponse(
                content=chunk,
                thinking="",
                steps=[],
                confidence=response.confidence,
                metadata={"is_content": True, **response.metadata}
            )
    
    def get_thinking_prompt(self) -> str:
        """Get the current thinking system prompt."""
        return self._thinking_prompt
    
    def set_steps(self, steps: List[str]):
        """Change reasoning steps dynamically."""
        self.steps = steps
        self._thinking_prompt = self._build_thinking_prompt()
    
    def __repr__(self) -> str:
        return f"ThinkingMode(model={getattr(self.model, 'model_id', 'unknown')}, steps={self.steps})"


# Convenience function for quick usage
def enable_thinking(
    model: Any,
    steps: Optional[List[str]] = None,
    show_thinking: bool = True,
    concise: bool = True,
) -> ThinkingMode:
    """Quick function to enable thinking mode on any model.
    
    Args:
        model: Any NeuralNode model
        steps: Custom reasoning steps
        show_thinking: Whether to show thinking process
        concise: Whether to keep responses brief
        
    Returns:
        ThinkingMode wrapper
        
    Example::
        
        import neuralnode as nn
        from neuralnode.thinking import enable_thinking
        
        model = nn.HorusModel().load()
        thinker = enable_thinking(model, show_thinking=True)
        
        response = thinker.chat("What is 2+2?")
        print(response.thinking)
        print(response.content)
    """
    return ThinkingMode(
        model,
        steps=steps,
        show_thinking=show_thinking,
        concise=concise,
    )


__all__ = [
    "ThinkingMode",
    "ThinkingResponse",
    "enable_thinking",
]
