"""Universal Reasoning Layer - Advanced reasoning for ALL models.

This module adds reasoning capabilities to ANY model, including:
- Chain-of-Thought prompting
- Step-by-step decomposition
- Self-reflection and correction
- Multi-step planning
- Structured reasoning output
- Reasoning templates for different tasks

Works with ALL models: GPT-4, Claude, local LLMs, GGUF models, etc.
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

from ..core import BaseProvider, BaseTool, Message, MessageRole, LLMResponse, ensure_llm_response


class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    STEP_BY_STEP = "step_by_step"
    DECOMPOSITION = "decomposition"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"
    PROBLEM_SOLVING = "problem_solving"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    ABDUCTIVE = "abductive"


@dataclass
class ReasoningStep:
    """A step in the reasoning process."""
    number: int
    title: str
    content: str
    evidence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    substeps: List['ReasoningStep'] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "title": self.title,
            "content": self.content,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "substeps": [s.to_dict() for s in self.substeps]
        }


@dataclass
class ReasoningResult:
    """Result of reasoning process."""
    original_query: str
    strategy: ReasoningStrategy
    steps: List[ReasoningStep]
    conclusion: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.original_query,
            "strategy": self.strategy.value,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "metadata": self.metadata
        }
    
    def format_trace(self) -> str:
        """Format reasoning trace for display."""
        lines = [
            f"🧠 Reasoning Process ({self.strategy.value})",
            f"Query: {self.original_query}",
            "=" * 70,
            ""
        ]
        
        for step in self.steps:
            lines.append(f"\n📍 Step {step.number}: {step.title}")
            lines.append(f"   {step.content}")
            if step.evidence:
                lines.append(f"   Evidence: {', '.join(step.evidence)}")
            if step.confidence < 1.0:
                lines.append(f"   Confidence: {step.confidence:.0%}")
        
        lines.extend([
            "",
            "=" * 70,
            f"🎯 Conclusion: {self.conclusion}",
            f"Overall Confidence: {self.confidence:.0%}"
        ])
        
        if self.alternatives:
            lines.append(f"\nAlternatives considered: {len(self.alternatives)}")
        
        return "\n".join(lines)


class ReasoningTemplate:
    """Templates for different reasoning tasks."""
    
    TEMPLATES = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: """Let's work through this step by step.

Problem: {query}

Think through this carefully:
1. What is the core question or problem?
2. What information do we have?
3. What are the key considerations?
4. Step-by-step reasoning:
   - Step 1: [First step]
   - Step 2: [Second step]
   - Step 3: [Third step]
5. What is the logical conclusion?

Provide your answer in this format:
REASONING: [Your detailed step-by-step thinking]
CONCLUSION: [Your final answer]
CONFIDENCE: [High/Medium/Low]""",

        ReasoningStrategy.STEP_BY_STEP: """Break down this task into clear steps.

Task: {query}

Step-by-Step Plan:
1. [First action to take]
2. [Second action to take]
3. [Third action to take]
...
N. [Final action]

For each step, explain:
- What needs to be done
- Why this step is necessary
- What the expected outcome is

Final Answer: [Complete solution after all steps]""",

        ReasoningStrategy.DECOMPOSITION: """Decompose this complex problem into smaller parts.

Complex Problem: {query}

Part 1: [First component]
- Analysis:
- Key factors:
- Partial solution:

Part 2: [Second component]
- Analysis:
- Key factors:
- Partial solution:

[Additional parts as needed...]

Integration:
How do these parts fit together?
What is the overall solution?

Synthesized Answer: [Complete integrated solution]""",

        ReasoningStrategy.SELF_CONSISTENCY: """Solve this problem multiple ways and check consistency.

Problem: {query}

Approach 1: [First method]
Solution: [Answer from first method]

Approach 2: [Second method]
Solution: [Answer from second method]

Approach 3: [Third method if applicable]
Solution: [Answer from third method]

Consistency Check:
- Do the solutions agree? 
- If not, why might they differ?
- Which solution is most reliable?

Final Answer: [Best answer with justification]""",

        ReasoningStrategy.REACT: """Use Reasoning and Acting to solve this.

Task: {query}

Think: [What do I need to know? What should I do?]
Act: [Action to take or question to answer]
Observe: [Result of the action]

[Repeat Think-Act-Observe as needed]

Final Answer: [Complete solution based on observations]""",

        ReasoningStrategy.PROBLEM_SOLVING: """Apply structured problem-solving.

Problem: {query}

1. PROBLEM DEFINITION
   - What exactly is the problem?
   - What are the constraints?
   - What would a solution look like?

2. ANALYSIS
   - What are the key variables?
   - What are the relationships?
   - What are the unknowns?

3. SOLUTION GENERATION
   - Possible approaches:
   - Pros/cons of each:
   - Selected approach:

4. VERIFICATION
   - Does this solution work?
   - Test cases/validation:
   - Edge cases considered:

Solution: [Complete validated solution]""",

        ReasoningStrategy.COMPARATIVE: """Compare and evaluate options.

Question: {query}

Options to Consider:
A) [First option]
   - Pros:
   - Cons:
   - Score: [1-10]

B) [Second option]
   - Pros:
   - Cons:
   - Score: [1-10]

C) [Third option if applicable]
   - Pros:
   - Cons:
   - Score: [1-10]

Comparison:
- Key differences:
- Trade-offs:
- Best for which scenarios:

Recommendation: [Best option with reasoning]""",

        ReasoningStrategy.CAUSAL: """Analyze causal relationships.

Question: {query}

Causal Chain:
- Root cause: [Underlying factor]
- Direct causes: [Immediate factors]
- Contributing factors: [Other relevant factors]
- Effects: [Outcomes]

Analysis:
- Correlation vs causation:
- Confounding variables:
- Evidence strength:

Conclusion: [Causal explanation]""",

        ReasoningStrategy.ABDUCTIVE: """Find the best explanation (inference to best explanation).

Observations: {query}

Possible Explanations:
1. [First hypothesis]
   - Evidence for:
   - Evidence against:
   - Plausibility: [High/Medium/Low]

2. [Second hypothesis]
   - Evidence for:
   - Evidence against:
   - Plausibility: [High/Medium/Low]

3. [Third hypothesis]
   - Evidence for:
   - Evidence against:
   - Plausibility: [High/Medium/Low]

Best Explanation:
- Which hypothesis best fits the evidence?
- Why is it superior to alternatives?
- What would falsify it?

Conclusion: [Most likely explanation]"""
    }
    
    @classmethod
    def get(cls, strategy: ReasoningStrategy, query: str) -> str:
        """Get formatted template."""
        template = cls.TEMPLATES.get(strategy, cls.TEMPLATES[ReasoningStrategy.CHAIN_OF_THOUGHT])
        return template.format(query=query)


class UniversalReasoner:
    """Universal reasoning layer that works with ANY model.
    
    Adds advanced reasoning capabilities to any LLM, whether it's
    GPT-4, Claude, or a 1B parameter local model.
    """
    
    def __init__(
        self,
        llm: BaseProvider,
        default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        max_reasoning_steps: int = 10,
        enable_self_correction: bool = True,
        confidence_threshold: float = 0.7
    ):
        self.llm = llm
        self.default_strategy = default_strategy
        self.max_reasoning_steps = max_reasoning_steps
        self.enable_self_correction = enable_self_correction
        self.confidence_threshold = confidence_threshold
        self.reasoning_history: List[ReasoningResult] = []
    
    def reason(
        self,
        query: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None
    ) -> ReasoningResult:
        """Perform reasoning on a query.
        
        Args:
            query: The question or problem to reason about
            strategy: Reasoning strategy to use
            context: Additional context
            tools: Tools to use during reasoning
        
        Returns:
            ReasoningResult with full reasoning trace
        """
        strategy = strategy or self.default_strategy
        
        # Build prompt with reasoning template
        prompt = self._build_reasoning_prompt(query, strategy, context)
        
        # Generate reasoning
        messages = [Message.user(prompt)]
        response = ensure_llm_response(self.llm.chat(messages))
        
        # Parse structured reasoning
        result = self._parse_reasoning_response(query, strategy, response.content)
        
        # Store in history
        self.reasoning_history.append(result)
        
        # Self-correction if enabled and low confidence
        if self.enable_self_correction and result.confidence < self.confidence_threshold:
            result = self._self_correct(result, tools)
        
        return result
    
    def _build_reasoning_prompt(
        self,
        query: str,
        strategy: ReasoningStrategy,
        context: Optional[str]
    ) -> str:
        """Build the reasoning prompt."""
        template = ReasoningTemplate.get(strategy, query)
        
        if context:
            template = f"Context: {context}\n\n{template}"
        
        return template
    
    def _parse_reasoning_response(
        self,
        query: str,
        strategy: ReasoningStrategy,
        content: str
    ) -> ReasoningResult:
        """Parse the model's reasoning response."""
        steps = []
        conclusion = ""
        confidence = 0.8  # Default
        
        # Try to extract conclusion
        conclusion_patterns = [
            r'CONCLUSION:\s*(.+?)(?=CONFIDENCE:|$)',
            r'Final Answer:\s*(.+?)(?=\n\n|$)',
            r'Solution:\s*(.+?)(?=\n\n|$)',
            r'Recommendation:\s*(.+?)(?=\n\n|$)',
        ]
        
        for pattern in conclusion_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                conclusion = match.group(1).strip()
                break
        
        if not conclusion:
            # Use last paragraph as conclusion
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            if paragraphs:
                conclusion = paragraphs[-1]
        
        # Try to extract confidence
        confidence_match = re.search(
            r'CONFIDENCE:\s*(High|Medium|Low|(\d+)%?)',
            content,
            re.IGNORECASE
        )
        if confidence_match:
            conf_str = confidence_match.group(1).lower()
            if conf_str == "high" or "90" in conf_str or "80" in conf_str:
                confidence = 0.9
            elif conf_str == "medium" or "70" in conf_str or "60" in conf_str:
                confidence = 0.7
            elif conf_str == "low":
                confidence = 0.5
            else:
                try:
                    confidence = int(conf_str) / 100
                except:
                    pass
        
        # Extract steps
        step_patterns = [
            r'(?:Step\s+(\d+)[.:]|(\d+)[.])\s*(.+?)(?=Step\s+\d+:|\d+\.|$)',
            r'-\s*(.+?)(?=\n-|\n\n|$)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for i, match in enumerate(matches[:self.max_reasoning_steps], 1):
                if isinstance(match, tuple):
                    if len(match) == 3:
                        step_num = match[0] or match[1] or str(i)
                        step_content = match[2]
                    else:
                        step_content = match[0]
                        step_num = str(i)
                else:
                    step_content = match
                    step_num = str(i)
                
                step_content = step_content.strip()
                if len(step_content) > 20:  # Filter out very short matches
                    steps.append(ReasoningStep(
                        number=i,
                        title=f"Step {step_num}",
                        content=step_content
                    ))
        
        # If no steps extracted, create one from content
        if not steps:
            steps.append(ReasoningStep(
                number=1,
                title="Reasoning",
                content=content[:1000]  # Truncate if very long
            ))
        
        return ReasoningResult(
            original_query=query,
            strategy=strategy,
            steps=steps,
            conclusion=conclusion or content[:500],
            confidence=confidence
        )
    
    def _self_correct(
        self,
        result: ReasoningResult,
        tools: Optional[List[BaseTool]] = None
    ) -> ReasoningResult:
        """Attempt to self-correct low-confidence reasoning."""
        correction_prompt = f"""Review your previous reasoning and improve it:

Original question: {result.original_query}
Your previous conclusion: {result.conclusion}
Confidence: {result.confidence:.0%}

Please reconsider:
1. What might you have missed?
2. Are there alternative interpretations?
3. Can you verify any facts?
4. What would strengthen this conclusion?

Provide an improved conclusion with higher confidence."""

        messages = [Message.user(correction_prompt)]
        
        # Use tools if available for fact-checking
        if tools and self.llm.supports_tools():
            response = ensure_llm_response(self.llm.chat(messages, tools=tools))
        else:
            response = ensure_llm_response(self.llm.chat(messages))
        
        # Parse improved reasoning
        improved = self._parse_reasoning_response(
            result.original_query,
            result.strategy,
            response.content
        )
        
        # Use improved version if confidence increased
        if improved.confidence > result.confidence:
            improved.metadata["self_corrected"] = True
            improved.metadata["original_confidence"] = result.confidence
            return improved
        
        return result
    
    def reason_with_tools(
        self,
        query: str,
        tools: List[BaseTool],
        strategy: ReasoningStrategy = ReasoningStrategy.REACT
    ) -> ReasoningResult:
        """Reasoning with tool use capability."""
        if not self.llm.supports_tools():
            # Fallback to manual tool calling
            return self._manual_tool_reasoning(query, tools, strategy)
        
        # Use native tool calling
        prompt = f"""Solve this problem using available tools:

{query}

Think step by step. When you need information, use the appropriate tool."""

        messages = [Message.user(prompt)]
        
        for iteration in range(self.max_reasoning_steps):
            response = ensure_llm_response(self.llm.chat(messages, tools=tools))
            
            if response.tool_calls:
                messages.append(Message.assistant(response.content))
                
                # Execute tools
                for tc in response.tool_calls:
                    tool = next((t for t in tools if t.name == tc.name), None)
                    if tool:
                        result = tool.execute(**tc.arguments)
                        messages.append(Message(
                            role=MessageRole.TOOL,
                            content=result.content,
                            metadata={"tool_name": tc.name}
                        ))
            else:
                # No more tool calls, we have the answer
                return self._parse_reasoning_response(query, strategy, response.content)
        
        return ReasoningResult(
            original_query=query,
            strategy=strategy,
            steps=[],
            conclusion="Maximum iterations reached",
            confidence=0.0
        )
    
    def _manual_tool_reasoning(
        self,
        query: str,
        tools: List[BaseTool],
        strategy: ReasoningStrategy
    ) -> ReasoningResult:
        """Manual tool reasoning for models without native tool support."""
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in tools
        ])
        
        prompt = f"""You have access to these tools:
{tool_descriptions}

To use a tool, output:
TOOL: [tool_name]
PARAMETERS: {{"param": "value"}}

Problem: {query}

Think about what tools you need, then solve step by step."""

        messages = [Message.user(prompt)]
        
        for iteration in range(self.max_reasoning_steps):
            response = ensure_llm_response(self.llm.chat(messages))
            content = response.content
            
            # Check for tool calls
            tool_match = re.search(
                r'TOOL:\s*(\w+)\s*PARAMETERS:\s*(\{[^}]*\})',
                content,
                re.IGNORECASE
            )
            
            if tool_match:
                tool_name = tool_match.group(1)
                params_str = tool_match.group(2)
                
                try:
                    params = json.loads(params_str)
                except:
                    params = {}
                
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    result = tool.execute(**params)
                    messages.append(Message.assistant(content))
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=result.content,
                        metadata={"tool_name": tool_name}
                    ))
                    continue
            
            # No tool call, we're done
            return self._parse_reasoning_response(query, strategy, content)
        
        return ReasoningResult(
            original_query=query,
            strategy=strategy,
            steps=[],
            conclusion="Maximum iterations reached",
            confidence=0.0
        )
    
    def compare_strategies(
        self,
        query: str,
        strategies: Optional[List[ReasoningStrategy]] = None
    ) -> Dict[ReasoningStrategy, ReasoningResult]:
        """Run same query with multiple strategies and compare."""
        strategies = strategies or [
            ReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningStrategy.STEP_BY_STEP,
            ReasoningStrategy.DECOMPOSITION
        ]
        
        results = {}
        for strategy in strategies:
            result = self.reason(query, strategy=strategy)
            results[strategy] = result
        
        return results
    
    def get_reasoning_history(self) -> List[ReasoningResult]:
        """Get history of all reasoning performed."""
        return self.reasoning_history
    
    def clear_history(self):
        """Clear reasoning history."""
        self.reasoning_history.clear()


# Convenience functions

def reason(
    query: str,
    llm: BaseProvider,
    strategy: str = "chain_of_thought",
    **kwargs
) -> ReasoningResult:
    """Quick reasoning function."""
    reasoner = UniversalReasoner(llm, **kwargs)
    strat = ReasoningStrategy(strategy)
    return reasoner.reason(query, strategy=strat)


def enable_reasoning(agent_class):
    """Decorator to add reasoning capabilities to any agent."""
    original_run = agent_class.run
    
    def run_with_reasoning(self, task: str, **kwargs):
        # Add reasoning before execution
        if hasattr(self, 'llm') and self.llm:
            reasoner = UniversalReasoner(self.llm)
            reasoning = reasoner.reason(task)
            
            # Enhanced task with reasoning
            enhanced_task = f"""Task: {task}

Reasoning to consider:
{reasoning.conclusion}

Now complete the task."""
            
            return original_run(self, enhanced_task, **kwargs)
        
        return original_run(self, task, **kwargs)
    
    agent_class.run = run_with_reasoning
    return agent_class
