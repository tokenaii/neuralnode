"""
ReAct (Reasoning + Acting) Agent Pattern Implementation.
Advanced agent that reasons about each step and can self-correct.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class StepType(Enum):
    """Types of steps in ReAct cycle."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class ReActStep:
    """Single step in ReAct cycle."""
    step_number: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReActState:
    """Current state of ReAct agent."""
    task: str
    steps: List[ReActStep] = field(default_factory=list)
    current_step: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    finished: bool = False
    final_answer: Optional[str] = None


class ReActAgent:
    """
    ReAct Agent: Reasoning + Acting with self-correction.
    
    Based on the paper: "ReAct: Synergizing Reasoning and Acting in Language Models"
    """
    
    def __init__(self, llm, tools: Dict[str, Callable], max_iterations: int = 10):
        """
        Initialize ReAct Agent.
        
        Args:
            llm: Language model instance
            tools: Dictionary of available tools {name: function}
            max_iterations: Maximum reasoning steps
        """
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.history: List[ReActState] = []
    
    def run(self, task: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute task using ReAct pattern.
        
        Returns:
            Dict with 'final_answer', 'steps', and 'success'
        """
        state = ReActState(
            task=task,
            context=context or {},
            max_iterations=self.max_iterations,
        )
        
        for iteration in range(self.max_iterations):
            # Step 1: THINK
            thought = self._generate_thought(state)
            state.steps.append(ReActStep(
                step_number=iteration,
                step_type=StepType.THOUGHT,
                content=thought,
            ))
            
            # Check if we should finish
            if self._should_finish(thought):
                final_answer = self._extract_final_answer(thought)
                state.finished = True
                state.final_answer = final_answer
                state.steps.append(ReActStep(
                    step_number=iteration,
                    step_type=StepType.FINAL_ANSWER,
                    content=final_answer,
                ))
                break
            
            # Step 2: ACT
            action = self._determine_action(state)
            if action:
                tool_name, tool_input = action
                state.steps.append(ReActStep(
                    step_number=iteration,
                    step_type=StepType.ACTION,
                    content=f"Using tool: {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_input,
                ))
                
                # Execute action
                try:
                    if tool_name in self.tools:
                        tool_output = self.tools[tool_name](**tool_input)
                        success = True
                        error = None
                    else:
                        tool_output = f"Error: Tool '{tool_name}' not found"
                        success = False
                        error = tool_output
                except Exception as e:
                    tool_output = f"Error: {str(e)}"
                    success = False
                    error = str(e)
                
                # Step 3: OBSERVE
                state.steps.append(ReActStep(
                    step_number=iteration,
                    step_type=StepType.OBSERVATION,
                    content=str(tool_output)[:1000],  # Truncate long outputs
                    tool_output=str(tool_output)[:1000],
                    success=success,
                    error_message=error,
                ))
                
                # If action failed, add to context for self-correction
                if not success:
                    state.context[f"error_step_{iteration}"] = error
            
            state.current_step = iteration + 1
        
        # Store history
        self.history.append(state)
        
        return {
            "success": state.finished,
            "final_answer": state.final_answer,
            "steps": self._format_steps(state.steps),
            "iterations": state.current_step,
        }
    
    def _generate_thought(self, state: ReActState) -> str:
        """Generate reasoning thought based on current state."""
        # Build prompt with history
        prompt = self._build_react_prompt(state)
        
        # Get response from LLM
        response = self.llm.chat(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    
    def _build_react_prompt(self, state: ReActState) -> str:
        """Build ReAct prompt with full history."""
        prompt_parts = [
            "You are a ReAct agent. Solve the task by thinking step-by-step and using tools.",
            "",
            "Available tools:",
        ]
        
        for tool_name, tool_fn in self.tools.items():
            prompt_parts.append(f"- {tool_name}: {tool_fn.__doc__ or 'No description'}")
        
        prompt_parts.extend([
            "",
            f"Task: {state.task}",
            "",
            "Previous steps:",
        ])
        
        # Add previous steps
        for step in state.steps[-5:]:  # Last 5 steps to avoid context overflow
            if step.step_type == StepType.THOUGHT:
                prompt_parts.append(f"Thought: {step.content}")
            elif step.step_type == StepType.ACTION:
                prompt_parts.append(f"Action: {step.tool_name}({step.tool_input})")
            elif step.step_type == StepType.OBSERVATION:
                prompt_parts.append(f"Observation: {step.content}")
        
        prompt_parts.extend([
            "",
            "What is your next thought?",
            "If you have the answer, start with 'Final Answer:'",
            "If you need to use a tool, include 'Action: tool_name' and 'Input: {...}'",
            "Thought: ",
        ])
        
        return "\n".join(prompt_parts)
    
    def _should_finish(self, thought: str) -> bool:
        """Check if agent thinks task is complete."""
        finish_indicators = [
            "final answer:",
            "i have completed",
            "the answer is",
            "task completed",
            "finished",
        ]
        thought_lower = thought.lower()
        return any(indicator in thought_lower for indicator in finish_indicators)
    
    def _extract_final_answer(self, thought: str) -> str:
        """Extract final answer from thought."""
        # Try to find "Final Answer:" section
        if "Final Answer:" in thought:
            return thought.split("Final Answer:")[-1].strip()
        return thought.strip()
    
    def _determine_action(self, state: ReActState) -> Optional[Tuple[str, Dict]]:
        """Determine what action to take based on latest thought."""
        if not state.steps:
            return None
        
        latest_thought = state.steps[-1].content
        
        # Parse action from thought
        # Look for patterns like "Action: tool_name" and "Input: {...}"
        import re
        
        action_match = re.search(r'Action:\s*(\w+)', latest_thought, re.IGNORECASE)
        if not action_match:
            return None
        
        tool_name = action_match.group(1)
        
        # Try to parse input
        input_match = re.search(r'Input:\s*(\{[^}]+\})', latest_thought, re.DOTALL)
        if input_match:
            try:
                tool_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Try to parse as simple key=value pairs
                tool_input = self._parse_simple_input(input_match.group(1))
        else:
            # Extract parameters from thought context
            tool_input = self._extract_params_from_context(latest_thought, state)
        
        return tool_name, tool_input
    
    def _parse_simple_input(self, input_str: str) -> Dict:
        """Parse simple key=value input format."""
        params = {}
        for pair in input_str.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                params[key.strip()] = value.strip().strip('"\'')
        return params
    
    def _extract_params_from_context(self, thought: str, state: ReActState) -> Dict:
        """Extract parameters based on tool and context."""
        # Default empty params
        return {"query": state.task}
    
    def _format_steps(self, steps: List[ReActStep]) -> List[Dict]:
        """Format steps for output."""
        return [
            {
                "step": s.step_number,
                "type": s.step_type.value,
                "content": s.content,
                "tool": s.tool_name,
                "success": s.success,
            }
            for s in steps
        ]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full execution history."""
        return [
            {
                "task": h.task,
                "success": h.finished,
                "iterations": h.current_step,
                "final_answer": h.final_answer,
            }
            for h in self.history
        ]


class PlanAndExecuteAgent:
    """
    Advanced agent that creates a plan first, then executes with ReAct.
    """
    
    def __init__(self, llm, tools: Dict[str, Callable], max_plan_steps: int = 5):
        self.llm = llm
        self.tools = tools
        self.max_plan_steps = max_plan_steps
        self.react_agent = ReActAgent(llm, tools, max_iterations=10)
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        Execute task with planning + ReAct execution.
        
        Returns:
            Dict with plan, execution results, and final answer
        """
        # Step 1: Create plan
        plan = self._create_plan(task)
        
        # Step 2: Execute each plan item with ReAct
        results = []
        context = {}
        
        for i, plan_item in enumerate(plan):
            print(f"\n📝 Executing plan step {i+1}/{len(plan)}: {plan_item}")
            
            result = self.react_agent.run(
                task=plan_item,
                context=context,
            )
            
            results.append({
                "plan_step": plan_item,
                "result": result,
            })
            
            # Update context with results
            if result["success"]:
                context[f"step_{i}_result"] = result["final_answer"]
            else:
                # Plan item failed - need to replan
                print(f"❌ Step failed, replanning...")
                new_plan = self._replan(task, plan, i, result, context)
                plan = plan[:i+1] + new_plan
        
        # Compile final answer
        final_answer = self._compile_results(task, results)
        
        return {
            "success": all(r["result"]["success"] for r in results),
            "plan": plan,
            "execution_results": results,
            "final_answer": final_answer,
        }
    
    def _create_plan(self, task: str) -> List[str]:
        """Create execution plan."""
        prompt = f"""Create a step-by-step plan to accomplish this task.

Task: {task}

Available tools: {', '.join(self.tools.keys())}

Create a plan with at most {self.max_plan_steps} steps.
Each step should be a simple, actionable task.

Format your response as a JSON list:
["step 1", "step 2", "step 3"]
"""
        
        response = self.llm.chat(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Try to parse JSON
        try:
            # Find JSON array in response
            import re
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback: parse line by line
        lines = [l.strip().lstrip("0123456789.- ") for l in text.split('\n') if l.strip()]
        return lines[:self.max_plan_steps]
    
    def _replan(
        self,
        original_task: str,
        current_plan: List[str],
        failed_step: int,
        failure_result: Dict,
        context: Dict,
    ) -> List[str]:
        """Replan after failure."""
        prompt = f"""The current plan failed. Create a new plan to complete the task.

Original task: {original_task}
Current plan: {current_plan}
Failed at step: {failed_step + 1}
Failure reason: {failure_result.get('final_answer', 'Unknown')}

Context from previous steps:
{json.dumps(context, indent=2)}

Create a new plan (JSON list) to complete the remaining task.
"""
        
        response = self.llm.chat(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        try:
            import re
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return ["Continue with alternative approach"]
    
    def _compile_results(self, task: str, results: List[Dict]) -> str:
        """Compile all results into final answer."""
        # If only one result, return it directly
        if len(results) == 1:
            return results[0]["result"]["final_answer"]
        
        # Otherwise, synthesize results
        synthesis_prompt = f"""Synthesize the following results into a coherent answer.

Original task: {task}

Results from each step:
"""
        
        for i, r in enumerate(results):
            synthesis_prompt += f"\nStep {i+1} ({r['plan_step']}):\n{r['result']['final_answer']}\n"
        
        synthesis_prompt += "\nProvide a final, comprehensive answer:"
        
        response = self.llm.chat(synthesis_prompt)
        return response.text if hasattr(response, 'text') else str(response)


# Convenience functions
def create_react_agent(llm, tools: Dict[str, Callable]) -> ReActAgent:
    """Quick ReAct agent creation."""
    return ReActAgent(llm, tools)


def run_react(task: str, llm, tools: Dict[str, Callable]) -> Dict[str, Any]:
    """Quick ReAct execution."""
    agent = ReActAgent(llm, tools)
    return agent.run(task)
