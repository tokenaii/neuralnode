"""Auto-Agent that builds and improves itself automatically."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Capability:
    """A capability the agent can acquire."""
    name: str
    description: str
    code: str
    dependencies: List[str]
    success_rate: float = 0.0
    usage_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Improvement:
    """An improvement made to the agent."""
    timestamp: datetime
    type: str  # 'new_capability', 'optimized', 'bug_fix'
    description: str
    before: str
    after: str
    metrics: Dict[str, float]


class AutoAgent:
    """Self-building agent that creates its own tools."""
    
    def __init__(self, llm, initial_capabilities: Optional[List[Capability]] = None):
        self.llm = llm
        self.capabilities: Dict[str, Capability] = {}
        self.improvements: List[Improvement] = []
        self.learning_history: List[Dict[str, Any]] = []
        
        # Add initial capabilities
        if initial_capabilities:
            for cap in initial_capabilities:
                self.add_capability(cap)
    
    def add_capability(self, capability: Capability) -> None:
        """Add a new capability."""
        self.capabilities[capability.name] = capability
    
    def learn_from_task(self, task_description: str, examples: List[Dict[str, Any]]) -> Capability:
        """Learn a new capability from task examples."""
        print(f"Learning new capability from: {task_description}")
        
        # Analyze task
        prompt = f"""
Task: {task_description}

Examples:
{json.dumps(examples, indent=2)}

Create a Python function that can handle this task. The function should:
1. Be self-contained
2. Handle errors gracefully
3. Return clear results

Provide the function code and a brief description.
"""
        
        response = self.llm.chat(prompt)
        code = response.text if hasattr(response, 'text') else str(response)
        
        # Extract function name and create capability
        capability_name = self._extract_function_name(code) or "learned_function"
        
        capability = Capability(
            name=capability_name,
            description=task_description,
            code=code,
            dependencies=self._extract_dependencies(code),
            success_rate=0.5,  # Initial estimate
        )
        
        self.add_capability(capability)
        
        # Record improvement
        self.improvements.append(Improvement(
            timestamp=datetime.now(),
            type="new_capability",
            description=f"Learned: {task_description}",
            before="",
            after=code,
            metrics={"examples_used": len(examples)},
        ))
        
        return capability
    
    def optimize_capability(self, capability_name: str, test_cases: List[Dict]) -> Capability:
        """Optimize an existing capability."""
        if capability_name not in self.capabilities:
            raise ValueError(f"Capability {capability_name} not found")
        
        cap = self.capabilities[capability_name]
        
        prompt = f"""
Optimize this Python function:

Current code:
{cap.code}

Test cases:
{json.dumps(test_cases, indent=2)}

Improve the code to:
1. Run faster
2. Handle edge cases
3. Use better algorithms

Provide the optimized code.
"""
        
        response = self.llm.chat(prompt)
        optimized_code = response.text if hasattr(response, 'text') else str(response)
        
        # Update capability
        old_code = cap.code
        cap.code = optimized_code
        cap.success_rate = self._test_capability(cap, test_cases)
        
        # Record improvement
        self.improvements.append(Improvement(
            timestamp=datetime.now(),
            type="optimized",
            description=f"Optimized: {capability_name}",
            before=old_code,
            after=optimized_code,
            metrics={"success_rate": cap.success_rate},
        ))
        
        return cap
    
    def create_tool_for_task(self, task: str) -> Callable:
        """Automatically create a tool for a specific task."""
        # Check if similar capability exists
        similar = self._find_similar_capability(task)
        
        if similar and similar.success_rate > 0.8:
            # Use existing capability
            return self._create_callable(similar)
        
        # Learn new capability
        examples = self._generate_examples(task)
        capability = self.learn_from_task(task, examples)
        
        return self._create_callable(capability)
    
    def self_improve(self, performance_data: List[Dict[str, Any]]) -> List[Improvement]:
        """Analyze performance and self-improve."""
        improvements_made = []
        
        for data in performance_data:
            capability_name = data.get("capability")
            success = data.get("success", False)
            error = data.get("error")
            
            if capability_name in self.capabilities:
                cap = self.capabilities[capability_name]
                cap.usage_count += 1
                
                # Update success rate
                cap.success_rate = (
                    (cap.success_rate * (cap.usage_count - 1) + (1.0 if success else 0.0))
                    / cap.usage_count
                )
                
                # If failing, try to fix
                if not success and cap.success_rate < 0.7:
                    fix = self._fix_capability(cap, error)
                    if fix:
                        improvements_made.append(fix)
        
        return improvements_made
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import dependencies from code."""
        import re
        imports = re.findall(r'(?:from|import)\s+(\w+)', code)
        return list(set(imports))
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code."""
        import re
        match = re.search(r'def\s+(\w+)\s*\(', code)
        return match.group(1) if match else None
    
    def _extract_dependencies(self, code: str) -> List[str]:
        """Extract import dependencies from code."""
        import re
        imports = re.findall(r'(?:from|import)\s+(\w+)', code)
        return list(set(imports))
    
    def _find_similar_capability(self, task: str) -> Optional[Capability]:
        """Find a capability similar to the task."""
        # Simple keyword matching
        task_keywords = set(task.lower().split())
        
        best_match = None
        best_score = 0
        
        for cap in self.capabilities.values():
            cap_keywords = set(cap.description.lower().split())
            overlap = len(task_keywords & cap_keywords)
            score = overlap / max(len(task_keywords), len(cap_keywords))
            
            if score > best_score and score > 0.5:
                best_score = score
                best_match = cap
        
        return best_match
    
    def _generate_examples(self, task: str) -> List[Dict[str, Any]]:
        """Generate training examples for a task."""
        prompt = f"""
Generate 3 examples for this task:
{task}

Format each as:
- Input: <input>
- Expected output: <output>
"""
        
        response = self.llm.chat(prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Parse examples (simplified)
        examples = []
        # Would parse structured output
        
        return examples or [{"input": "example", "output": "result"}]
    
    def _create_callable(self, capability: Capability) -> Callable:
        """Create a callable from capability code."""
        # Execute code in safe namespace
        namespace = {}
        exec(capability.code, namespace)
        
        # Find the function
        func_name = self._extract_function_name(capability.code)
        if func_name and func_name in namespace:
            return namespace[func_name]
        
        # Return placeholder if no function found
        return lambda **kwargs: f"[Capability: {capability.name}]"
    
    def _test_capability(self, capability: Capability, test_cases: List[Dict]) -> float:
        """Test capability and return success rate."""
        if not test_cases:
            return 0.5
        
        passed = 0
        for test in test_cases:
            try:
                func = self._create_callable(capability)
                result = func(**test.get("input", {}))
                if result == test.get("expected"):
                    passed += 1
            except Exception:
                pass
        
        return passed / len(test_cases)
    
    def _fix_capability(self, capability: Capability, error: str) -> Optional[Improvement]:
        """Fix a failing capability."""
        prompt = f"""
Fix this code that has an error:

Code:
{capability.code}

Error:
{error}

Provide the corrected code.
"""
        
        try:
            response = self.llm.chat(prompt)
            fixed_code = response.text if hasattr(response, 'text') else str(response)
            
            old_code = capability.code
            capability.code = fixed_code
            
            improvement = Improvement(
                timestamp=datetime.now(),
                type="bug_fix",
                description=f"Fixed: {capability.name}",
                before=old_code,
                after=fixed_code,
                metrics={},
            )
            
            self.improvements.append(improvement)
            return improvement
        
        except Exception as e:
            print(f"Failed to fix capability: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "capabilities": len(self.capabilities),
            "improvements": len(self.improvements),
            "avg_success_rate": sum(c.success_rate for c in self.capabilities.values()) / len(self.capabilities) if self.capabilities else 0,
            "total_usage": sum(c.usage_count for c in self.capabilities.values()),
            "recent_improvements": [
                {
                    "type": i.type,
                    "description": i.description,
                    "timestamp": i.timestamp.isoformat(),
                }
                for i in self.improvements[-10:]
            ],
        }


class SelfModifyingAgent(AutoAgent):
    """Agent that can modify its own code."""
    
    def __init__(self, llm, code_file: Optional[str] = None):
        super().__init__(llm)
        self.code_file = code_file
    
    def add_capability_with_persistence(self, capability: Capability) -> None:
        """Add capability and save to file."""
        super().add_capability(capability)
        
        if self.code_file:
            self._save_capabilities()
    
    def _save_capabilities(self) -> None:
        """Save capabilities to file."""
        if not self.code_file:
            return
        
        data = {
            "capabilities": [
                {
                    "name": c.name,
                    "description": c.description,
                    "code": c.code,
                    "dependencies": c.dependencies,
                    "success_rate": c.success_rate,
                }
                for c in self.capabilities.values()
            ]
        }
        
        with open(self.code_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_capabilities(self) -> None:
        """Load capabilities from file."""
        if not self.code_file:
            return
        
        try:
            with open(self.code_file, 'r') as f:
                data = json.load(f)
            
            for cap_data in data.get("capabilities", []):
                cap = Capability(
                    name=cap_data["name"],
                    description=cap_data["description"],
                    code=cap_data["code"],
                    dependencies=cap_data.get("dependencies", []),
                    success_rate=cap_data.get("success_rate", 0),
                )
                self.add_capability(cap)
        
        except FileNotFoundError:
            pass


# Convenience functions
def create_auto_agent(llm, initial_tools: Optional[List[str]] = None) -> AutoAgent:
    """Quick auto-agent creation."""
    initial_capabilities = []
    
    if initial_tools:
        for tool in initial_tools:
            cap = Capability(
                name=tool,
                description=f"Tool: {tool}",
                code=f"def {tool}(): pass",
                dependencies=[],
            )
            initial_capabilities.append(cap)
    
    return AutoAgent(llm, initial_capabilities)
