"""Self-Healing system for automatic code repair and error recovery."""

import ast
import traceback
import re
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodeIssue:
    """Detected code issue."""
    line_number: int
    column: int
    issue_type: str
    severity: str  # error, warning, info
    message: str
    suggestion: str
    auto_fixable: bool


@dataclass
class RepairAttempt:
    """Attempt to repair code."""
    original_code: str
    repaired_code: str
    issue: CodeIssue
    success: bool
    test_results: Dict[str, Any]
    timestamp: datetime


class StaticAnalyzer:
    """Static code analysis for Python code."""
    
    def analyze(self, code: str) -> List[CodeIssue]:
        """Analyze code and return list of issues."""
        issues = []
        
        # Syntax errors
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(CodeIssue(
                line_number=e.lineno or 1,
                column=e.offset or 0,
                issue_type="syntax_error",
                severity="error",
                message=str(e),
                suggestion="Check syntax near the indicated line",
                auto_fixable=False,
            ))
        
        # Common patterns
        issues.extend(self._check_common_issues(code))
        issues.extend(self._check_imports(code))
        issues.extend(self._check_variables(code))
        
        return issues
    
    def _check_common_issues(self, code: str) -> List[CodeIssue]:
        """Check for common coding issues."""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for bare except
            if re.match(r'^\s*except\s*:', line):
                issues.append(CodeIssue(
                    line_number=i,
                    column=0,
                    issue_type="bare_except",
                    severity="warning",
                    message="Bare except clause",
                    suggestion="Use 'except Exception:' instead of bare except",
                    auto_fixable=True,
                ))
            
            # Check for mutable default arguments
            if re.search(r'def\s+\w+\s*\([^)]*=\s*(\[|\{)', line):
                issues.append(CodeIssue(
                    line_number=i,
                    column=0,
                    issue_type="mutable_default",
                    severity="warning",
                    message="Mutable default argument",
                    suggestion="Use None as default and initialize inside function",
                    auto_fixable=True,
                ))
            
            # Check for == vs is with None
            if re.search(r'==\s*None', line) or re.search(r'!=\s*None', line):
                issues.append(CodeIssue(
                    line_number=i,
                    column=line.find('None') - 2,
                    issue_type="equality_with_none",
                    severity="warning",
                    message="Using == with None",
                    suggestion="Use 'is None' or 'is not None' instead",
                    auto_fixable=True,
                ))
        
        return issues
    
    def _check_imports(self, code: str) -> List[CodeIssue]:
        """Check for import issues."""
        issues = []
        
        # Check for unused imports (simplified)
        import_pattern = re.findall(r'^\s*(?:from\s+(\S+)\s+)?import\s+(\S+)', code, re.MULTILINE)
        
        for match in import_pattern:
            module = match[0] or match[1]
            # Check if module is used
            if module and module not in code.split('import')[1]:
                pass  # Would need more sophisticated analysis
        
        return issues
    
    def _check_variables(self, code: str) -> List[CodeIssue]:
        """Check for variable issues."""
        issues = []
        
        # Find undefined variables (simplified)
        undefined = re.findall(r'NameError.*\'(\w+)\'', traceback.format_exc())
        
        for var in undefined:
            # Would need AST analysis for accurate detection
            pass
        
        return issues


class AutoRepair:
    """Automatic code repair system."""
    
    def __init__(self, llm):
        self.llm = llm
        self.analyzer = StaticAnalyzer()
        self.repair_history: List[RepairAttempt] = []
    
    def repair(self, code: str, error_message: Optional[str] = None) -> Tuple[str, List[RepairAttempt]]:
        """Attempt to repair code."""
        # Analyze code
        issues = self.analyzer.analyze(code)
        
        if not issues and not error_message:
            return code, []
        
        repairs = []
        repaired_code = code
        
        # Try to auto-fix each issue
        for issue in issues:
            if issue.auto_fixable:
                new_code, attempt = self._auto_fix(repaired_code, issue)
                if attempt.success:
                    repaired_code = new_code
                    repairs.append(attempt)
        
        # If runtime error provided, use LLM to fix
        if error_message and not repairs:
            repaired_code, attempt = self._llm_repair(code, error_message)
            if attempt:
                repairs.append(attempt)
        
        return repaired_code, repairs
    
    def _auto_fix(self, code: str, issue: CodeIssue) -> Tuple[str, RepairAttempt]:
        """Auto-fix a specific issue."""
        lines = code.split('\n')
        line_idx = issue.line_number - 1
        
        if issue.issue_type == "bare_except":
            lines[line_idx] = lines[line_idx].replace('except:', 'except Exception:')
        
        elif issue.issue_type == "equality_with_none":
            lines[line_idx] = lines[line_idx].replace('== None', 'is None')
            lines[line_idx] = lines[line_idx].replace('!= None', 'is not None')
        
        repaired = '\n'.join(lines)
        
        attempt = RepairAttempt(
            original_code=code,
            repaired_code=repaired,
            issue=issue,
            success=True,
            test_results={},
            timestamp=datetime.now(),
        )
        
        return repaired, attempt
    
    def _llm_repair(self, code: str, error_message: str) -> Tuple[str, Optional[RepairAttempt]]:
        """Use LLM to repair code based on error."""
        prompt = f"""
Fix this Python code that has an error:

Original code:
```python
{code}
```

Error message:
{error_message}

Provide ONLY the corrected code, no explanations.
"""
        
        try:
            response = self.llm.chat(prompt)
            repaired = response.text if hasattr(response, 'text') else str(response)
            
            # Extract code from markdown if present
            if '```python' in repaired:
                repaired = re.search(r'```python\n(.*?)\n```', repaired, re.DOTALL).group(1)
            elif '```' in repaired:
                repaired = re.search(r'```\n(.*?)\n```', repaired, re.DOTALL).group(1)
            
            attempt = RepairAttempt(
                original_code=code,
                repaired_code=repaired,
                issue=CodeIssue(
                    line_number=0,
                    column=0,
                    issue_type="runtime_error",
                    severity="error",
                    message=error_message,
                    suggestion="Fixed by LLM",
                    auto_fixable=True,
                ),
                success=True,
                test_results={},
                timestamp=datetime.now(),
            )
            
            return repaired, attempt
        
        except Exception as e:
            print(f"LLM repair failed: {e}")
            return code, None
    
    def repair_and_test(
        self,
        code: str,
        test_fn: Callable,
        max_attempts: int = 3,
    ) -> Tuple[str, bool]:
        """Repair code and verify with tests."""
        current_code = code
        
        for attempt in range(max_attempts):
            # Try to repair
            repaired, repairs = self.repair(current_code)
            current_code = repaired
            
            # Test
            try:
                result = test_fn(current_code)
                if result:
                    return current_code, True
            except Exception as e:
                # Continue to next repair attempt
                current_code, _ = self._llm_repair(current_code, str(e))
        
        return current_code, False


class SelfHealingExecutor:
    """Executor with self-healing capabilities."""
    
    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.auto_repair = AutoRepair(llm)
        self.execution_log: List[Dict[str, Any]] = []
    
    def execute(self, code: str, globals_dict: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute code with self-healing."""
        current_code = code
        globals_dict = globals_dict or {}
        
        for attempt in range(self.max_retries):
            try:
                # Try to execute
                exec(current_code, globals_dict)
                
                self.execution_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "attempts": attempt + 1,
                    "final_code": current_code,
                })
                
                return {
                    "success": True,
                    "result": globals_dict.get("__result__"),
                    "attempts": attempt + 1,
                    "code": current_code,
                }
            
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                
                # Try to repair
                repaired, repairs = self.auto_repair.repair(current_code, error_msg)
                
                if repairs:
                    current_code = repaired
                    continue
                else:
                    # Could not repair
                    self.execution_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1,
                    })
                    
                    return {
                        "success": False,
                        "error": error_msg,
                        "attempts": attempt + 1,
                        "code": current_code,
                    }
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "attempts": self.max_retries,
            "code": current_code,
        }
    
    def get_repair_stats(self) -> Dict[str, Any]:
        """Get repair statistics."""
        total = len(self.execution_log)
        successful = sum(1 for e in self.execution_log if e.get("success"))
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "avg_attempts": sum(e.get("attempts", 0) for e in self.execution_log) / total if total > 0 else 0,
        }


# Convenience functions
def repair_code(code: str, error: Optional[str] = None, llm=None) -> str:
    """Quick code repair."""
    if llm:
        repair = AutoRepair(llm)
        repaired, _ = repair.repair(code, error)
        return repaired
    
    # Simple fixes without LLM
    analyzer = StaticAnalyzer()
    issues = analyzer.analyze(code)
    
    repaired = code
    for issue in issues:
        if issue.auto_fixable:
            lines = repaired.split('\n')
            line_idx = issue.line_number - 1
            
            if issue.issue_type == "bare_except":
                lines[line_idx] = lines[line_idx].replace('except:', 'except Exception:')
            
            repaired = '\n'.join(lines)
    
    return repaired


def self_healing_execute(code: str, llm=None, max_retries: int = 3) -> Dict[str, Any]:
    """Execute code with self-healing."""
    if llm:
        executor = SelfHealingExecutor(llm, max_retries)
        return executor.execute(code)
    
    # Basic execution without healing
    try:
        namespace = {}
        exec(code, namespace)
        return {"success": True, "result": namespace}
    except Exception as e:
        return {"success": False, "error": str(e)}
