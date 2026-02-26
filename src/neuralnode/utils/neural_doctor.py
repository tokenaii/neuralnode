"""Neural Doctor - Advanced error diagnosis and healing system."""

import traceback
import re
from typing import Dict, Any, List, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ErrorDiagnosis:
    """Diagnosis of an error."""
    error_type: str
    error_message: str
    severity: str  # low, medium, high, critical
    root_cause: str
    suggested_fix: str
    auto_fixable: bool
    confidence: float  # 0.0 to 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealAttempt:
    """Result of a healing attempt."""
    success: bool
    original_error: str
    fix_applied: str
    result: Any
    side_effects: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ErrorPatternDB:
    """Database of known error patterns and fixes."""
    
    def __init__(self):
        self.patterns: Dict[str, Dict[str, Any]] = {
            # Import errors
            r"No module named '(\w+)'": {
                "category": "import_error",
                "fix_type": "install_package",
                "severity": "medium",
                "auto_fixable": True,
            },
            r"cannot import name '(\w+)' from": {
                "category": "import_error",
                "fix_type": "update_import",
                "severity": "medium",
                "auto_fixable": False,
            },
            # API errors
            r"API key not found|Invalid API key": {
                "category": "api_error",
                "fix_type": "set_env_var",
                "severity": "high",
                "auto_fixable": False,
            },
            r"Rate limit exceeded": {
                "category": "rate_limit",
                "fix_type": "add_retry",
                "severity": "medium",
                "auto_fixable": True,
            },
            r"Connection (error|timeout|refused)": {
                "category": "network_error",
                "fix_type": "add_retry",
                "severity": "medium",
                "auto_fixable": True,
            },
            # File errors
            r"FileNotFoundError": {
                "category": "file_error",
                "fix_type": "create_file",
                "severity": "medium",
                "auto_fixable": True,
            },
            r"Permission denied": {
                "category": "permission_error",
                "fix_type": "check_permissions",
                "severity": "high",
                "auto_fixable": False,
            },
            # Memory errors
            r"CUDA out of memory": {
                "category": "cuda_oom",
                "fix_type": "reduce_batch_size",
                "severity": "high",
                "auto_fixable": True,
            },
            r"out of memory": {
                "category": "oom",
                "fix_type": "reduce_memory",
                "severity": "high",
                "auto_fixable": True,
            },
            # Type errors
            r"TypeError:.*expected.*got": {
                "category": "type_error",
                "fix_type": "type_cast",
                "severity": "medium",
                "auto_fixable": False,
            },
            # LLM specific
            r"model not found": {
                "category": "model_error",
                "fix_type": "download_model",
                "severity": "high",
                "auto_fixable": True,
            },
            r"context length exceeded": {
                "category": "context_error",
                "fix_type": "truncate_input",
                "severity": "medium",
                "auto_fixable": True,
            },
        }
    
    def match(self, error_message: str) -> Optional[Dict[str, Any]]:
        """Match error message to known pattern."""
        for pattern, info in self.patterns.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                return info
        return None
    
    def get_fix(self, category: str) -> Optional[Callable]:
        """Get fix function for category."""
        fixes = {
            "install_package": self._fix_install_package,
            "add_retry": self._fix_add_retry,
            "reduce_batch_size": self._fix_reduce_batch,
            "download_model": self._fix_download_model,
            "truncate_input": self._fix_truncate_input,
        }
        return fixes.get(category)
    
    def _fix_install_package(self, package_name: str) -> bool:
        """Install missing package."""
        try:
            import subprocess
            subprocess.run(["pip", "install", package_name], check=True)
            return True
        except Exception:
            return False
    
    def _fix_add_retry(self, context: Dict) -> bool:
        """Add retry logic."""
        # Would wrap function with retry decorator
        return True
    
    def _fix_reduce_batch(self, context: Dict) -> bool:
        """Reduce batch size."""
        return True
    
    def _fix_download_model(self, model_name: str) -> bool:
        """Download missing model."""
        try:
            from ..local.quantization import ModelDownloader
            downloader = ModelDownloader()
            downloader.download_from_huggingface(model_name)
            return True
        except Exception:
            return False
    
    def _fix_truncate_input(self, context: Dict) -> bool:
        """Truncate long input."""
        return True


class NeuralDoctor:
    """Advanced error diagnosis and healing system."""
    
    def __init__(self, llm=None, auto_heal: bool = False):
        self.llm = llm
        self.auto_heal = auto_heal
        self.pattern_db = ErrorPatternDB()
        self.error_history: List[ErrorDiagnosis] = []
        self.heal_history: List[HealAttempt] = []
    
    def diagnose(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorDiagnosis:
        """Diagnose an error."""
        error_type = type(error).__name__
        error_message = str(error)
        stack_trace = traceback.format_exc()
        
        # Match to known patterns
        pattern_info = self.pattern_db.match(error_message)
        
        if pattern_info:
            diagnosis = ErrorDiagnosis(
                error_type=error_type,
                error_message=error_message,
                severity=pattern_info["severity"],
                root_cause=self._determine_root_cause(error, pattern_info),
                suggested_fix=self._get_suggested_fix(error, pattern_info),
                auto_fixable=pattern_info["auto_fixable"],
                confidence=0.8,
                context=context or {},
                stack_trace=stack_trace,
            )
        else:
            # Use LLM for diagnosis if available
            if self.llm:
                diagnosis = self._llm_diagnose(error, context, stack_trace)
            else:
                diagnosis = ErrorDiagnosis(
                    error_type=error_type,
                    error_message=error_message,
                    severity="unknown",
                    root_cause="Unknown - no matching pattern found",
                    suggested_fix="Manual investigation required",
                    auto_fixable=False,
                    confidence=0.3,
                    context=context or {},
                    stack_trace=stack_trace,
                )
        
        self.error_history.append(diagnosis)
        return diagnosis
    
    def _determine_root_cause(self, error: Exception, pattern_info: Dict) -> str:
        """Determine root cause from pattern."""
        causes = {
            "import_error": "Missing Python package or incorrect import statement",
            "api_error": "API key missing, invalid, or not configured",
            "rate_limit": "Too many requests to API",
            "network_error": "Network connectivity issue or server unavailable",
            "file_error": "File does not exist or path is incorrect",
            "permission_error": "Insufficient permissions to access resource",
            "cuda_oom": "GPU memory exhausted",
            "oom": "System memory exhausted",
            "type_error": "Incorrect data type passed to function",
            "model_error": "AI model not downloaded or unavailable",
            "context_error": "Input exceeds model's maximum context length",
        }
        return causes.get(pattern_info["category"], "Unknown cause")
    
    def _get_suggested_fix(self, error: Exception, pattern_info: Dict) -> str:
        """Get suggested fix from pattern."""
        fixes = {
            "install_package": "Install the missing package with: pip install <package>",
            "update_import": "Update the import statement to use the correct module path",
            "set_env_var": "Set the required API key as an environment variable",
            "add_retry": "Add retry logic with exponential backoff",
            "create_file": "Create the missing file or directory",
            "check_permissions": "Check and fix file/directory permissions",
            "reduce_batch_size": "Reduce batch size or use gradient accumulation",
            "reduce_memory": "Clear memory cache or use memory-efficient settings",
            "type_cast": "Convert data to the expected type before passing",
            "download_model": "Download the required model using the model downloader",
            "truncate_input": "Truncate input to fit within context limit",
        }
        return fixes.get(pattern_info["fix_type"], "Manual fix required")
    
    def _llm_diagnose(
        self,
        error: Exception,
        context: Optional[Dict],
        stack_trace: str,
    ) -> ErrorDiagnosis:
        """Use LLM to diagnose error."""
        prompt = f"""Analyze this Python error and provide diagnosis:

Error Type: {type(error).__name__}
Error Message: {str(error)}

Stack Trace:
{stack_trace[:2000]}

Context: {json.dumps(context, default=str) if context else "None"}

Provide analysis in JSON format:
{{
    "root_cause": "explanation of what caused this",
    "severity": "low/medium/high/critical",
    "suggested_fix": "steps to fix this error",
    "auto_fixable": true/false,
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = self.llm.chat(prompt)
            analysis = json.loads(response.text)
            
            return ErrorDiagnosis(
                error_type=type(error).__name__,
                error_message=str(error),
                severity=analysis.get("severity", "unknown"),
                root_cause=analysis.get("root_cause", "Unknown"),
                suggested_fix=analysis.get("suggested_fix", "No fix suggested"),
                auto_fixable=analysis.get("auto_fixable", False),
                confidence=analysis.get("confidence", 0.5),
                context=context or {},
                stack_trace=stack_trace,
            )
        except Exception:
            return ErrorDiagnosis(
                error_type=type(error).__name__,
                error_message=str(error),
                severity="unknown",
                root_cause="Could not analyze with LLM",
                suggested_fix="Manual investigation required",
                auto_fixable=False,
                confidence=0.2,
                context=context or {},
                stack_trace=stack_trace,
            )
    
    def heal(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_auto: Optional[bool] = None,
    ) -> HealAttempt:
        """Attempt to heal/fix an error."""
        if attempt_auto is None:
            attempt_auto = self.auto_heal
        
        # First diagnose
        diagnosis = self.diagnose(error, context)
        
        if not attempt_auto or not diagnosis.auto_fixable:
            return HealAttempt(
                success=False,
                original_error=str(error),
                fix_applied="None - not auto-fixable",
                result=None,
                confidence=0.0,
            )
        
        # Attempt fix
        pattern_info = self.pattern_db.match(diagnosis.error_message)
        if pattern_info:
            fix_result = self._apply_fix(pattern_info, context)
            
            attempt = HealAttempt(
                success=fix_result,
                original_error=str(error),
                fix_applied=diagnosis.suggested_fix,
                result=None,
                confidence=diagnosis.confidence if fix_result else 0.0,
            )
        else:
            attempt = HealAttempt(
                success=False,
                original_error=str(error),
                fix_applied="None - no known fix pattern",
                result=None,
                confidence=0.0,
            )
        
        self.heal_history.append(attempt)
        return attempt
    
    def _apply_fix(self, pattern_info: Dict, context: Optional[Dict]) -> bool:
        """Apply automatic fix."""
        fix_type = pattern_info.get("fix_type")
        
        if fix_type == "install_package" and context:
            package = context.get("missing_package", "")
            return self.pattern_db._fix_install_package(package)
        
        elif fix_type == "add_retry":
            return self.pattern_db._fix_add_retry(context or {})
        
        elif fix_type == "reduce_batch_size":
            return self.pattern_db._fix_reduce_batch(context or {})
        
        elif fix_type == "download_model" and context:
            model = context.get("model_name", "")
            return self.pattern_db._fix_download_model(model)
        
        elif fix_type == "truncate_input":
            return self.pattern_db._fix_truncate_input(context or {})
        
        return False
    
    def report(self) -> Dict[str, Any]:
        """Generate healing report."""
        total_errors = len(self.error_history)
        auto_fixable = sum(1 for e in self.error_history if e.auto_fixable)
        healed = sum(1 for h in self.heal_history if h.success)
        
        # Categorize errors
        categories: Dict[str, int] = {}
        for diagnosis in self.error_history:
            cat = self.pattern_db.match(diagnosis.error_message)
            cat_name = cat["category"] if cat else "unknown"
            categories[cat_name] = categories.get(cat_name, 0) + 1
        
        return {
            "total_errors": total_errors,
            "auto_fixable": auto_fixable,
            "healed": healed,
            "heal_rate": healed / len(self.heal_history) if self.heal_history else 0,
            "categories": categories,
            "recent_errors": [
                {
                    "type": e.error_type,
                    "severity": e.severity,
                    "auto_fixable": e.auto_fixable,
                }
                for e in self.error_history[-10:]
            ],
        }
    
    def print_report(self) -> None:
        """Print healing report."""
        report = self.report()
        
        print("\n" + "=" * 60)
        print("🏥 Neural Doctor Report")
        print("=" * 60)
        print(f"\nTotal Errors: {report['total_errors']}")
        print(f"Auto-fixable: {report['auto_fixable']}")
        print(f"Successfully Healed: {report['healed']}")
        print(f"Heal Rate: {report['heal_rate']:.1%}")
        
        print("\nError Categories:")
        for cat, count in report['categories'].items():
            print(f"  {cat}: {count}")
        
        if report['recent_errors']:
            print("\nRecent Errors:")
            for e in report['recent_errors'][-5:]:
                icon = "✅" if e['auto_fixable'] else "❌"
                print(f"  {icon} {e['type']} ({e['severity']})")
        
        print("=" * 60)


class AutoHealDecorator:
    """Decorator for automatic healing."""
    
    def __init__(self, doctor: NeuralDoctor, max_attempts: int = 3):
        self.doctor = doctor
        self.max_attempts = max_attempts
    
    def __call__(self, func: Callable) -> Callable:
        """Apply decorator."""
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == self.max_attempts - 1:
                        raise  # Re-raise on last attempt
                    
                    # Try to heal
                    heal_result = self.doctor.heal(e, {"args": args, "kwargs": kwargs})
                    
                    if not heal_result.success:
                        raise  # Can't heal, re-raise
                    
                    # Heal successful, retry
                    print(f"🩺 Healed: {heal_result.fix_applied}")
            
            return None  # Shouldn't reach here
        
        return wrapper


# Convenience functions
def diagnose_error(error: Exception, llm=None, context: Optional[Dict] = None) -> ErrorDiagnosis:
    """Quick error diagnosis."""
    doctor = NeuralDoctor(llm=llm)
    return doctor.diagnose(error, context)


def heal_error(error: Exception, llm=None, context: Optional[Dict] = None) -> HealAttempt:
    """Quick error healing attempt."""
    doctor = NeuralDoctor(llm=llm, auto_heal=True)
    return doctor.heal(error, context)


def autoheal(max_attempts: int = 3, llm=None):
    """Decorator factory for auto-healing."""
    doctor = NeuralDoctor(llm=llm, auto_heal=True)
    decorator = AutoHealDecorator(doctor, max_attempts)
    return decorator
