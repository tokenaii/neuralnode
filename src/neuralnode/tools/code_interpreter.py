"""Code interpreter for executing Python code safely."""

import sys
import io
import contextlib
import traceback
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import subprocess
import tempfile
import os


@dataclass
class CodeResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class CodeInterpreter:
    """Safe Python code interpreter."""
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_size: int = 10000,
        allowed_modules: Optional[List[str]] = None,
        blocked_modules: Optional[List[str]] = None,
    ):
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allowed_modules = allowed_modules
        self.blocked_modules = blocked_modules or [
            'os.system', 'subprocess.call', 'subprocess.run',
            'eval', 'exec', 'compile', '__import__',
            'open', 'file', 'write', 'delete'
        ]
        self.execution_history: List[CodeResult] = []
        self.globals_dict: Dict[str, Any] = {}
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> CodeResult:
        """Execute Python code safely."""
        import time
        
        start_time = time.time()
        
        # Security check
        security_check = self._security_check(code)
        if not security_check["safe"]:
            return CodeResult(
                success=False,
                output="",
                error=f"Security violation: {security_check['reason']}",
                execution_time=time.time() - start_time,
            )
        
        # Setup execution environment
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        # Update globals with context
        if context:
            self.globals_dict.update(context)
        
        # Execute with timeout
        try:
            with self._timeout_context(self.timeout):
                # Capture stdout and stderr
                with contextlib.redirect_stdout(output_buffer), \
                     contextlib.redirect_stderr(error_buffer):
                    
                    exec(code, self.globals_dict)
                
                output = output_buffer.getvalue()
                error_output = error_buffer.getvalue()
                
                # Truncate if too long
                if len(output) > self.max_output_size:
                    output = output[:self.max_output_size] + "\n... (truncated)"
                
                result = CodeResult(
                    success=True,
                    output=output,
                    error=error_output if error_output else None,
                    variables=self._extract_variables(),
                    execution_time=time.time() - start_time,
                )
        
        except TimeoutError:
            result = CodeResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"Code execution timed out after {self.timeout} seconds",
                execution_time=time.time() - start_time,
            )
        
        except Exception as e:
            result = CodeResult(
                success=False,
                output=output_buffer.getvalue(),
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                execution_time=time.time() - start_time,
            )
        
        self.execution_history.append(result)
        return result
    
    def _security_check(self, code: str) -> Dict[str, Any]:
        """Check code for security violations."""
        # Check for blocked patterns
        for blocked in self.blocked_modules:
            if blocked in code:
                return {"safe": False, "reason": f"Blocked: {blocked}"}
        
        # Check for file operations
        dangerous_patterns = [
            'open(', 'file(', '.write(', '.delete(', 'os.remove',
            'shutil.rmtree', 'os.rmdir', 'os.unlink'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return {"safe": False, "reason": f"File operation not allowed: {pattern}"}
        
        return {"safe": True, "reason": None}
    
    def _timeout_context(self, timeout: int):
        """Context manager for timeout."""
        import signal
        
        def handler(signum, frame):
            raise TimeoutError()
        
        class TimeoutContext:
            def __enter__(ctx):
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout)
                return ctx
            
            def __exit__(ctx, exc_type, exc_val, exc_tb):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, signal.SIG_DFL)
                return False
        
        return TimeoutContext()
    
    def _extract_variables(self) -> Dict[str, Any]:
        """Extract user-defined variables."""
        # Filter out built-ins and modules
        excluded = set(dir(__builtins__)) if hasattr(__builtins__, '__iter__') else set()
        excluded.update(['__builtins__', '__doc__', '__file__', '__name__', '__package__'])
        
        variables = {}
        for name, value in self.globals_dict.items():
            if name not in excluded and not name.startswith('_'):
                try:
                    # Try to get a string representation
                    variables[name] = str(value)[:1000]  # Limit size
                except Exception:
                    variables[name] = f"<{type(value).__name__}>"
        
        return variables
    
    def install_package(self, package_name: str) -> bool:
        """Install a Python package."""
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package_name],
                check=True,
                capture_output=True,
                timeout=60,
            )
            return True
        except Exception as e:
            print(f"Error installing package: {e}")
            return False
    
    def reset(self) -> None:
        """Reset the interpreter state."""
        self.globals_dict.clear()
        self.execution_history.clear()
    
    def get_history(self) -> List[CodeResult]:
        """Get execution history."""
        return self.execution_history.copy()


class JupyterStyleInterpreter(CodeInterpreter):
    """Jupyter-style code interpreter with cell execution."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cells: List[Dict[str, Any]] = []
    
    def execute_cell(self, code: str, cell_id: Optional[str] = None) -> CodeResult:
        """Execute a code cell."""
        result = self.execute(code)
        
        cell = {
            'id': cell_id or f"cell_{len(self.cells)}",
            'code': code,
            'result': result,
        }
        self.cells.append(cell)
        
        return result
    
    def get_notebook(self) -> Dict[str, Any]:
        """Get full notebook with all cells."""
        return {
            'cells': [
                {
                    'id': c['id'],
                    'code': c['code'],
                    'output': c['result'].output,
                    'success': c['result'].success,
                }
                for c in self.cells
            ],
            'execution_count': len(self.cells),
        }


class SandboxedInterpreter(CodeInterpreter):
    """Fully sandboxed interpreter using subprocess."""
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> CodeResult:
        """Execute code in isolated subprocess."""
        import time
        import json
        
        start_time = time.time()
        
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write context variables
            if context:
                f.write("# Context variables\n")
                for name, value in context.items():
                    f.write(f"{name} = {repr(value)}\n")
                f.write("\n")
            
            # Write user code
            f.write(code)
            
            temp_path = f.name
        
        try:
            # Execute in subprocess with restrictions
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            output = result.stdout
            error = result.stderr if result.returncode != 0 else None
            
            # Truncate if too long
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + "\n... (truncated)"
            
            return CodeResult(
                success=result.returncode == 0,
                output=output,
                error=error,
                execution_time=time.time() - start_time,
            )
        
        except subprocess.TimeoutExpired:
            return CodeResult(
                success=False,
                output="",
                error=f"Code execution timed out after {self.timeout} seconds",
                execution_time=time.time() - start_time,
            )
        
        except Exception as e:
            return CodeResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
            )
        
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except Exception:
                pass


# Convenience functions
def execute_python(code: str, timeout: int = 30) -> CodeResult:
    """Quick Python code execution."""
    interpreter = CodeInterpreter(timeout=timeout)
    return interpreter.execute(code)


def execute_safely(code: str, timeout: int = 30) -> CodeResult:
    """Execute code in full sandbox."""
    interpreter = SandboxedInterpreter(timeout=timeout)
    return interpreter.execute(code)
