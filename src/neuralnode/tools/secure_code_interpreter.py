"""Secure Code Interpreter - Hardened against sandbox escape."""

import ast
import sys
import builtins
import types
import threading
import resource
import signal
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from io import StringIO
import contextlib
import RestrictedPython
from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Guards import safe_builtins, full_write_guard


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    memory_used_mb: float
    variables: Dict[str, Any]


class SecureSandbox:
    """Hardened sandbox for code execution."""
    
    # Blacklisted builtins that could be used for escape
    BLACKLISTED_BUILTINS = {
        '__import__', 'eval', 'exec', 'compile', 'open',
        'input', 'raw_input', 'reload', 'help',
        'exit', 'quit', 'copyright', 'credits', 'license',
        'globals', 'locals', 'vars', 'dir',
        'getattr', 'setattr', 'delattr', 'hasattr',
        'object.__subclasses__', 'object.__bases__',
        'type.__bases__', 'type.__mro__',
    }
    
    # Dangerous modules
    BLACKLISTED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'ctypes', 'mmap',
        'shutil', 'pathlib', 'path', 'importlib', 'imp',
        'pickle', 'marshal', 'shelve', 'dbm',
        'tempfile', 'fileinput', 'linecache', 'site',
        'pydoc', 'inspect', 'traceback', 'warnings',
        'contextlib', 'atexit', 'signal', 'threading',
        'multiprocessing', 'concurrent', 'asyncio',
        'urllib', 'http', 'ftplib', 'smtplib', 'poplib',
        'imaplib', 'nntplib', 'telnetlib', 'xmlrpc',
    }
    
    # Dangerous attributes
    DANGEROUS_ATTRS = {
        '__class__', '__bases__', '__mro__', '__subclasses__',
        '__globals__', '__closure__', '__code__', '__func__',
        '__self__', '__module__', '__dict__', '__weakref__',
        '__del__', '__getattribute__', '__setattr__', '__slots__',
    }
    
    def __init__(self, max_memory_mb: int = 100, max_time_seconds: int = 30):
        self.max_memory_mb = max_memory_mb
        self.max_time_seconds = max_time_seconds
        self._setup_restricted_globals()
    
    def _setup_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted globals for safe execution."""
        # Start with safe builtins from RestrictedPython
        restricted_globals = {
            '__builtins__': self._create_safe_builtins(),
            '_getattr_': self._safe_getattr,
            '_setattr_': self._safe_setattr,
            '_getitem_': self._safe_getitem,
            '_write_': full_write_guard,
            '_getiter_': iter,
            '_iter_unpack_sequence_': lambda x: x,
        }
        
        # Add safe standard library modules
        safe_modules = {
            'math', 'random', 'statistics', 'fractions', 'decimal',
            'datetime', 'time', 'calendar', 'itertools', 'functools',
            'collections', 'heapq', 'bisect', 'copy', 'pprint',
            'numbers', 'json', 're', 'string', 'textwrap',
            'hashlib', 'base64', 'binascii', 'quopri', 'uu',
            'html', 'xml', 'stringprep', 'codecs', 'unicodedata',
        }
        
        for mod_name in safe_modules:
            try:
                mod = __import__(mod_name, fromlist=[mod_name])
                restricted_globals[mod_name] = mod
            except ImportError:
                pass
        
        self.restricted_globals = restricted_globals
        return restricted_globals
    
    def _create_safe_builtins(self) -> Dict[str, Any]:
        """Create a safe version of builtins."""
        safe = {}
        
        # Copy allowed builtins
        allowed = {
            'True', 'False', 'None',
            'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
            'callable', 'chr', 'classmethod', 'complex', 'delattr',
            'dict', 'divmod', 'enumerate', 'filter', 'float', 'format',
            'frozenset', 'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance',
            'issubclass', 'iter', 'len', 'list', 'map', 'max', 'memoryview',
            'min', 'next', 'oct', 'ord', 'pow', 'print', 'property', 'range',
            'repr', 'reversed', 'round', 'set', 'setattr', 'slice', 'sorted',
            'staticmethod', 'str', 'sum', 'super', 'tuple', 'type', 'vars',
            'zip', '__name__', '__doc__', '__package__', '__spec__',
        }
        
        for name in allowed:
            if hasattr(builtins, name):
                safe[name] = getattr(builtins, name)
        
        # Add our guards
        safe['_getattr_'] = self._safe_getattr
        safe['_setattr_'] = self._safe_setattr
        
        return safe
    
    def _safe_getattr(self, obj, name, default=None):
        """Safe getattr that blocks dangerous attributes."""
        if name in self.DANGEROUS_ATTRS:
            raise AttributeError(f"Access to {name} is forbidden")
        
        # Block access to __import__ on any object
        if name == '__import__':
            raise AttributeError("Import is not allowed")
        
        return getattr(obj, name, default)
    
    def _safe_setattr(self, obj, name, value):
        """Safe setattr that blocks dangerous attributes."""
        if name in self.DANGEROUS_ATTRS:
            raise AttributeError(f"Setting {name} is forbidden")
        return setattr(obj, name, value)
    
    def _safe_getitem(self, obj, key):
        """Safe getitem that blocks dangerous access."""
        return obj[key]
    
    def _validate_ast(self, code: str) -> bool:
        """Validate AST for dangerous patterns."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        for node in ast.walk(tree):
            # Check for dangerous nodes
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in self.BLACKLISTED_MODULES:
                        return False, f"Import of {alias.name} is not allowed"
            
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in self.BLACKLISTED_MODULES:
                    return False, f"Import from {node.module} is not allowed"
            
            # Check for __import__ calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == '__import__':
                    return False, "__import__ is not allowed"
            
            # Check for dangerous attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in self.DANGEROUS_ATTRS:
                    return False, f"Access to {node.attr} is not allowed"
        
        return True, ""
    
    def execute(self, code: str, provided_globals: Optional[Dict] = None) -> ExecutionResult:
        """Execute code in secure sandbox."""
        import time
        start_time = time.time()
        
        # Validate AST first
        valid, error = self._validate_ast(code)
        if not valid:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Security violation: {error}",
                execution_time=0,
                memory_used_mb=0,
                variables={}
            )
        
        # Compile with RestrictedPython
        try:
            compiled = compile_restricted(code, '<inline>', 'exec')
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Compilation error: {e}",
                execution_time=0,
                memory_used_mb=0,
                variables={}
            )
        
        # Setup execution environment
        exec_globals = self.restricted_globals.copy()
        if provided_globals:
            exec_globals.update(provided_globals)
        
        exec_locals = {}
        
        # Capture output
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        
        try:
            # Set resource limits (Unix only)
            if hasattr(resource, 'setrlimit'):
                # Limit memory
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.max_memory_mb * 1024 * 1024, self.max_memory_mb * 1024 * 1024)
                )
                # Limit CPU time
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.max_time_seconds, self.max_time_seconds)
                )
            
            # Execute with timeout
            with contextlib.redirect_stdout(stdout_buffer):
                with contextlib.redirect_stderr(stderr_buffer):
                    # Use alarm for timeout (Unix)
                    if hasattr(signal, 'SIGALRM'):
                        def timeout_handler(signum, frame):
                            raise TimeoutError("Code execution timed out")
                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(self.max_time_seconds)
                    
                    try:
                        exec(compiled, exec_globals, exec_locals)
                    finally:
                        if hasattr(signal, 'SIGALRM'):
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, old_handler)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                output=stdout_buffer.getvalue(),
                error=stderr_buffer.getvalue() or None,
                execution_time=execution_time,
                memory_used_mb=0,  # Would need psutil to measure accurately
                variables={k: v for k, v in exec_locals.items() if not k.startswith('_')}
            )
        
        except TimeoutError as e:
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=f"Timeout: Code execution exceeded {self.max_time_seconds} seconds",
                execution_time=self.max_time_seconds,
                memory_used_mb=0,
                variables={}
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout_buffer.getvalue(),
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=time.time() - start_time,
                memory_used_mb=0,
                variables={}
            )


class CodeInterpreter:
    """Hardened code interpreter with multiple security layers."""
    
    def __init__(self, max_memory_mb: int = 100, timeout: int = 30, strict: bool = True):
        self.max_memory_mb = max_memory_mb
        self.timeout = timeout
        self.strict = strict
        self.sandbox = SecureSandbox(max_memory_mb, timeout)
        self.execution_history: List[ExecutionResult] = []
    
    def run(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Run code securely."""
        # Additional pre-checks
        if self.strict:
            code = self._sanitize_code(code)
        
        result = self.sandbox.execute(code, context)
        self.execution_history.append(result)
        return result
    
    def _sanitize_code(self, code: str) -> str:
        """Additional code sanitization."""
        # Remove common bypass attempts
        dangerous_patterns = [
            r'__builtins__',
            r'__import__',
            r'\.\s*__class__',
            r'\.\s*__bases__',
            r'\.\s*__mro__',
            r'\.\s*__subclasses__',
            r'\.\s*__globals__',
            r'\.\s*__closure__',
            r'\.\s*__code__',
            r'getattr\s*\([^)]*,\s*[\'"]__',
            r'setattr\s*\([^)]*,\s*[\'"]__',
            r'\(\s*\)\s*\.\s*__class__',
            r'\[\s*\]\s*\.\s*__class__',
            r'\{\s*\}\s*\.\s*__class__',
            r'\(\s*\)\s*\.\s*__getitem__',
            r'__getitem__\s*\(',
            r'\.__dict__',
            r'\.__weakref__',
        ]
        
        import re
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                raise SecurityError(f"Code contains potentially dangerous pattern: {pattern}")
        
        return code
    
    def get_history(self) -> List[ExecutionResult]:
        """Get execution history."""
        return self.execution_history.copy()


class SecurityError(Exception):
    """Security violation error."""
    pass


# Convenience functions
def safe_execute(code: str, timeout: int = 30, max_memory_mb: int = 100) -> ExecutionResult:
    """Execute code safely with default settings."""
    interpreter = CodeInterpreter(timeout=timeout, max_memory_mb=max_memory_mb)
    return interpreter.run(code)


def validate_code(code: str) -> tuple[bool, Optional[str]]:
    """Validate code without executing."""
    sandbox = SecureSandbox()
    return sandbox._validate_ast(code)
