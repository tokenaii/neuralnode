"""System control tools for AI agents to control the computer - SECURE VERSION."""

import os
import subprocess
import shutil
import psutil
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import json
import platform
import shlex


@dataclass
class FileInfo:
    """File information."""
    name: str
    path: str
    size: int
    is_directory: bool
    modified_time: float
    created_time: float


@dataclass
class ProcessInfo:
    """Process process information."""
    pid: int
    name: str
    cpu_percent: float
    memory_mb: float
    status: str
    created_time: float


class FileManager:
    """File management tools - SECURITY HARDENED."""
    
    # Critical system paths that should NEVER be accessible
    BLOCKED_PATHS = [
        '/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin', '/lib', '/lib64',
        '/boot', '/dev', '/proc', '/sys', '/root',
        'C:\\Windows', 'C:\\Program Files', 'C:\\ProgramData',
        '~/.ssh', '~/.gnupg', '~/.aws', '~/.azure', '~/.config',
        '/var/log', '/var/spool', '/tmp', '/var/tmp',
    ]
    
    def __init__(self, base_path: Optional[str] = None, allow_home_access: bool = False):
        """
        Initialize FileManager with restricted access.
        
        Args:
            base_path: Base directory for all operations (default: current working dir)
            allow_home_access: Whether to allow access to home directory (default: False for security)
        """
        self.base_path = Path(base_path).resolve() if base_path else Path.cwd().resolve()
        self.allow_home_access = allow_home_access
        
        # Build allowed paths list - ONLY specific directories
        self.allowed_paths: List[Path] = [self.base_path]
        
        if allow_home_access:
            # Even with home access, restrict to safe subdirectories
            home = Path.home()
            self.allowed_paths.extend([
                home / "Documents",
                home / "Downloads",
                home / "Desktop",
            ])
    
    def _is_blocked(self, path: Path) -> bool:
        """Check if path is in blocked system directories."""
        path_str = str(path.resolve()).lower()
        
        for blocked in self.BLOCKED_PATHS:
            blocked_lower = blocked.lower().replace('~', str(Path.home()))
            if path_str.startswith(blocked_lower):
                return True
        
        # Check for path traversal attempts
        try:
            path.resolve().relative_to(self.base_path)
        except ValueError:
            # Path is outside base_path
            if not self.allow_home_access:
                return True
            # Check if in allowed home subdirectories
            home = Path.home()
            try:
                path.resolve().relative_to(home)
                # It's in home, check if in allowed subdirs
                for allowed in self.allowed_paths:
                    try:
                        path.resolve().relative_to(allowed)
                        return False  # Found in allowed path
                    except ValueError:
                        continue
                return True  # In home but not in allowed subdirs
            except ValueError:
                return True  # Outside both base and home
        
        return False
    
    def _is_allowed(self, path: Union[str, Path]) -> bool:
        """Check if path is in allowed directories - STRICT CHECK."""
        path = Path(path).resolve()
        
        # First check if explicitly blocked
        if self._is_blocked(path):
            return False
        
        # Check if within allowed paths
        for allowed in self.allowed_paths:
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue
        
        return False
    
    def read_file(self, path: Union[str, Path], limit: int = 10000) -> str:
        """Read file contents."""
        path = Path(path)
        
        if not path.exists():
            return f"Error: File not found: {path}"
        
        if not self._is_allowed(path):
            return "Error: Access denied - path not allowed"
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(limit)
                if len(content) == limit:
                    content += "\n... (truncated)"
                return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def write_file(self, path: Union[str, Path], content: str, append: bool = False) -> str:
        """Write content to file."""
        path = Path(path)
        
        if not self._is_allowed(path):
            return "Error: Access denied - path not allowed"
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    def delete_file(self, path: Union[str, Path]) -> str:
        """Delete file or directory."""
        path = Path(path)
        
        if not path.exists():
            return f"Error: Path not found: {path}"
        
        if not self._is_allowed(path):
            return "Error: Access denied"
        
        try:
            if path.is_dir():
                shutil.rmtree(path)
                return f"Deleted directory: {path}"
            else:
                path.unlink()
                return f"Deleted file: {path}"
        except Exception as e:
            return f"Error deleting: {str(e)}"
    
    def list_directory(self, path: Union[str, Path] = ".") -> List[FileInfo]:
        """List directory contents."""
        path = Path(path)
        
        if not path.exists():
            return []
        
        if not self._is_allowed(path):
            return []
        
        files = []
        try:
            for item in path.iterdir():
                stat = item.stat()
                files.append(FileInfo(
                    name=item.name,
                    path=str(item),
                    size=stat.st_size,
                    is_directory=item.is_dir(),
                    modified_time=stat.st_mtime,
                    created_time=stat.st_ctime,
                ))
        except PermissionError as e:
            logger.warning(f"Permission denied accessing directory {path}: {e}")
        
        return files
    
    def search_files(self, pattern: str, path: Union[str, Path] = ".") -> List[str]:
        """Search for files matching pattern."""
        path = Path(path)
        
        if not self._is_allowed(path):
            return []
        
        results = []
        try:
            for item in path.rglob(pattern):
                results.append(str(item))
        except Exception as e:
            logger.warning(f"Error searching files in {path}: {e}")
        
        return results
    
    def create_directory(self, path: Union[str, Path]) -> str:
        """Create directory."""
        path = Path(path)
        
        if not self._is_allowed(path):
            return "Error: Access denied"
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            return f"Created directory: {path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> str:
        """Move/rename file."""
        source = Path(source)
        destination = Path(destination)
        
        if not self._is_allowed(source) or not self._is_allowed(destination):
            return "Error: Access denied"
        
        try:
            shutil.move(str(source), str(destination))
            return f"Moved {source} to {destination}"
        except Exception as e:
            return f"Error moving file: {str(e)}"
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> str:
        """Copy file."""
        source = Path(source)
        destination = Path(destination)
        
        if not self._is_allowed(source) or not self._is_allowed(destination):
            return "Error: Access denied"
        
        try:
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
            return f"Copied {source} to {destination}"
        except Exception as e:
            return f"Error copying: {str(e)}"


class ProcessManager:
    """Process management tools."""
    
    def list_processes(self) -> List[ProcessInfo]:
        """List running processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status', 'create_time']):
            try:
                info = proc.info
                processes.append(ProcessInfo(
                    pid=info['pid'],
                    name=info['name'] or "unknown",
                    cpu_percent=info['cpu_percent'] or 0.0,
                    memory_mb=(info['memory_info'].rss / 1024 / 1024) if info['memory_info'] else 0.0,
                    status=info['status'] or "unknown",
                    created_time=info['create_time'] or 0,
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Process access error: {e}")
        
        return sorted(processes, key=lambda p: p.memory_mb, reverse=True)
    
    def kill_process(self, pid: int) -> str:
        """Kill a process by PID."""
        try:
            process = psutil.Process(pid)
            process.terminate()
            return f"Process {pid} terminated"
        except psutil.NoSuchProcess:
            return f"Process {pid} not found"
        except Exception as e:
            return f"Error killing process: {str(e)}"
    
    def get_process_info(self, pid: int) -> Optional[ProcessInfo]:
        """Get detailed process info."""
        try:
            proc = psutil.Process(pid)
            with proc.oneshot():
                return ProcessInfo(
                    pid=pid,
                    name=proc.name(),
                    cpu_percent=proc.cpu_percent(),
                    memory_mb=proc.memory_info().rss / 1024 / 1024,
                    status=proc.status(),
                    created_time=proc.create_time(),
                )
        except Exception:
            return None
    
    def find_process_by_name(self, name: str) -> List[ProcessInfo]:
        """Find processes by name."""
        matching = []
        for proc in self.list_processes():
            if name.lower() in proc.name.lower():
                matching.append(proc)
        return matching


class CommandRunner:
    """Execute system commands - SECURE VERSION without shell=True."""
    
    def __init__(self, allowed_commands: Optional[List[str]] = None, require_confirmation: bool = False, strict_mode: bool = True):
        self.allowed_commands = allowed_commands or []
        self.require_confirmation = require_confirmation
        self.strict_mode = strict_mode  # When True, only allow whitelisted commands
        
        # Whitelist of safe commands only (PRIMARY SECURITY MECHANISM)
        self.safe_commands = {
            'ls', 'cat', 'echo', 'pwd', 'cd', 'mkdir', 'cp', 'mv', 
            'find', 'grep', 'head', 'tail', 'wc', 'sort', 'uniq', 
            'python', 'pip', 'git', 'pytest', 'which', 'whoami',
            'date', 'df', 'du', 'file', 'id', 'uname'
        }
        
        # Additional allowed commands if user provides allowed_commands
        if self.allowed_commands:
            self.safe_commands.update(self.allowed_commands)
        
        # Blocked patterns that are ALWAYS dangerous (defense in depth)
        self.blocked_patterns = [
            'rm -rf /', 'rm -rf /*', ':(){:|:&};:', '> /dev/null', 
            '| bash', '| sh', 'chmod 777 /', 'mkfs'
        ]
    
    def _is_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe using WHITELIST approach.
        
        Returns (is_safe, reason).
        Security: Only allows explicitly whitelisted commands.
        """
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            
            executable = parts[0]
            
            # STRICT MODE: Only allow whitelisted commands
            if self.strict_mode:
                if executable not in self.safe_commands:
                    return False, f"Command '{executable}' not in whitelist. Allowed: {', '.join(sorted(self.safe_commands))}"
            
            # Defense in depth: Check blocked patterns even for whitelisted commands
            cmd_lower = command.lower()
            for blocked in self.blocked_patterns:
                if blocked.lower() in cmd_lower:
                    return False, f"Command contains dangerous pattern: {blocked}"
            
            # Check for shell injection characters
            dangerous_chars = [';', '&&', '||', '|', '`', '$', '<', '>']
            for char in dangerous_chars:
                if char in command and not self._is_quoted(command, char):
                    return False, f"Command contains shell metacharacter: {char}"
            
            return True, ""
            
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"
    
    def _is_quoted(self, command: str, char: str) -> bool:
        """Check if character is inside quotes."""
        in_single = False
        in_double = False
        for c in command:
            if c == "'" and not in_double:
                in_single = not in_single
            elif c == '"' and not in_single:
                in_double = not in_double
            elif c == char and not in_single and not in_double:
                return False
        return True
    
    def _get_user_confirmation(self, command: str) -> bool:
        """Get user confirmation before executing dangerous command."""
        print(f"\n⚠️  Command requires confirmation:")
        print(f"   {command}")
        response = input("Execute? [y/N]: ").lower().strip()
        return response in ('y', 'yes')
    
    def run(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
        shell: bool = False,  # CHANGED: Default is now False for security
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run a command securely without shell=True.
        
        Args:
            command: Command string to execute
            cwd: Working directory
            timeout: Timeout in seconds
            shell: DEPRECATED - Always uses safe execution regardless of this value
            env: Environment variables
        """
        # Security check
        is_safe, reason = self._is_safe(command)
        if not is_safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command blocked for security: {reason}",
                "returncode": -1,
            }
        
        # User confirmation for potentially dangerous operations
        if self.require_confirmation and self._needs_confirmation(command):
            if not self._get_user_confirmation(command):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Command cancelled by user",
                    "returncode": -1,
                }
        
        try:
            # Use shlex to safely parse the command
            cmd_parts = shlex.split(command)
            
            # Execute without shell for security
            result = subprocess.run(
                cmd_parts,
                cwd=cwd,
                shell=False,  # SECURE: Never use shell=True
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, **(env or {})},
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout[:10000],  # Limit output
                "stderr": result.stderr[:10000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
            }
        except FileNotFoundError as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command not found: {e}",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
    
    def _needs_confirmation(self, command: str) -> bool:
        """Check if command needs user confirmation."""
        dangerous_ops = ['rm', 'del', 'format', 'mkfs', 'dd', 'shred', '>', 'mv', 'chmod', 'chown']
        cmd_lower = command.lower()
        return any(op in cmd_lower for op in dangerous_ops)
    
    def run_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Run Python code safely in sandboxed environment."""
        import tempfile
        
        # Create isolated temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
            # Add safety header
            f.write("""
# Sandboxed execution - restricted imports
import sys
# Block dangerous imports
blocked_modules = ['os', 'subprocess', 'socket', 'ctypes', 'mmap']
for mod in blocked_modules:
    sys.modules[mod] = None
""")
            f.write(code)
            temp_path = f.name
        
        try:
            # Run with restricted python
            result = self.run(
                f"python -I -B {temp_path}",  # -I for isolated, -B for no bytecode
                timeout=timeout
            )
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
        
        return result


class SystemInfo:
    """System information tools."""
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive system info."""
        return {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
        }
    
    def get_memory(self) -> Dict[str, Any]:
        """Get memory info."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / 1024**3, 2),
            "available_gb": round(mem.available / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "percent": mem.percent,
        }
    
    def get_disk(self, path: str = "/") -> Dict[str, Any]:
        """Get disk usage."""
        disk = psutil.disk_usage(path)
        return {
            "total_gb": round(disk.total / 1024**3, 2),
            "used_gb": round(disk.used / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
            "percent": disk.percent,
        }
    
    def get_network(self) -> Dict[str, Any]:
        """Get network info."""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent_mb": round(net_io.bytes_sent / 1024**2, 2),
            "bytes_recv_mb": round(net_io.bytes_recv / 1024**2, 2),
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }


class SystemController:
    """Unified system controller for AI agents - SECURE VERSION."""
    
    def __init__(self, safe_mode: bool = True, require_confirmation: bool = False):
        self.files = FileManager()
        self.processes = ProcessManager()
        self.commands = CommandRunner(require_confirmation=require_confirmation)
        self.system = SystemInfo()
        self.safe_mode = safe_mode
        self.require_confirmation = require_confirmation
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for agent."""
        tools = [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "parameters": {
                    "path": {"type": "string", "description": "File path"},
                    "limit": {"type": "integer", "description": "Max characters to read"}
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                    "append": {"type": "boolean", "description": "Append to file"}
                },
            },
            {
                "name": "list_directory",
                "description": "List files in a directory",
                "parameters": {
                    "path": {"type": "string", "description": "Directory path"}
                },
            },
            {
                "name": "execute_command",
                "description": "Execute a system command (restricted, no shell)",
                "parameters": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds"}
                },
            },
            {
                "name": "get_system_info",
                "description": "Get system information",
                "parameters": {},
            },
        ]
        
        return tools
    
    def execute(self, tool_name: str, **params) -> str:
        """Execute a system tool."""
        if tool_name == "read_file":
            return self.files.read_file(params["path"], params.get("limit", 10000))
        
        elif tool_name == "write_file":
            return self.files.write_file(
                params["path"],
                params["content"],
                params.get("append", False)
            )
        
        elif tool_name == "list_directory":
            files = self.files.list_directory(params.get("path", "."))
            return json.dumps([
                {
                    "name": f.name,
                    "type": "directory" if f.is_directory else "file",
                    "size": f.size,
                }
                for f in files
            ], indent=2)
        
        elif tool_name == "execute_command":
            result = self.commands.run(
                params["command"],
                timeout=params.get("timeout", 60)
            )
            return json.dumps(result, indent=2)
        
        elif tool_name == "get_system_info":
            return json.dumps(self.system.get_info(), indent=2)
        
        return f"Unknown tool: {tool_name}"


# Convenience functions
def read_file(path: str, limit: int = 10000) -> str:
    """Quick file read."""
    return FileManager().read_file(path, limit)


def write_file(path: str, content: str) -> str:
    """Quick file write."""
    return FileManager().write_file(path, content)


def run_command(command: str, timeout: int = 60) -> Dict[str, Any]:
    """Quick command execution (SECURE - no shell)."""
    return CommandRunner().run(command, timeout=timeout, shell=False)


def get_system_info() -> Dict[str, Any]:
    """Quick system info."""
    return SystemInfo().get_info()
