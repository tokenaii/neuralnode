"""System control tools for NeuralNode - Secure Version."""

import os
import subprocess
import shlex
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ...core import BaseTool, ToolResult, ParameterSpec


class ExecuteCommandTool(BaseTool):
    """Execute system commands securely."""
    
    name = "execute_command"
    description = "Execute a system command in a secure sandbox. Only safe commands are allowed."
    parameters = {
        "command": ParameterSpec(
            type="string",
            description="The command to execute",
            required=True
        ),
        "timeout": ParameterSpec(
            type="integer",
            description="Maximum time to wait for command (seconds)",
            required=False,
            default=30
        )
    }
    
    # Safe commands whitelist
    SAFE_COMMANDS = {
        'dir', 'ls', 'cat', 'echo', 'pwd', 'cd', 'mkdir', 'cp', 'copy', 'mv', 'move',
        'find', 'grep', 'head', 'tail', 'wc', 'sort', 'uniq', 'clear', 'cls',
        'python', 'python3', 'pip', 'git', 'which', 'where', 'whoami', 'hostname',
        'date', 'time', 'df', 'du', 'file', 'id', 'uname', 'ver', 'systeminfo',
        'npm', 'node', 'pytest', 'flake8', 'black', 'mypy', 'curl', 'wget',
        'type', 'more', 'less', 'tree', 'ping'
    }
    
    # Dangerous patterns to block
    DANGEROUS_PATTERNS = [
        'rm -rf /', 'rm -rf /*', 'rmdir /s /q', 'del /f /s /q',
        'format', 'mkfs', 'fdisk', 'dd if=', 'shred', 'wipe',
        '>',' :(){ :|:& };:', 'shutdown', 'reboot', 'halt', 'poweroff',
        'reg delete', 'regedit /d'
    ]
    
    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        """Execute a command securely."""
        try:
            # Security checks
            is_safe, reason = self._check_safety(command)
            if not is_safe:
                return ToolResult.error(f"Command blocked for security: {reason}")
            
            # Parse command
            try:
                cmd_parts = shlex.split(command)
            except ValueError as e:
                return ToolResult.error(f"Invalid command syntax: {e}")
            
            if not cmd_parts:
                return ToolResult.error("Empty command")
            
            # Execute
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False  # Never use shell
            )
            
            # Format output
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            
            # Truncate long output
            if len(output) > 10000:
                output = output[:10000] + "\n\n[Output truncated...]"
            
            return ToolResult.ok(
                content=output,
                returncode=result.returncode,
                success=result.returncode == 0
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult.error(f"Command timed out after {timeout} seconds")
        except FileNotFoundError as e:
            return ToolResult.error(f"Command not found: {e}")
        except Exception as e:
            return ToolResult.error(f"Command execution failed: {str(e)}")
    
    def _check_safety(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        cmd_lower = command.lower().strip()
        
        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in cmd_lower:
                return False, f"Contains dangerous pattern: {pattern}"
        
        # Parse to get command name
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            cmd_name = parts[0].lower()
        except ValueError:
            return False, "Could not parse command"
        
        # Handle Windows .exe extension
        if cmd_name.endswith('.exe'):
            cmd_name = cmd_name[:-4]
        
        # Check if in safe list
        if cmd_name not in self.SAFE_COMMANDS:
            return False, f"Command '{cmd_name}' not in safe command list"
        
        # Check for shell metacharacters outside of allowed contexts
        dangerous_chars = [';', '&&', '||', '|', '`', '$(']
        for char in dangerous_chars:
            if char in command:
                return False, f"Shell metacharacter not allowed: {char}"
        
        # Check for output redirection that could overwrite files
        if '>' in command and cmd_name not in ['echo', 'cat', 'ls', 'dir', 'grep', 'find']:
            return False, "Output redirection not allowed for this command"
        
        return True, ""


class SystemInfoTool(BaseTool):
    """Get system information."""
    
    name = "system_info"
    description = "Get information about the operating system, hardware, and environment."
    parameters = {}  # No parameters needed
    
    def execute(self) -> ToolResult:
        """Get system information."""
        try:
            info = {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
            }
            
            # Try to get memory info if psutil available
            try:
                import psutil
                mem = psutil.virtual_memory()
                info["memory_total_gb"] = round(mem.total / (1024**3), 2)
                info["memory_available_gb"] = round(mem.available / (1024**3), 2)
                
                disk = psutil.disk_usage('/')
                info["disk_total_gb"] = round(disk.total / (1024**3), 2)
                info["disk_free_gb"] = round(disk.free / (1024**3), 2)
            except ImportError:
                pass
            
            # Format for display
            lines = ["System Information:", ""]
            for key, value in info.items():
                lines.append(f"  {key}: {value}")
            
            return ToolResult.ok(
                content="\n".join(lines),
                **info
            )
            
        except Exception as e:
            return ToolResult.error(f"Failed to get system info: {str(e)}")


# Note: FileReadTool, FileWriteTool, ListDirectoryTool have been moved to tools.advanced module
# Import from there: from neuralnode.tools.advanced import FileReadTool, FileWriteTool, ListDirectoryTool

from .operations import DateTimeTool, SystemOperationsTool


__all__ = [
    "ExecuteCommandTool",
    "SystemInfoTool",
    "DateTimeTool",
    "SystemOperationsTool",
]
