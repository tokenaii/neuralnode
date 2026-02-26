"""Docker-based secure code execution for AI agents."""

import logging
import tempfile
import os
import json
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import shutil

logger = logging.getLogger(__name__)


@dataclass
class DockerExecutionResult:
    """Result of Docker code execution."""
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    memory_usage_mb: float
    container_id: Optional[str] = None


class DockerCodeExecutor:
    """
    Secure code execution using Docker containers.
    Creates isolated, ephemeral containers for code execution.
    """
    
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "128m",
        cpu_limit: float = 0.5,
        network_disabled: bool = True,
        allowed_packages: Optional[List[str]] = None,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.allowed_packages = allowed_packages or [
            "numpy", "pandas", "requests", "json", "math", "random"
        ]
        self._docker_available = self._check_docker()
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        packages: Optional[List[str]] = None,
    ) -> DockerExecutionResult:
        """
        Execute Python code in a Docker container.
        
        Args:
            code: Python code to execute
            context: Variables to inject into the execution environment
            packages: Additional packages to install
        """
        import time
        
        start_time = time.time()
        
        if not self._docker_available:
            return DockerExecutionResult(
                success=False,
                output="",
                error="Docker not available. Please install Docker.",
                execution_time=0,
                memory_usage_mb=0,
            )
        
        container_id = None
        temp_dir = None
        
        try:
            # Create temporary directory for code
            temp_dir = tempfile.mkdtemp(prefix="neuralnode_docker_")
            
            # Prepare code file
            code_file = os.path.join(temp_dir, "script.py")
            with open(code_file, "w") as f:
                # Add security header
                f.write(self._get_security_header())
                
                # Add context variables
                if context:
                    f.write("\n# Context variables\n")
                    for name, value in context.items():
                        try:
                            f.write(f"{name} = {repr(value)}\n")
                        except Exception:
                            pass
                
                # Write user code
                f.write("\n# User code\n")
                f.write(code)
            
            # Create Dockerfile
            dockerfile = os.path.join(temp_dir, "Dockerfile")
            with open(dockerfile, "w") as f:
                f.write(self._create_dockerfile(packages))
            
            # Build image
            image_tag = f"neuralnode-exec-{uuid.uuid4().hex[:8]}"
            build_result = self._build_image(temp_dir, image_tag)
            
            if not build_result["success"]:
                return DockerExecutionResult(
                    success=False,
                    output="",
                    error=f"Build failed: {build_result['error']}",
                    execution_time=time.time() - start_time,
                    memory_usage_mb=0,
                )
            
            # Run container
            container_id = self._run_container(image_tag)
            
            if not container_id:
                return DockerExecutionResult(
                    success=False,
                    output="",
                    error="Failed to create container",
                    execution_time=time.time() - start_time,
                    memory_usage_mb=0,
                )
            
            # Wait for completion and get logs
            result = self._wait_for_container(container_id)
            
            # Get container stats
            memory_usage = self._get_container_memory(container_id)
            
            # Cleanup
            self._cleanup(container_id, image_tag)
            
            return DockerExecutionResult(
                success=result["success"],
                output=result["output"],
                error=result["error"],
                execution_time=time.time() - start_time,
                memory_usage_mb=memory_usage,
                container_id=container_id,
            )
            
        except Exception as e:
            logger.error(f"Docker execution failed: {e}", exc_info=True)
            
            # Cleanup on error
            if container_id:
                self._cleanup_container(container_id)
            
            return DockerExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                memory_usage_mb=0,
                container_id=container_id,
            )
        
        finally:
            # Cleanup temp directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _get_security_header(self) -> str:
        """Generate security header for Python code."""
        return '''
import sys
import os

# Security restrictions
sys.path = []
os.environ.clear()

# Block dangerous imports
blocked_modules = [
    "os", "sys", "subprocess", "socket", "ctypes", 
    "platform", "pwd", "grp", "shutil", "pathlib"
]

for mod in blocked_modules:
    sys.modules[mod] = None

# Limit built-ins
allowed_builtins = [
    "True", "False", "None", "abs", "all", "any", "ascii",
    "bin", "bool", "bytearray", "bytes", "chr", "complex",
    "dict", "dir", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "hasattr", "hash", "hex", "id",
    "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "memoryview", "min", "next", "oct", "ord",
    "pow", "print", "property", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "staticmethod", "str",
    "sum", "tuple", "type", "vars", "zip"
]

__builtins__ = {k: v for k, v in __builtins__.items() if k in allowed_builtins}
'''
    
    def _create_dockerfile(self, packages: Optional[List[str]]) -> str:
        """Create Dockerfile for code execution."""
        dockerfile = f"""FROM {self.image}

# Create non-root user
RUN useradd -m -u 1000 executor && \\
    mkdir -p /app && \\
    chown executor:executor /app

WORKDIR /app

# Install allowed packages
"""
        # Install requested packages
        if packages:
            for pkg in packages:
                if pkg in self.allowed_packages:
                    dockerfile += f"RUN pip install --no-cache-dir {pkg}\n"
        
        # Add security hardening
        dockerfile += """
# Copy and set permissions
COPY --chown=executor:executor script.py /app/

# Switch to non-root user
USER executor

# Run with resource limits
CMD ["python", "-u", "script.py"]
"""
        return dockerfile
    
    def _build_image(self, context_dir: str, tag: str) -> Dict[str, Any]:
        """Build Docker image."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "build", "-t", tag, "."],
                cwd=context_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
            }
            
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Build timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _run_container(self, image_tag: str) -> Optional[str]:
        """Run Docker container and return container ID."""
        import subprocess
        
        try:
            # Build run command with security options
            cmd = [
                "docker", "run", "-d",
                "--memory", self.memory_limit,
                "--memory-swap", self.memory_limit,
                "--cpus", str(self.cpu_limit),
                "--read-only",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--network", "none" if self.network_disabled else "bridge",
                "--rm",
                image_tag,
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                logger.info(f"Started container: {container_id[:12]}")
                return container_id
            else:
                logger.error(f"Failed to start container: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting container: {e}")
            return None
    
    def _wait_for_container(self, container_id: str) -> Dict[str, Any]:
        """Wait for container to finish and get results."""
        import subprocess
        import time
        
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Check if container is still running
                result = subprocess.run(
                    ["docker", "inspect", "-f", "{{.State.Status}}", container_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                
                if result.returncode == 0:
                    status = result.stdout.strip()
                    
                    if status == "exited":
                        # Container finished, get logs
                        logs_result = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        
                        # Get exit code
                        exit_result = subprocess.run(
                            ["docker", "inspect", "-f", "{{.State.ExitCode}}", container_id],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        
                        exit_code = int(exit_result.stdout.strip()) if exit_result.returncode == 0 else -1
                        
                        return {
                            "success": exit_code == 0,
                            "output": logs_result.stdout,
                            "error": logs_result.stderr if exit_code != 0 else None,
                        }
                
                # Wait a bit before checking again
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error waiting for container: {e}")
                return {"success": False, "output": "", "error": str(e)}
        
        # Timeout reached
        logger.warning(f"Container {container_id[:12]} timed out")
        return {
            "success": False,
            "output": "",
            "error": f"Execution timed out after {self.timeout} seconds",
        }
    
    def _get_container_memory(self, container_id: str) -> float:
        """Get container memory usage."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", container_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                # Parse memory usage (format: "10.5MiB / 128MiB")
                mem_str = result.stdout.strip().split()[0]
                if "MiB" in mem_str:
                    return float(mem_str.replace("MiB", ""))
                elif "GiB" in mem_str:
                    return float(mem_str.replace("GiB", "")) * 1024
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _cleanup(self, container_id: str, image_tag: str) -> None:
        """Clean up container and image."""
        # Kill container if still running
        self._cleanup_container(container_id)
        
        # Remove image
        try:
            import subprocess
            subprocess.run(
                ["docker", "rmi", "-f", image_tag],
                capture_output=True,
                timeout=30,
            )
            logger.info(f"Removed image: {image_tag}")
        except Exception as e:
            logger.warning(f"Failed to remove image {image_tag}: {e}")
    
    def _cleanup_container(self, container_id: str) -> None:
        """Kill and remove container."""
        try:
            import subprocess
            subprocess.run(
                ["docker", "kill", container_id],
                capture_output=True,
                timeout=10,
            )
            logger.info(f"Killed container: {container_id[:12]}")
        except Exception:
            pass


class SecureCodeExecutor:
    """
    Unified secure code execution interface.
    Uses Docker if available, falls back to subprocess with restrictions.
    """
    
    def __init__(self, prefer_docker: bool = True):
        self.prefer_docker = prefer_docker
        self.docker_executor = None
        
        if prefer_docker:
            self.docker_executor = DockerCodeExecutor()
            if not self.docker_executor._docker_available:
                logger.warning("Docker not available, using subprocess fallback")
                self.docker_executor = None
    
    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        packages: Optional[List[str]] = None,
    ) -> DockerExecutionResult:
        """Execute code securely."""
        if self.docker_executor:
            return self.docker_executor.execute(code, context, packages)
        else:
            return self._execute_subprocess(code, context)
    
    def _execute_subprocess(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DockerExecutionResult:
        """Fallback execution using subprocess with restrictions."""
        import subprocess
        import tempfile
        import time
        import shutil
        
        start_time = time.time()
        temp_dir = None
        
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix="neuralnode_subprocess_")
            
            # Write code
            code_file = os.path.join(temp_dir, "script.py")
            with open(code_file, "w") as f:
                f.write(self.docker_executor._get_security_header() if self.docker_executor else "")
                
                if context:
                    f.write("\n# Context\n")
                    for name, value in context.items():
                        try:
                            f.write(f"{name} = {repr(value)}\n")
                        except Exception:
                            pass
                
                f.write("\n# User code\n")
                f.write(code)
            
            # Run with restrictions
            result = subprocess.run(
                ["python", "-B", "-I", code_file],  # -B no bytecode, -I isolated
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            return DockerExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None,
                execution_time=time.time() - start_time,
                memory_usage_mb=0,
            )
            
        except subprocess.TimeoutExpired:
            return DockerExecutionResult(
                success=False,
                output="",
                error="Execution timed out",
                execution_time=time.time() - start_time,
                memory_usage_mb=0,
            )
        except Exception as e:
            return DockerExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                memory_usage_mb=0,
            )
        
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)


# Convenience function
def execute_code_securely(
    code: str,
    context: Optional[Dict[str, Any]] = None,
    use_docker: bool = True,
) -> DockerExecutionResult:
    """Execute code with best available security."""
    executor = SecureCodeExecutor(prefer_docker=use_docker)
    return executor.execute(code, context)
