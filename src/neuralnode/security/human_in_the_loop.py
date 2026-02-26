"""
Human-in-the-Loop system for sensitive operations.
Requires user approval before executing dangerous actions.
"""

from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json


class RiskLevel(Enum):
    """Risk levels for operations."""
    LOW = "low"           # Safe operations (read file, list dir)
    MEDIUM = "medium"     # Potential risk (write file, delete)
    HIGH = "high"         # Dangerous (execute command, delete system files)
    CRITICAL = "critical" # System critical (format disk, modify system)


@dataclass
class OperationRequest:
    """Request for sensitive operation."""
    operation_type: str
    description: str
    risk_level: RiskLevel
    details: Dict[str, Any]
    timestamp: datetime
    approved: Optional[bool] = None
    approver: Optional[str] = None
    denial_reason: Optional[str] = None


class HumanInTheLoop:
    """
    Human approval system for sensitive agent operations.
    """
    
    # Risk classification for different operations
    RISK_MATRIX = {
        # File operations
        "read_file": RiskLevel.LOW,
        "list_directory": RiskLevel.LOW,
        "search_files": RiskLevel.LOW,
        "write_file": RiskLevel.MEDIUM,
        "delete_file": RiskLevel.HIGH,
        "move_file": RiskLevel.MEDIUM,
        "copy_file": RiskLevel.LOW,
        
        # System operations
        "execute_command": RiskLevel.HIGH,
        "run_python": RiskLevel.HIGH,
        "kill_process": RiskLevel.HIGH,
        "system_info": RiskLevel.LOW,
        
        # Network operations
        "web_search": RiskLevel.LOW,
        "download_file": RiskLevel.MEDIUM,
        "send_request": RiskLevel.MEDIUM,
        
        # Code execution
        "execute_code": RiskLevel.CRITICAL,
        "install_package": RiskLevel.HIGH,
        
        # Email/Communication
        "send_email": RiskLevel.MEDIUM,
        "delete_email": RiskLevel.HIGH,
        "send_message": RiskLevel.MEDIUM,
        
        # Database
        "query_database": RiskLevel.LOW,
        "update_database": RiskLevel.MEDIUM,
        "delete_database": RiskLevel.CRITICAL,
    }
    
    # High-risk patterns in commands
    DANGEROUS_PATTERNS = [
        "rm -rf", "del /", "format", "fdisk", "mkfs",
        "drop database", "delete from", "truncate table",
        "shutdown", "reboot", "poweroff",
        "wget", "curl", "nc -l", "ncat",
        "> /etc/passwd", "> /etc/shadow",
        "chmod 777 /", "chown root",
    ]
    
    def __init__(
        self,
        auto_approve: List[str] = None,
        require_confirmation_for: List[RiskLevel] = None,
        approval_callback: Optional[Callable[[OperationRequest], bool]] = None,
    ):
        """
        Initialize HITL system.
        
        Args:
            auto_approve: List of operation types to auto-approve
            require_confirmation_for: Risk levels requiring approval
            approval_callback: Custom function to get approval
        """
        self.auto_approve = set(auto_approve or [])
        self.require_confirmation = set(require_confirmation_for or [RiskLevel.HIGH, RiskLevel.CRITICAL])
        self.approval_callback = approval_callback or self._default_approval
        self.history: List[OperationRequest] = []
        self.approved_count = 0
        self.denied_count = 0
    
    def check_operation(
        self,
        operation_type: str,
        details: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Check if operation requires approval and get it.
        
        Args:
            operation_type: Type of operation
            details: Operation details
        
        Returns:
            (approved, reason) - approved is bool, reason is None if approved
        """
        # Determine risk level
        risk_level = self._assess_risk(operation_type, details)
        
        # Check if auto-approved
        if operation_type in self.auto_approve:
            return True, None
        
        # Check if risk level requires confirmation
        if risk_level not in self.require_confirmation:
            return True, None
        
        # Create request
        request = OperationRequest(
            operation_type=operation_type,
            description=self._format_description(operation_type, details),
            risk_level=risk_level,
            details=details,
            timestamp=datetime.now(),
        )
        
        # Get approval
        approved = self.approval_callback(request)
        request.approved = approved
        
        if approved:
            request.approver = "user"
            self.approved_count += 1
        else:
            request.denial_reason = "User denied"
            self.denied_count += 1
        
        self.history.append(request)
        
        if approved:
            return True, None
        else:
            return False, f"Operation denied by user (Risk: {risk_level.value})"
    
    def _assess_risk(self, operation_type: str, details: Dict[str, Any]) -> RiskLevel:
        """Assess risk level of operation."""
        base_risk = self.RISK_MATRIX.get(operation_type, RiskLevel.LOW)
        
        # Check for dangerous patterns in details
        details_str = json.dumps(details).lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.lower() in details_str:
                # Elevate to HIGH if dangerous pattern found
                if base_risk.value in ["low", "medium"]:
                    base_risk = RiskLevel.HIGH
                break
        
        # Additional checks for specific operations
        if operation_type == "execute_command":
            command = details.get("command", "").lower()
            
            # Check for deletion commands
            if any(cmd in command for cmd in ["rm -rf", "del ", "rmdir"]):
                base_risk = RiskLevel.CRITICAL
            
            # Check for system modifications
            if any(cmd in command for cmd in ["sudo", "su -", "passwd", "useradd"]):
                base_risk = RiskLevel.CRITICAL
            
            # Check for network commands
            if any(cmd in command for cmd in ["wget", "curl", "nc ", "netcat"]):
                base_risk = max(base_risk, RiskLevel.HIGH)
        
        elif operation_type in ["write_file", "delete_file"]:
            path = details.get("path", "")
            
            # Critical system paths
            critical_paths = [
                "/etc/", "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
                "C:\\Windows", "C:\\Program Files",
                "~/.ssh", "~/.bashrc", "~/.profile",
            ]
            
            for critical in critical_paths:
                if critical in path or path.startswith(critical):
                    base_risk = RiskLevel.CRITICAL
                    break
        
        return base_risk
    
    def _format_description(self, operation_type: str, details: Dict[str, Any]) -> str:
        """Format operation description for user."""
        descriptions = {
            "read_file": f"Read file: {details.get('path', 'unknown')}",
            "write_file": f"Write to file: {details.get('path', 'unknown')}",
            "delete_file": f"Delete: {details.get('path', 'unknown')}",
            "execute_command": f"Execute command: {details.get('command', 'unknown')}",
            "send_email": f"Send email to: {details.get('to', 'unknown')}",
            "web_search": f"Search web for: {details.get('query', 'unknown')}",
        }
        
        return descriptions.get(operation_type, f"{operation_type}: {details}")
    
    def _default_approval(self, request: OperationRequest) -> bool:
        """Default approval method - console input."""
        print("\n" + "=" * 70)
        print("🔒 HUMAN APPROVAL REQUIRED")
        print("=" * 70)
        print(f"Operation: {request.operation_type}")
        print(f"Description: {request.description}")
        print(f"Risk Level: {request.risk_level.value.upper()}")
        
        if request.details:
            print("\nDetails:")
            for key, value in request.details.items():
                print(f"  {key}: {value}")
        
        print("\n" + "-" * 70)
        
        while True:
            response = input("Approve this operation? [y/N/d(details)]: ").strip().lower()
            
            if response in ('y', 'yes'):
                print("✅ Approved\n")
                return True
            elif response in ('n', 'no', ''):
                print("❌ Denied\n")
                return False
            elif response in ('d', 'details'):
                print("\nFull details:")
                print(json.dumps(request.details, indent=2))
                print()
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'd' (details)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get approval statistics."""
        return {
            "total_requests": len(self.history),
            "approved": self.approved_count,
            "denied": self.denied_count,
            "approval_rate": self.approved_count / len(self.history) if self.history else 0,
            "by_risk_level": self._stats_by_risk(),
        }
    
    def _stats_by_risk(self) -> Dict[str, int]:
        """Get stats grouped by risk level."""
        stats = {}
        for level in RiskLevel:
            count = sum(1 for r in self.history if r.risk_level == level)
            if count > 0:
                stats[level.value] = count
        return stats
    
    def generate_report(self) -> str:
        """Generate approval report."""
        stats = self.get_stats()
        
        report = [
            "Human-in-the-Loop Approval Report",
            "=" * 50,
            f"Total Operations: {stats['total_requests']}",
            f"Approved: {stats['approved']}",
            f"Denied: {stats['denied']}",
            f"Approval Rate: {stats['approval_rate']:.1%}",
            "",
            "By Risk Level:",
        ]
        
        for level, count in stats['by_risk_level'].items():
            report.append(f"  {level}: {count}")
        
        if self.history:
            report.extend(["", "Recent Operations:"])
            for req in self.history[-10:]:
                status = "✅" if req.approved else "❌"
                report.append(
                    f"  {status} {req.operation_type} ({req.risk_level.value})"
                )
        
        return "\n".join(report)


class ApprovalDecorator:
    """Decorator to add HITL approval to functions."""
    
    def __init__(self, hitl: HumanInTheLoop, operation_type: str):
        self.hitl = hitl
        self.operation_type = operation_type
    
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Check approval
            approved, reason = self.hitl.check_operation(
                self.operation_type,
                {"args": str(args), "kwargs": kwargs}
            )
            
            if not approved:
                raise PermissionError(reason)
            
            # Execute function
            return func(*args, **kwargs)
        
        return wrapper


class SafeAgentWrapper:
    """Wrapper to add HITL to any agent."""
    
    def __init__(self, agent, hitl: HumanInTheLoop):
        self.agent = agent
        self.hitl = hitl
        self._wrap_tools()
    
    def _wrap_tools(self):
        """Wrap agent tools with HITL approval."""
        if hasattr(self.agent, 'tools'):
            original_tools = self.agent.tools.copy()
            self.agent.tools = {}
            
            for name, tool in original_tools.items():
                self.agent.tools[name] = self._create_safe_tool(name, tool)
    
    def _create_safe_tool(self, name: str, tool: Callable) -> Callable:
        """Create HITL-wrapped tool."""
        def safe_tool(*args, **kwargs):
            approved, reason = self.hitl.check_operation(
                name,
                {"args": args, "kwargs": kwargs}
            )
            
            if not approved:
                return f"Error: {reason}"
            
            return tool(*args, **kwargs)
        
        return safe_tool
    
    def run(self, task: str, **kwargs):
        """Run agent with HITL protection."""
        # Check task-level approval for complex tasks
        approved, reason = self.hitl.check_operation(
            "agent_task",
            {"task": task}
        )
        
        if not approved:
            return {"error": reason, "success": False}
        
        return self.agent.run(task, **kwargs)


# Convenience functions
def create_hitl(
    auto_approve: List[str] = None,
    require_confirmation: List[str] = None,
) -> HumanInTheLoop:
    """Quick HITL creation."""
    risk_levels = [RiskLevel[r.upper()] for r in (require_confirmation or [])]
    return HumanInTheLoop(auto_approve=auto_approve, require_confirmation_for=risk_levels)


def with_approval(hitl: HumanInTheLoop, operation_type: str):
    """Decorator factory for HITL approval."""
    def decorator(func: Callable) -> Callable:
        return ApprovalDecorator(hitl, operation_type)(func)
    return decorator
