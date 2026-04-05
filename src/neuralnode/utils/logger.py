"""Logging and error handling system for NeuralNode.

Phase 1: Core Stability - Structured logging and error tracking.
"""

import logging


import sys
import json
import traceback
from typing import Any, Dict, Optional, Callable
from datetime import datetime
from pathlib import Path
from functools import wraps


# Configure structured logging
class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = traceback.format_exception(*record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """Setup structured logging for NeuralNode.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logs
        json_format: Use JSON formatting for structured logs
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("neuralnode")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_format:
        console_handler.setFormatter(StructuredLogFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        
        if json_format:
            file_handler.setFormatter(StructuredLogFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
        
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the global NeuralNode logger."""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


class NeuralNodeError(Exception):
    """Base exception for NeuralNode errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "type": self.__class__.__name__,
        }


class ProviderError(NeuralNodeError):
    """Error from LLM provider."""
    
    def __init__(self, message: str, provider: str, status_code: Optional[int] = None):
        super().__init__(
            message=message,
            error_code=f"PROVIDER_ERROR_{provider.upper()}",
            details={"provider": provider, "status_code": status_code}
        )
        self.provider = provider
        self.status_code = status_code


class ToolError(NeuralNodeError):
    """Error from tool execution."""
    
    def __init__(self, message: str, tool_name: str):
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            details={"tool": tool_name}
        )
        self.tool_name = tool_name


class ConfigurationError(NeuralNodeError):
    """Configuration error."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key}
        )


class ValidationError(NeuralNodeError):
    """Validation error."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field}
        )


def log_execution(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger()
            func_name = func.__name__
            module_name = func.__module__
            
            log.debug(
                f"Executing {module_name}.{func_name}",
                extra={"function": func_name, "module": module_name}
            )
            
            try:
                result = func(*args, **kwargs)
                log.debug(
                    f"Successfully executed {func_name}",
                    extra={"function": func_name, "status": "success"}
                )
                return result
            except Exception as e:
                log.error(
                    f"Error in {func_name}: {str(e)}",
                    extra={
                        "function": func_name,
                        "error": str(e),
                        "status": "error"
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


class ErrorHandler:
    """Central error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()
        self.error_counts: Dict[str, int] = {}
        self.max_retries = 3
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        retry_callback: Optional[Callable] = None
    ) -> bool:
        """Handle an error with optional retry.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            retry_callback: Function to call for retry
        
        Returns:
            True if recovered, False if failed
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(
            f"Error occurred: {error_type}: {str(error)}",
            extra={
                "error_type": error_type,
                "error_message": str(error),
                "context": context or {},
                "error_count": self.error_counts[error_type]
            },
            exc_info=True
        )
        
        # Attempt retry if callback provided and under max retries
        if retry_callback and self.error_counts[error_type] <= self.max_retries:
            self.logger.info(f"Attempting retry {self.error_counts[error_type]}/{self.max_retries}")
            try:
                retry_callback()
                self.logger.info("Retry successful")
                return True
            except Exception as retry_error:
                self.logger.error(f"Retry failed: {str(retry_error)}")
        
        return False
    
    def reset_error_count(self, error_type: Optional[str] = None):
        """Reset error counts."""
        if error_type:
            self.error_counts[error_type] = 0
        else:
            self.error_counts.clear()


# Convenience exports
def log_info(message: str, **kwargs):
    """Log info message."""
    get_logger().info(message, extra=kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message."""
    get_logger().warning(message, extra=kwargs)


def log_error(message: str, **kwargs):
    """Log error message."""
    get_logger().error(message, extra=kwargs)


def log_debug(message: str, **kwargs):
    """Log debug message."""
    get_logger().debug(message, extra=kwargs)
