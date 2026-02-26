"""Rate limiting and retry utilities using tenacity."""

from typing import Callable, Optional, TypeVar, Any
from functools import wraps

from tenacity import (
    retry as tenacity_retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
    RetryCallState,
)
import logging

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        max_delay: Optional[float] = None,
        min_wait: float = 1,
        max_wait: float = 10,
        exponential: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.max_delay = max_delay
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.exponential = exponential
        self.retryable_exceptions = retryable_exceptions
    
    def get_retry_decorator(self):
        """Get tenacity retry decorator based on configuration."""
        stop = stop_after_attempt(self.max_attempts)
        if self.max_delay:
            stop = stop | stop_after_delay(self.max_delay)
        
        if self.exponential:
            wait = wait_exponential(multiplier=1, min=self.min_wait, max=self.max_wait)
        else:
            wait = wait_fixed(self.min_wait)
        
        return tenacity_retry(
            stop=stop,
            wait=wait,
            retry=retry_if_exception_type(self.retryable_exceptions),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )


# Default retry configurations
DEFAULT_RETRY = RetryConfig()
API_RETRY = RetryConfig(
    max_attempts=3,
    min_wait=1,
    max_wait=30,
    retryable_exceptions=(ConnectionError, TimeoutError, Exception),
)
EMBEDDING_RETRY = RetryConfig(
    max_attempts=5,
    min_wait=2,
    max_wait=60,
    retryable_exceptions=(ConnectionError, TimeoutError),
)


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(
        self,
        max_requests: int = 60,
        time_window: float = 60.0,  # seconds
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self._requests: list[float] = []
    
    def can_proceed(self) -> bool:
        """Check if a new request can proceed."""
        import time
        
        now = time.time()
        # Remove old requests outside the window
        cutoff = now - self.time_window
        self._requests = [t for t in self._requests if t > cutoff]
        
        return len(self._requests) < self.max_requests
    
    def acquire(self) -> bool:
        """Acquire a rate limit slot. Returns True if acquired."""
        import time
        
        if self.can_proceed():
            self._requests.append(time.time())
            return True
        return False
    
    def wait_time(self) -> float:
        """Get estimated wait time until next slot available."""
        import time
        
        if self.can_proceed():
            return 0
        
        if not self._requests:
            return 0
        
        oldest = min(self._requests)
        return max(0, (oldest + self.time_window) - time.time())


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on API responses."""
    
    def __init__(
        self,
        initial_rate: int = 60,
        min_rate: int = 1,
        max_rate: int = 1000,
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._rate_limiter = RateLimiter(max_requests=initial_rate)
        self._consecutive_errors = 0
        self._consecutive_successes = 0
    
    def on_success(self):
        """Call when a request succeeds."""
        self._consecutive_successes += 1
        self._consecutive_errors = 0
        
        # Gradually increase rate
        if self._consecutive_successes >= 10:
            self.current_rate = min(self.max_rate, self.current_rate + 5)
            self._rate_limiter.max_requests = self.current_rate
            self._consecutive_successes = 0
    
    def on_rate_limit_error(self):
        """Call when hitting rate limit (429)."""
        self._consecutive_errors += 1
        self._consecutive_successes = 0
        
        # Decrease rate
        self.current_rate = max(self.min_rate, self.current_rate // 2)
        self._rate_limiter.max_requests = self.current_rate
    
    def acquire(self) -> bool:
        """Acquire a rate limit slot."""
        return self._rate_limiter.acquire()
    
    def wait_time(self) -> float:
        """Get wait time."""
        return self._rate_limiter.wait_time()


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
    exponential: bool = True,
    retryable_exceptions: tuple = (Exception,),
):
    """
    Decorator to add retry logic to a function.
    
    Example:
        @with_retry(max_attempts=5)
        def call_api():
            return requests.get("...")
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        min_wait=min_wait,
        max_wait=max_wait,
        exponential=exponential,
        retryable_exceptions=retryable_exceptions,
    )
    return config.get_retry_decorator()


def with_rate_limit(
    max_requests: int = 60,
    time_window: float = 60.0,
):
    """
    Decorator to add rate limiting to a function.
    
    Example:
        @with_rate_limit(max_requests=10, time_window=60)
        def call_api():
            return requests.get("...")
    """
    limiter = RateLimiter(max_requests=max_requests, time_window=time_window)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            import time
            
            while not limiter.acquire():
                wait = limiter.wait_time()
                logger.warning(f"Rate limit hit, waiting {wait:.2f}s")
                time.sleep(wait)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Retryable exception types for common APIs
OPENAI_RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    Exception,  # OpenAI specific errors
)

ANTHROPIC_RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    Exception,
)

LOCAL_RETRY_EXCEPTIONS = (
    ConnectionError,
    RuntimeError,
    OSError,
)
