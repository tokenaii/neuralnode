"""Monitoring and observability for NeuralNode.

Phase 3: Enterprise - Metrics, tracing, and cost tracking.
"""

import time
import functools
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import threading

from ..utils.logger import get_logger



@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    cost_usd: float = 0.0


class MetricsCollector:
    """Collect and aggregate metrics."""
    
    # Pricing per 1K tokens (approximate)
    PRICING = {
        "anthropic_claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "google_gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "ollama": {"input": 0.0, "output": 0.0},  # Free
    }
    
    def __init__(self):
        self.requests: List[RequestMetrics] = []
        self._lock = threading.Lock()
        self.logger = get_logger()
    
    def record_request(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None
    ):
        """Record a request."""
        # Calculate cost
        cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
        
        metrics = RequestMetrics(
            timestamp=datetime.utcnow(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
            cost_usd=cost
        )
        
        with self._lock:
            self.requests.append(metrics)
        
        self.logger.debug(
            f"Request recorded: {provider}/{model}, "
            f"tokens={input_tokens}+{output_tokens}, "
            f"cost=${cost:.4f}, latency={latency_ms:.2f}ms"
        )
    
    def _calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate request cost in USD."""
        key = f"{provider}_{model}"
        
        # Find matching pricing
        for pricing_key, pricing in self.PRICING.items():
            if pricing_key in key or key in pricing_key:
                input_cost = (input_tokens / 1000) * pricing["input"]
                output_cost = (output_tokens / 1000) * pricing["output"]
                return input_cost + output_cost
        
        return 0.0  # Unknown pricing
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics summary."""
        with self._lock:
            requests = self.requests[-last_n:] if last_n else self.requests
        
        if not requests:
            return {"total_requests": 0}
        
        total_cost = sum(r.cost_usd for r in requests)
        total_tokens = sum(r.input_tokens + r.output_tokens for r in requests)
        avg_latency = sum(r.latency_ms for r in requests) / len(requests)
        success_rate = sum(1 for r in requests if r.success) / len(requests)
        
        # Group by provider
        by_provider = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for r in requests:
            by_provider[r.provider]["requests"] += 1
            by_provider[r.provider]["cost"] += r.cost_usd
            by_provider[r.provider]["tokens"] += r.input_tokens + r.output_tokens
        
        return {
            "total_requests": len(requests),
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_latency_ms": avg_latency,
            "success_rate": success_rate,
            "by_provider": dict(by_provider),
            "time_range": {
                "start": requests[0].timestamp.isoformat(),
                "end": requests[-1].timestamp.isoformat()
            }
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Request counter
        lines.append("# HELP neuralnode_requests_total Total requests")
        lines.append("# TYPE neuralnode_requests_total counter")
        
        by_provider = defaultdict(int)
        for r in self.requests:
            by_provider[(r.provider, r.model, r.success)] += 1
        
        for (provider, model, success), count in by_provider.items():
            status = "success" if success else "error"
            lines.append(
                f'neuralnode_requests_total{{provider="{provider}",model="{model}",status="{status}"}} {count}'
            )
        
        # Cost counter
        lines.append("\n# HELP neuralnode_cost_usd_total Total cost in USD")
        lines.append("# TYPE neuralnode_cost_usd_total counter")
        
        cost_by_provider = defaultdict(float)
        for r in self.requests:
            cost_by_provider[(r.provider, r.model)] += r.cost_usd
        
        for (provider, model), cost in cost_by_provider.items():
            lines.append(
                f'neuralnode_cost_usd_total{{provider="{provider}",model="{model}"}} {cost:.6f}'
            )
        
        return "\n".join(lines)


def track_metrics(collector: Optional[MetricsCollector] = None):
    """Decorator to track function metrics."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = collector or MetricsCollector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                if hasattr(result, 'usage') and hasattr(result, 'model'):
                    latency = (time.time() - start_time) * 1000
                    usage = result.usage or {}
                    
                    metrics.record_request(
                        provider=result.provider,
                        model=result.model,
                        input_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('completion_tokens', 0),
                        latency_ms=latency,
                        success=True
                    )
                
                return result
            
            except Exception as e:
                # Record failure
                latency = (time.time() - start_time) * 1000
                metrics.record_request(
                    provider="unknown",
                    model="unknown",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    success=False,
                    error_type=type(e).__name__
                )
                raise
        
        return wrapper
    return decorator


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics
