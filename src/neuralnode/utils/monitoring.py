"""Monitoring and observability for NeuralNode (LangSmith-like)."""

import json
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import threading
import uuid


@dataclass
class Run:
    """A single LLM run/trace."""
    id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    parent_id: Optional[str] = None
    children: List["Run"] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get run duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


class RunManager:
    """Manage runs/traces."""
    
    def __init__(self):
        self._runs: Dict[str, Run] = {}
        self._current_run: Optional[Run] = None
        self._lock = threading.Lock()
    
    def start_run(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Run:
        """Start a new run."""
        run = Run(
            id=str(uuid.uuid4()),
            name=name,
            start_time=time.time(),
            inputs=inputs or {},
            tags=tags or [],
            metadata=metadata or {},
        )
        
        with self._lock:
            self._runs[run.id] = run
            self._current_run = run
        
        return run
    
    def end_run(
        self,
        run_id: Optional[str] = None,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """End a run."""
        with self._lock:
            if run_id:
                run = self._runs.get(run_id)
            else:
                run = self._current_run
            
            if run:
                run.end_time = time.time()
                if outputs:
                    run.outputs = outputs
                if error:
                    run.error = error
                if metrics:
                    run.metrics.update(metrics)
                
                if run == self._current_run:
                    self._current_run = None
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        return self._runs.get(run_id)
    
    def list_runs(
        self,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Run]:
        """List runs with filtering."""
        runs = list(self._runs.values())
        
        if name:
            runs = [r for r in runs if r.name == name]
        
        if tags:
            runs = [r for r in runs if any(t in r.tags for t in tags)]
        
        # Sort by start time (newest first)
        runs.sort(key=lambda r: r.start_time, reverse=True)
        
        return runs[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get run statistics."""
        total = len(self._runs)
        completed = sum(1 for r in self._runs.values() if r.end_time is not None)
        errors = sum(1 for r in self._runs.values() if r.error is not None)
        
        durations = [
            r.duration_ms for r in self._runs.values()
            if r.duration_ms is not None
        ]
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_runs": total,
            "completed": completed,
            "in_progress": total - completed,
            "errors": errors,
            "success_rate": (completed - errors) / completed if completed > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2),
        }


class MetricsCollector:
    """Collect metrics from runs."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, metric_name: str, value: float) -> None:
        """Record a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        values = self.metrics.get(metric_name, [])
        if not values:
            return {}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values),
        }


class NeuralMonitor:
    """Main monitoring dashboard (LangSmith alternative)."""
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.run_manager = RunManager()
        self.metrics = MetricsCollector()
        self.callbacks: List[Callable] = []
        self._storage_path = Path.home() / ".neuralnode" / "monitoring"
        self._storage_path.mkdir(parents=True, exist_ok=True)
    
    def start_trace(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        **metadata,
    ) -> str:
        """Start a new trace."""
        run = self.run_manager.start_run(
            name=name,
            inputs=inputs,
            metadata={"project": self.project_name, **metadata},
        )
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback("trace_start", run)
            except Exception as e:
                logger.warning(f"Callback error during trace_start: {e}")
        
        return run.id
    
    def end_trace(
        self,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a trace."""
        self.run_manager.end_run(run_id=run_id, outputs=outputs, error=error)
        
        run = self.run_manager.get_run(run_id)
        if run:
            # Save to disk
            self._save_run(run)
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback("trace_end", run)
                except Exception as e:
                    logger.warning(f"Callback error during trace_end: {e}")
    
    def record_metric(self, run_id: str, name: str, value: float) -> None:
        """Record a metric for a run."""
        run = self.run_manager.get_run(run_id)
        if run:
            run.metrics[name] = value
            self.metrics.record(name, value)
    
    def record_token_usage(
        self,
        run_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        """Record token usage."""
        total = prompt_tokens + completion_tokens
        
        # Estimate cost (simplified)
        rates = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.0015,
        }
        rate = rates.get(model, 0.01)
        cost = (total / 1000) * rate
        
        self.record_metric(run_id, "prompt_tokens", prompt_tokens)
        self.record_metric(run_id, "completion_tokens", completion_tokens)
        self.record_metric(run_id, "total_tokens", total)
        self.record_metric(run_id, "cost_usd", cost)
    
    def add_callback(self, callback: Callable) -> None:
        """Add a monitoring callback."""
        self.callbacks.append(callback)
    
    def _save_run(self, run: Run) -> None:
        """Save run to disk."""
        run_file = self._storage_path / f"{run.id}.json"
        with open(run_file, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        stats = self.run_manager.get_stats()
        recent_runs = self.run_manager.list_runs(limit=50)
        
        # Token usage stats
        token_stats = self.metrics.get_stats("total_tokens")
        cost_stats = self.metrics.get_stats("cost_usd")
        
        return {
            "project": self.project_name,
            "stats": stats,
            "recent_runs": [
                {
                    "id": r.id,
                    "name": r.name,
                    "duration_ms": r.duration_ms,
                    "status": "error" if r.error else "success",
                    "start_time": datetime.fromtimestamp(r.start_time).isoformat(),
                }
                for r in recent_runs
            ],
            "token_usage": token_stats,
            "cost": cost_stats,
        }
    
    def print_dashboard(self) -> None:
        """Print dashboard to console."""
        data = self.get_dashboard_data()
        
        print("\n" + "=" * 60)
        print(f"🧠 NeuralNode Monitor - {data['project']}")
        print("=" * 60)
        
        stats = data["stats"]
        print(f"\n📊 Runs: {stats['total_runs']} total | {stats['in_progress']} active | {stats['errors']} errors")
        print(f"✅ Success Rate: {stats['success_rate']:.1%}")
        print(f"⏱️ Avg Duration: {stats['avg_duration_ms']:.0f}ms")
        
        if data["cost"]:
            print(f"💰 Total Cost: ${data['cost']['sum']:.4f}")
        
        print("\n🔥 Recent Runs:")
        for run in data["recent_runs"][:10]:
            status_icon = "❌" if run["status"] == "error" else "✅"
            print(f"  {status_icon} {run['name'][:40]:40} | {run['duration_ms']:.0f}ms")
        
        print("=" * 60)


class MonitoringCallback:
    """Callback for automatic monitoring."""
    
    def __init__(self, monitor: NeuralMonitor):
        self.monitor = monitor
        self._active_runs: Dict[str, str] = {}  # func call -> run_id
    
    def on_llm_start(self, **kwargs) -> str:
        """Called when LLM call starts."""
        run_id = self.monitor.start_trace(
            name=kwargs.get("model", "llm_call"),
            inputs={
                "prompt": kwargs.get("prompt", "")[:1000],
                "model": kwargs.get("model"),
            },
            **kwargs,
        )
        return run_id
    
    def on_llm_end(self, run_id: str, response, **kwargs) -> None:
        """Called when LLM call ends."""
        outputs = {"text": response.text[:1000]} if hasattr(response, 'text') else {"response": str(response)[:1000]}
        
        self.monitor.end_trace(run_id, outputs=outputs)
        
        # Record token usage if available
        if hasattr(response, 'usage'):
            self.monitor.record_token_usage(
                run_id,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                kwargs.get("model", "unknown"),
            )
    
    def on_llm_error(self, run_id: str, error: Exception) -> None:
        """Called when LLM call errors."""
        self.monitor.end_trace(run_id, error=str(error))


# Global monitor instance
_global_monitor = NeuralMonitor()


def get_monitor() -> NeuralMonitor:
    """Get global monitor instance."""
    return _global_monitor


def trace(
    name: Optional[str] = None,
    project: Optional[str] = None,
    **metadata,
):
    """Decorator for automatic tracing."""
    monitor = _global_monitor
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            
            # Start trace
            run_id = monitor.start_trace(
                name=trace_name,
                inputs={"args": args, "kwargs": kwargs},
                **metadata,
            )
            
            try:
                # Execute
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # End trace
                outputs = {"result": str(result)[:1000]}
                monitor.end_trace(run_id, outputs=outputs)
                monitor.record_metric(run_id, "duration_seconds", duration)
                
                return result
            except Exception as e:
                monitor.end_trace(run_id, error=str(e))
                raise
        
        return wrapper
    return decorator
