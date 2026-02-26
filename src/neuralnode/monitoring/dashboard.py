"""Observability Dashboard for monitoring NeuralNode deployments."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str]


@dataclass
class TraceSpan:
    """Distributed trace span."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str = "ok"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0


class MetricsCollector:
    """Collect and store metrics."""
    
    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.retention = timedelta(hours=retention_hours)
        self._lock = threading.Lock()
    
    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                labels=labels or {},
            )
            
            self.metrics[name].append(point)
            
            # Cleanup old data
            cutoff = datetime.now() - self.retention
            self.metrics[name] = [
                p for p in self.metrics[name] if p.timestamp > cutoff
            ]
    
    def get_metric(
        self,
        name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """Get metric data."""
        with self._lock:
            points = self.metrics.get(name, [])
            
            if start:
                points = [p for p in points if p.timestamp >= start]
            if end:
                points = [p for p in points if p.timestamp <= end]
            
            return points
    
    def aggregate(
        self,
        name: str,
        func: str = "avg",
        window_minutes: int = 5,
    ) -> List[Dict[str, Any]]:
        """Aggregate metric over time windows."""
        points = self.get_metric(name)
        if not points:
            return []
        
        # Group by time window
        windows: Dict[str, List[float]] = {}
        
        for point in points:
            window_key = point.timestamp.strftime("%Y-%m-%d %H:%M")[:15] + "0:00"
            if window_key not in windows:
                windows[window_key] = []
            windows[window_key].append(point.value)
        
        # Calculate aggregation
        results = []
        for window, values in sorted(windows.items()):
            if func == "avg":
                value = sum(values) / len(values)
            elif func == "sum":
                value = sum(values)
            elif func == "min":
                value = min(values)
            elif func == "max":
                value = max(values)
            elif func == "count":
                value = len(values)
            else:
                value = sum(values) / len(values)
            
            results.append({
                "timestamp": window,
                "value": value,
                "count": len(values),
            })
        
        return results


class Tracer:
    """Distributed tracing."""
    
    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self._lock = threading.Lock()
    
    def start_span(
        self,
        operation: str,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Start a new trace span."""
        span_id = f"span_{datetime.now().timestamp()}"
        trace_id = trace_id or f"trace_{datetime.now().timestamp()}"
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            operation=operation,
            start_time=datetime.now(),
            end_time=None,
        )
        
        with self._lock:
            if trace_id not in self.traces:
                self.traces[trace_id] = []
            self.traces[trace_id].append(span)
            self.active_spans[span_id] = span
        
        return span_id
    
    def end_span(self, span_id: str, status: str = "ok") -> None:
        """End a trace span."""
        with self._lock:
            if span_id in self.active_spans:
                span = self.active_spans[span_id]
                span.end_time = datetime.now()
                span.status = status
                del self.active_spans[span_id]
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self._lock:
            return self.traces.get(trace_id, [])
    
    def search_traces(
        self,
        operation: Optional[str] = None,
        status: Optional[str] = None,
        start: Optional[datetime] = None,
    ) -> List[str]:
        """Search for trace IDs."""
        results = []
        
        with self._lock:
            for trace_id, spans in self.traces.items():
                for span in spans:
                    match = True
                    if operation and span.operation != operation:
                        match = False
                    if status and span.status != status:
                        match = False
                    if start and span.start_time < start:
                        match = False
                    
                    if match:
                        results.append(trace_id)
                        break
        
        return results


class ObservabilityDashboard:
    """Main observability dashboard."""
    
    def __init__(self, project_name: str = "neuralnode"):
        self.project = project_name
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        self.logs: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
    
    def log(self, level: str, message: str, **kwargs) -> None:
        """Add a log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs,
        }
        self.logs.append(entry)
        
        # Keep only last 10000 logs
        if len(self.logs) > 10000:
            self.logs = self.logs[-10000:]
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Record LLM API call metrics."""
        self.metrics.record(
            "llm_requests_total",
            1,
            {"model": model, "status": "success" if success else "error"},
        )
        
        self.metrics.record(
            "llm_tokens_total",
            prompt_tokens + completion_tokens,
            {"model": model, "type": "total"},
        )
        
        self.metrics.record(
            "llm_latency_ms",
            latency_ms,
            {"model": model},
        )
    
    def record_cache_hit(self, hit: bool) -> None:
        """Record cache hit/miss."""
        self.metrics.record(
            "cache_access_total",
            1,
            {"result": "hit" if hit else "miss"},
        )
    
    def create_trace(self, operation: str) -> str:
        """Create a new trace."""
        return self.tracer.start_span(operation)
    
    def end_trace(self, span_id: str, status: str = "ok") -> None:
        """End a trace."""
        self.tracer.end_span(span_id, status)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data."""
        # Get last hour of metrics
        last_hour = datetime.now() - timedelta(hours=1)
        
        # LLM metrics
        llm_requests = self.metrics.aggregate("llm_requests_total", "sum")
        llm_latency = self.metrics.aggregate("llm_latency_ms", "avg")
        
        # Cache metrics
        cache_hits = self.metrics.get_metric("cache_access_total", start=last_hour)
        cache_hit_rate = 0
        if cache_hits:
            hits = sum(1 for p in cache_hits if p.labels.get("result") == "hit")
            cache_hit_rate = hits / len(cache_hits)
        
        # Recent traces
        recent_traces = self.tracer.search_traces(start=last_hour)[:10]
        
        # Recent logs
        recent_logs = [l for l in self.logs[-100:] if l["level"] in ["ERROR", "WARN"]]
        
        return {
            "project": self.project,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "llm_requests_1h": sum(p["value"] for p in llm_requests),
                "avg_latency_ms": llm_latency[-1]["value"] if llm_latency else 0,
                "cache_hit_rate": cache_hit_rate,
            },
            "traces": len(recent_traces),
            "alerts": len([a for a in self.alerts if not a.get("resolved")]),
            "recent_errors": recent_logs,
        }
    
    def generate_report(self, output_path: str) -> None:
        """Generate observability report."""
        data = self.get_dashboard_data()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NeuralNode Observability - {self.project}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 8px; margin-bottom: 20px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .section {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h2 {{ margin-top: 0; color: #333; }}
        .error {{ color: #e74c3c; padding: 10px; background: #fdf2f2; border-radius: 4px; margin: 5px 0; }}
        .warn {{ color: #f39c12; padding: 10px; background: #fef9e7; border-radius: 4px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 NeuralNode Observability</h1>
            <p>Project: {self.project} | Generated: {data['timestamp']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{data['metrics']['llm_requests_1h']:.0f}</div>
                <div class="metric-label">LLM Requests (1h)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['metrics']['avg_latency_ms']:.0f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['metrics']['cache_hit_rate']:.1%}</div>
                <div class="metric-label">Cache Hit Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{data['traces']}</div>
                <div class="metric-label">Active Traces</div>
            </div>
        </div>
        
        <div class="section">
            <h2>⚠️ Recent Errors</h2>
"""
        
        for log in data['recent_errors']:
            css_class = log['level'].lower()
            html += f'<div class="{css_class}"><strong>{log["level"]}</strong>: {log["message"]} <small>({log["timestamp"]})</small></div>\n'
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)


# Convenience functions
def record_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """Quick metric recording."""
    collector = MetricsCollector()
    collector.record(name, value, labels)


def start_trace(operation: str) -> str:
    """Quick trace start."""
    tracer = Tracer()
    return tracer.start_span(operation)
