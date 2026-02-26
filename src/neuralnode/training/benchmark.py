"""Model benchmarking for comparing LLM performance."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import json
from enum import Enum


class BenchmarkTask(Enum):
    """Types of benchmark tasks."""
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    CODING = "coding"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    MATH = "math"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    provider: str
    task: BenchmarkTask
    score: float  # 0.0 to 1.0
    latency_ms: float
    tokens_used: int
    cost_usd: float
    output: str
    expected_output: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelScore:
    """Overall score for a model."""
    model_name: str
    provider: str
    overall_score: float
    task_scores: Dict[str, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0


class BenchmarkDataset:
    """Dataset of benchmark tasks."""
    
    TASKS = {
        BenchmarkTask.REASONING: [
            {
                "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "expected": "5 minutes",
                "check": "contains"
            },
            {
                "prompt": "A bat and a ball cost $11 in total. The bat costs $10 more than the ball. How much does the ball cost?",
                "expected": "0.5",
                "check": "contains"
            },
        ],
        BenchmarkTask.CODING: [
            {
                "prompt": "Write a Python function to calculate the factorial of a number recursively.",
                "expected": "def factorial",
                "check": "contains"
            },
            {
                "prompt": "Write a Python function to reverse a string without using built-in reverse methods.",
                "expected": "def reverse_string",
                "check": "contains"
            },
        ],
        BenchmarkTask.SUMMARIZATION: [
            {
                "prompt": "Summarize this in one sentence: The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet.",
                "expected": "pangram",
                "check": "contains"
            },
        ],
        BenchmarkTask.CREATIVITY: [
            {
                "prompt": "Write a haiku about artificial intelligence.",
                "expected": None,
                "check": "length",
                "min_length": 10
            },
        ],
        BenchmarkTask.MATH: [
            {
                "prompt": "What is 15 * 23?",
                "expected": "345",
                "check": "contains"
            },
            {
                "prompt": "Solve for x: 2x + 5 = 15",
                "expected": "5",
                "check": "contains"
            },
        ],
        BenchmarkTask.KNOWLEDGE: [
            {
                "prompt": "What is the capital of France?",
                "expected": "Paris",
                "check": "contains"
            },
            {
                "prompt": "Who wrote 'Romeo and Juliet'?",
                "expected": "Shakespeare",
                "check": "contains"
            },
        ],
        BenchmarkTask.INSTRUCTION_FOLLOWING: [
            {
                "prompt": "Reply with only the word 'banana' in uppercase, nothing else.",
                "expected": "BANANA",
                "check": "exact"
            },
        ],
    }
    
    @classmethod
    def get_tasks(cls, task_type: BenchmarkTask) -> List[Dict[str, Any]]:
        """Get tasks for a specific type."""
        return cls.TASKS.get(task_type, [])
    
    @classmethod
    def get_all_tasks(cls) -> Dict[BenchmarkTask, List[Dict[str, Any]]]:
        """Get all benchmark tasks."""
        return cls.TASKS


class ModelBenchmark:
    """Benchmark multiple models."""
    
    def __init__(self, llm_factory: Optional[Callable] = None):
        self.llm_factory = llm_factory
        self.results: List[BenchmarkResult] = []
        self.evaluator_llm = None
    
    def add_evaluator(self, llm) -> None:
        """Add an LLM to evaluate open-ended responses."""
        self.evaluator_llm = llm
    
    def run_benchmark(
        self,
        model_configs: List[Dict[str, Any]],
        tasks: Optional[List[BenchmarkTask]] = None,
        samples_per_task: int = 1,
    ) -> List[ModelScore]:
        """Run benchmark on multiple models."""
        if tasks is None:
            tasks = list(BenchmarkTask)
        
        for config in model_configs:
            print(f"\nBenchmarking {config['name']}...")
            
            # Create LLM instance
            if self.llm_factory:
                llm = self.llm_factory(**config)
            else:
                import nn
                llm = nn.NeuralNode(
                    provider=config.get('provider', 'openai'),
                    model=config.get('model', 'gpt-4'),
                )
            
            # Run tasks
            for task in tasks:
                task_results = self._run_task(
                    llm=llm,
                    model_name=config['name'],
                    provider=config.get('provider', 'openai'),
                    task=task,
                    samples=samples_per_task,
                )
                self.results.extend(task_results)
        
        return self._calculate_scores()
    
    def _run_task(
        self,
        llm,
        model_name: str,
        provider: str,
        task: BenchmarkTask,
        samples: int = 1,
    ) -> List[BenchmarkResult]:
        """Run a single benchmark task."""
        tasks = BenchmarkDataset.get_tasks(task)
        results = []
        
        for task_data in tasks[:samples]:
            start_time = time.time()
            
            try:
                # Execute prompt
                response = llm.chat(task_data['prompt'])
                output = response.text if hasattr(response, 'text') else str(response)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Calculate tokens and cost
                tokens_used = getattr(response, 'usage', None)
                if tokens_used:
                    tokens_used = tokens_used.total_tokens if hasattr(tokens_used, 'total_tokens') else 0
                else:
                    tokens_used = len(output.split())  # Rough estimate
                
                cost_usd = self._estimate_cost(model_name, tokens_used)
                
                # Score the result
                score = self._score_output(output, task_data)
                
                result = BenchmarkResult(
                    model_name=model_name,
                    provider=provider,
                    task=task,
                    score=score,
                    latency_ms=latency_ms,
                    tokens_used=tokens_used,
                    cost_usd=cost_usd,
                    output=output,
                    expected_output=task_data.get('expected'),
                )
                
                results.append(result)
            
            except Exception as e:
                print(f"Error running task {task.value}: {e}")
                results.append(BenchmarkResult(
                    model_name=model_name,
                    provider=provider,
                    task=task,
                    score=0.0,
                    latency_ms=0.0,
                    tokens_used=0,
                    cost_usd=0.0,
                    output="",
                    expected_output=task_data.get('expected'),
                    metadata={"error": str(e)},
                ))
        
        return results
    
    def _score_output(self, output: str, task_data: Dict[str, Any]) -> float:
        """Score the model output."""
        check_type = task_data.get('check', 'contains')
        expected = task_data.get('expected')
        
        if check_type == 'exact':
            return 1.0 if output.strip().upper() == expected.upper() else 0.0
        
        elif check_type == 'contains':
            if expected is None:
                return 0.5  # Partial credit for open-ended
            return 1.0 if expected.lower() in output.lower() else 0.0
        
        elif check_type == 'length':
            min_length = task_data.get('min_length', 0)
            return 1.0 if len(output) >= min_length else len(output) / min_length
        
        return 0.5
    
    def _estimate_cost(self, model_name: str, tokens: int) -> float:
        """Estimate cost for model execution."""
        rates = {
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01,
            'gpt-3.5-turbo': 0.0015,
            'claude-3-opus': 0.015,
            'claude-3-sonnet': 0.003,
            'gemini-pro': 0.001,
        }
        
        rate = rates.get(model_name.lower().replace('-preview', '').replace('-latest', ''), 0.01)
        return (tokens / 1000) * rate
    
    def _calculate_scores(self) -> List[ModelScore]:
        """Calculate overall scores for each model."""
        model_scores: Dict[str, ModelScore] = {}
        
        for result in self.results:
            key = f"{result.model_name}_{result.provider}"
            
            if key not in model_scores:
                model_scores[key] = ModelScore(
                    model_name=result.model_name,
                    provider=result.provider,
                    overall_score=0.0,
                )
            
            score = model_scores[key]
            
            # Update task scores
            task_key = result.task.value
            if task_key not in score.task_scores:
                score.task_scores[task_key] = []
            score.task_scores[task_key].append(result.score)
            
            # Update aggregates
            score.avg_latency_ms += result.latency_ms
            score.total_cost_usd += result.cost_usd
            score.total_tokens += result.tokens_used
        
        # Calculate averages and overall scores
        for score in model_scores.values():
            # Average task scores
            for task, values in score.task_scores.items():
                score.task_scores[task] = sum(values) / len(values)
            
            # Overall score is average of task scores
            if score.task_scores:
                score.overall_score = sum(score.task_scores.values()) / len(score.task_scores)
            
            # Average latency
            num_results = len([r for r in self.results 
                             if r.model_name == score.model_name and r.provider == score.provider])
            if num_results > 0:
                score.avg_latency_ms /= num_results
        
        return list(model_scores.values())
    
    def compare(
        self,
        model_names: List[str],
        tasks: List[str],
    ) -> Dict[str, Any]:
        """Compare specific models on specific tasks."""
        configs = [
            {'name': name, 'provider': 'openai', 'model': name}
            for name in model_names
        ]
        
        task_enums = [BenchmarkTask(t) for t in tasks if t in [e.value for e in BenchmarkTask]]
        
        return self.run_benchmark(configs, task_enums)
    
    def report(self, scores: List[ModelScore], output_file: Optional[str] = None) -> str:
        """Generate benchmark report."""
        lines = []
        lines.append("=" * 80)
        lines.append("Model Benchmark Report")
        lines.append("=" * 80)
        
        # Sort by overall score
        scores = sorted(scores, key=lambda s: s.overall_score, reverse=True)
        
        for i, score in enumerate(scores, 1):
            lines.append(f"\n{i}. {score.model_name} ({score.provider})")
            lines.append(f"   Overall Score: {score.overall_score:.2%}")
            lines.append(f"   Avg Latency: {score.avg_latency_ms:.0f}ms")
            lines.append(f"   Total Cost: ${score.total_cost_usd:.4f}")
            lines.append(f"   Total Tokens: {score.total_tokens}")
            lines.append("   Task Breakdown:")
            
            for task, task_score in sorted(score.task_scores.items()):
                lines.append(f"     - {task}: {task_score:.2%}")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report
    
    def export_results(self, filename: str) -> None:
        """Export results to JSON."""
        data = [
            {
                "model_name": r.model_name,
                "provider": r.provider,
                "task": r.task.value,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "tokens_used": r.tokens_used,
                "cost_usd": r.cost_usd,
                "timestamp": r.timestamp.isoformat(),
            }
            for r in self.results
        ]
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


# Convenience functions
def quick_benchmark(
    models: List[str],
    tasks: Optional[List[str]] = None,
) -> List[ModelScore]:
    """Quick benchmark of multiple models."""
    import nn
    
    benchmark = ModelBenchmark()
    
    configs = [
        {'name': model, 'provider': 'openai', 'model': model}
        for model in models
    ]
    
    if tasks:
        task_enums = [BenchmarkTask(t) for t in tasks]
    else:
        task_enums = None
    
    return benchmark.run_benchmark(configs, task_enums)


def compare_models(model_a: str, model_b: str, task: str = "reasoning") -> Dict[str, Any]:
    """Quick comparison of two models."""
    benchmark = ModelBenchmark()
    
    configs = [
        {'name': model_a, 'provider': 'openai', 'model': model_a},
        {'name': model_b, 'provider': 'openai', 'model': model_b},
    ]
    
    results = benchmark.run_benchmark(configs, [BenchmarkTask(task)])
    
    return {
        model_a: next((r for r in results if r.model_name == model_a), None),
        model_b: next((r for r in results if r.model_name == model_b), None),
    }
