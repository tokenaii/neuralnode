"""Distributed inference across multiple nodes/GPUs."""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Node:
    """A compute node for distributed inference."""
    id: str
    host: str
    port: int
    gpu_count: int = 0
    vram_gb: float = 0.0
    is_local: bool = True
    status: str = "unknown"
    
    def endpoint(self) -> str:
        """Get API endpoint."""
        return f"http://{self.host}:{port}"


class DistributedCluster:
    """Manage a cluster of nodes for distributed inference."""
    
    def __init__(self, nodes: Optional[List[Node]] = None):
        self.nodes = nodes or []
        self.local_node: Optional[Node] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
    
    def add_node(self, node: Node) -> None:
        """Add a node to the cluster."""
        self.nodes.append(node)
    
    def discover_local(self) -> Node:
        """Discover local GPU capabilities."""
        from ..local.hardware import HardwareDetector
        
        hw = HardwareDetector()
        
        node = Node(
            id="local",
            host="localhost",
            port=8000,
            gpu_count=1 if hw.has_cuda else 0,
            vram_gb=hw.vram_gb,
            is_local=True,
            status="ready",
        )
        
        self.local_node = node
        return node
    
    def get_total_vram(self) -> float:
        """Get total VRAM across all nodes."""
        return sum(node.vram_gb for node in self.nodes)
    
    def get_best_node(self, vram_required: float = 0) -> Optional[Node]:
        """Get best available node for a task."""
        available = [n for n in self.nodes if n.status == "ready"]
        
        if not available:
            return None
        
        if vram_required > 0:
            # Find node with enough VRAM
            for node in sorted(available, key=lambda n: n.vram_gb, reverse=True):
                if node.vram_gb >= vram_required:
                    return node
        
        # Return node with most VRAM
        return max(available, key=lambda n: n.vram_gb)
    
    def distribute_batch(
        self,
        prompts: List[str],
        inference_fn: Callable,
    ) -> List[Any]:
        """Distribute batch inference across nodes."""
        if not self.nodes:
            # Run locally
            return [inference_fn(p) for p in prompts]
        
        # Split prompts among nodes
        num_nodes = len(self.nodes)
        chunks = [prompts[i::num_nodes] for i in range(num_nodes)]
        
        # Submit tasks
        futures = {}
        for i, chunk in enumerate(chunks):
            if chunk:
                future = self._executor.submit(
                    self._run_on_node,
                    self.nodes[i],
                    chunk,
                    inference_fn
                )
                futures[future] = i
        
        # Collect results
        results = [None] * len(prompts)
        
        for future in as_completed(futures):
            node_idx = futures[future]
            try:
                chunk_results = future.result()
                # Place results in correct positions
                for j, result in enumerate(chunk_results):
                    original_idx = node_idx + j * num_nodes
                    if original_idx < len(results):
                        results[original_idx] = result
            except Exception as e:
                print(f"Node {node_idx} failed: {e}")
        
        return results
    
    def _run_on_node(
        self,
        node: Node,
        prompts: List[str],
        inference_fn: Callable,
    ) -> List[Any]:
        """Run inference on a specific node."""
        if node.is_local:
            return [inference_fn(p) for p in prompts]
        else:
            # Remote call would go here
            raise NotImplementedError("Remote nodes not yet implemented")
    
    def model_parallel(
        self,
        model_name: str,
        layer_distribution: Optional[Dict[str, int]] = None,
    ) -> "DistributedModel":
        """Distribute model layers across GPUs."""
        return DistributedModel(self, model_name, layer_distribution)


class DistributedModel:
    """Model distributed across multiple GPUs/nodes."""
    
    def __init__(
        self,
        cluster: DistributedCluster,
        model_name: str,
        layer_distribution: Optional[Dict[str, int]] = None,
    ):
        self.cluster = cluster
        self.model_name = model_name
        self.layer_distribution = layer_distribution or {}
        self._loaded = False
    
    def load(self) -> None:
        """Load model across cluster."""
        print(f"Loading {self.model_name} across {len(self.cluster.nodes)} nodes")
        
        # Calculate layer distribution
        if not self.layer_distribution:
            total_gpus = sum(n.gpu_count for n in self.cluster.nodes)
            # Simplified: equal distribution
            self.layer_distribution = {
                node.id: 100 // total_gpus  # Simplified percentage
                for node in self.cluster.nodes
            }
        
        self._loaded = True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using distributed model."""
        if not self._loaded:
            self.load()
        
        # In practice, this would involve:
        # 1. Send prompt to first node (input layers)
        # 2. Each node processes its layers
        # 3. Pass activations to next node
        # 4. Last node returns output
        
        print("Distributed generation not fully implemented")
        print("Falling back to local generation")
        return prompt  # Placeholder


class PipelineParallel:
    """Pipeline parallelism for large models."""
    
    def __init__(self, stages: List[Callable]):
        self.stages = stages
    
    def run(self, input_data: Any) -> Any:
        """Run through pipeline stages."""
        result = input_data
        for stage in self.stages:
            result = stage(result)
        return result
    
    async def arun(self, input_data: Any) -> Any:
        """Async pipeline execution."""
        result = input_data
        for stage in self.stages:
            if asyncio.iscoroutinefunction(stage):
                result = await stage(result)
            else:
                result = stage(result)
        return result


class TensorParallel:
    """Tensor parallelism for individual layers."""
    
    def __init__(self, num_shards: int = 2):
        self.num_shards = num_shards
    
    def shard_tensor(self, tensor: List[float]) -> List[List[float]]:
        """Split tensor into shards."""
        shard_size = len(tensor) // self.num_shards
        return [
            tensor[i * shard_size:(i + 1) * shard_size]
            for i in range(self.num_shards)
        ]
    
    def gather_tensor(self, shards: List[List[float]]) -> List[float]:
        """Combine tensor shards."""
        result = []
        for shard in shards:
            result.extend(shard)
        return result


class LoadBalancer:
    """Balance load across nodes."""
    
    def __init__(self, cluster: DistributedCluster):
        self.cluster = cluster
        self.node_loads: Dict[str, float] = {}
    
    def get_least_loaded(self) -> Optional[Node]:
        """Get node with lowest load."""
        if not self.nodes:
            return None
        
        return min(
            self.cluster.nodes,
            key=lambda n: self.node_loads.get(n.id, 0)
        )
    
    def update_load(self, node_id: str, load: float) -> None:
        """Update node load."""
        self.node_loads[node_id] = load


# Convenience functions
def create_cluster(nodes: Optional[List[str]] = None) -> DistributedCluster:
    """Create a distributed cluster."""
    cluster = DistributedCluster()
    
    # Add local node
    local = cluster.discover_local()
    
    # Add remote nodes if specified
    if nodes:
        for i, host in enumerate(nodes):
            cluster.add_node(Node(
                id=f"node_{i}",
                host=host,
                port=8000 + i,
                is_local=False,
            ))
    
    return cluster


def distribute_batch_inference(
    prompts: List[str],
    inference_fn: Callable,
    nodes: Optional[List[str]] = None,
) -> List[Any]:
    """Quick distributed batch inference."""
    cluster = create_cluster(nodes)
    return cluster.distribute_batch(prompts, inference_fn)
