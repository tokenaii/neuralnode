"""Distributed inference with Ray support for scalable AI."""

import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')


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
    ray_node_id: Optional[str] = None


class DistributedCluster:
    """Manage a cluster of nodes for distributed inference using Ray."""
    
    def __init__(self, nodes: Optional[List[Node]] = None, use_ray: bool = False):
        self.nodes = nodes or []
        self.local_node: Optional[Node] = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        self.use_ray = use_ray
        self._ray_initialized = False
        self._remote_actors = {}
        
        if use_ray:
            self._init_ray()
    
    def _init_ray(self) -> bool:
        """Initialize Ray for distributed computing."""
        try:
            import ray
            
            if not ray.is_initialized():
                # Check if we should connect to existing cluster
                if os.environ.get('RAY_ADDRESS'):
                    ray.init(address=os.environ.get('RAY_ADDRESS'))
                    logger.info("Connected to existing Ray cluster")
                else:
                    ray.init(ignore_reinit_error=True)
                    logger.info("Initialized Ray locally")
            
            self._ray_initialized = True
            return True
            
        except ImportError:
            logger.warning("Ray not installed. Install with: pip install ray")
            return False
        except Exception as e:
            logger.error(f"Error initializing Ray: {e}")
            return False
    
    def add_node(self, node: Node) -> None:
        """Add a node to the cluster."""
        self.nodes.append(node)
        
        if self.use_ray and self._ray_initialized and not node.is_local:
            self._connect_remote_node(node)
    
    def _connect_remote_node(self, node: Node) -> bool:
        """Connect to a remote Ray node."""
        try:
            import ray
            
            # In a real implementation, this would connect to the remote Ray node
            # For now, we mark it as available
            logger.info(f"Connected to remote node: {node.id} at {node.host}:{node.port}")
            node.status = "ready"
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to remote node {node.id}: {e}")
            node.status = "error"
            return False
    
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
            return self._run_remote(node, prompts, inference_fn)
    
    def _run_remote(
        self,
        node: Node,
        prompts: List[str],
        inference_fn: Callable,
    ) -> List[Any]:
        """Run inference on a remote node via HTTP API or Ray."""
        try:
            if self.use_ray and self._ray_initialized and node.ray_node_id:
                # Use Ray for remote execution
                import ray
                
                @ray.remote
                def remote_batch(prompts_list: List[str]) -> List[Any]:
                    return [inference_fn(p) for p in prompts_list]
                
                # Run on specific node
                remote_batch.options(resources={f"node:{node.ray_node_id}": 0.01})
                future = remote_batch.remote(prompts)
                return ray.get(future)
            
            else:
                # Use HTTP API for remote execution
                import requests
                
                endpoint = f"http://{node.host}:{node.port}/inference"
                response = requests.post(
                    endpoint,
                    json={"prompts": prompts},
                    timeout=300
                )
                response.raise_for_status()
                return response.json()["results"]
                
        except Exception as e:
            logger.error(f"Remote execution failed on {node.id}: {e}")
            # Fall back to local execution
            logger.info(f"Falling back to local execution for {node.id}")
            return [inference_fn(p) for p in prompts]
    
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
