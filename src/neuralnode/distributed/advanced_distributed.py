"""Advanced Distributed Inference for multi-GPU/multi-server deployment."""

import asyncio
import torch
import torch.distributed as dist
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np


class NodeStatus(Enum):
    """Status of a distributed node."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class NodeConfig:
    """Configuration for a distributed node."""
    id: str
    host: str
    port: int
    gpu_count: int = 0
    gpu_memory: List[int] = None
    status: NodeStatus = NodeStatus.OFFLINE
    priority: int = 1
    max_concurrent: int = 4


class DistributedInferenceEngine:
    """Advanced distributed inference across multiple GPUs/servers."""
    
    def __init__(self, master_node: str = "localhost", master_port: int = 29500):
        self.master_node = master_node
        self.master_port = master_port
        self.nodes: Dict[str, NodeConfig] = {}
        self.local_rank = 0
        self.world_size = 1
        self.initialized = False
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=16)
    
    def initialize(self, backend: str = "nccl") -> bool:
        """Initialize distributed backend."""
        if not torch.cuda.is_available() or not dist.is_available():
            print("Distributed training not available")
            return False
        
        try:
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{self.master_node}:{self.master_port}",
                world_size=self.world_size,
                rank=self.local_rank,
            )
            self.initialized = True
            print(f"Distributed initialized: rank {self.local_rank}/{self.world_size}")
            return True
        except Exception as e:
            print(f"Failed to initialize distributed: {e}")
            return False
    
    def add_node(self, config: NodeConfig) -> None:
        """Add a node to the cluster."""
        with self._lock:
            self.nodes[config.id] = config
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        with self._lock:
            self.nodes.pop(node_id, None)
    
    def discover_nodes(self, ip_range: str = "192.168.1.0/24", port: int = 29500) -> List[NodeConfig]:
        """Auto-discover nodes in network range."""
        discovered = []
        
        # Parse IP range
        base_ip = ip_range.rsplit('.', 1)[0]
        
        # Scan range
        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((ip, port))
                if result == 0:
                    node = NodeConfig(
                        id=f"node_{ip.replace('.', '_')}",
                        host=ip,
                        port=port,
                        status=NodeStatus.ONLINE,
                    )
                    discovered.append(node)
                sock.close()
            except Exception:
                pass
        
        return discovered
    
    def shard_model(
        self,
        model_state: Dict[str, torch.Tensor],
        num_shards: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Shard model across nodes."""
        shards = []
        
        for param_name, param_tensor in model_state.items():
            # Shard each parameter
            shard_size = param_tensor.numel() // num_shards
            
            for i in range(num_shards):
                if i >= len(shards):
                    shards.append({})
                
                start_idx = i * shard_size
                end_idx = start_idx + shard_size if i < num_shards - 1 else param_tensor.numel()
                
                shards[i][param_name] = param_tensor.flatten()[start_idx:end_idx]
        
        return shards
    
    def parallel_generate(
        self,
        prompts: List[str],
        generate_fn: Callable,
        batch_size: int = 4,
    ) -> List[str]:
        """Generate in parallel across nodes."""
        if not self.nodes:
            # Single-node execution
            return [generate_fn(p) for p in prompts]
        
        # Distribute across nodes
        node_ids = list(self.nodes.keys())
        num_nodes = len(node_ids)
        
        # Split prompts
        chunks = [prompts[i::num_nodes] for i in range(num_nodes)]
        
        # Submit to nodes
        futures = []
        for node_id, chunk in zip(node_ids, chunks):
            if chunk:
                future = self._executor.submit(
                    self._generate_on_node,
                    node_id,
                    chunk,
                    generate_fn,
                )
                futures.append((node_id, future))
        
        # Collect results
        results = [None] * len(prompts)
        
        for node_id, future in futures:
            try:
                node_results = future.result(timeout=300)
                # Place in correct order
                node_idx = node_ids.index(node_id)
                for i, result in enumerate(node_results):
                    original_idx = node_idx + i * num_nodes
                    if original_idx < len(results):
                        results[original_idx] = result
            except Exception as e:
                print(f"Node {node_id} failed: {e}")
        
        # Fill missing with retry
        for i, result in enumerate(results):
            if result is None:
                results[i] = generate_fn(prompts[i])
        
        return results
    
    def _generate_on_node(
        self,
        node_id: str,
        prompts: List[str],
        generate_fn: Callable,
    ) -> List[str]:
        """Generate on a specific node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        
        # Remote execution (simplified - would use RPC in production)
        if node.host == "localhost" or node.host == socket.gethostname():
            # Local execution
            return [generate_fn(p) for p in prompts]
        else:
            # Remote execution stub
            print(f"Remote execution to {node.host} not implemented")
            return []
    
    def pipeline_parallel(
        self,
        model_layers: List[torch.nn.Module],
        input_data: torch.Tensor,
    ) -> torch.Tensor:
        """Pipeline parallelism across layers."""
        if not self.initialized or len(model_layers) == 0:
            return input_data
        
        num_layers = len(model_layers)
        num_nodes = len(self.nodes) or 1
        
        # Assign layers to nodes
        layers_per_node = num_layers // num_nodes
        
        result = input_data
        for i, layer in enumerate(model_layers):
            result = layer(result)
            
            # If at layer boundary, communicate to next node
            if (i + 1) % layers_per_node == 0 and i < num_layers - 1:
                # In real implementation, would send to next node
                pass
        
        return result
    
    def tensor_parallel(
        self,
        tensor: torch.Tensor,
        num_shards: int,
    ) -> torch.Tensor:
        """Tensor parallelism - split tensor across GPUs."""
        if not torch.cuda.is_available():
            return tensor
        
        # Get available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            return tensor
        
        # Split tensor
        shards = torch.chunk(tensor, min(num_shards, num_gpus), dim=0)
        
        # Process on each GPU
        results = []
        for i, shard in enumerate(shards):
            gpu_id = i % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
            shard_gpu = shard.to(device)
            
            # Process (simplified)
            result = shard_gpu * 2  # Example operation
            results.append(result.cpu())
        
        # Concatenate
        return torch.cat(results, dim=0)
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        total_gpus = sum(n.gpu_count for n in self.nodes.values())
        total_memory = sum(sum(n.gpu_memory or [0]) for n in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE),
            "total_gpus": total_gpus,
            "total_gpu_memory_gb": total_memory / 1024,
            "initialized": self.initialized,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
        }
    
    def shutdown(self) -> None:
        """Shutdown distributed backend."""
        if self.initialized and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized = False


class ModelParallelWrapper:
    """Wrapper for model parallelism."""
    
    def __init__(self, model: torch.nn.Module, device_ids: List[int]):
        self.model = model
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        if self.num_devices > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    def forward(self, *args, **kwargs):
        """Forward pass with automatic parallelization."""
        return self.model(*args, **kwargs)


# Convenience functions
def setup_distributed(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 29500,
) -> DistributedInferenceEngine:
    """Setup distributed inference."""
    import os
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    engine = DistributedInferenceEngine(master_addr, master_port)
    engine.local_rank = rank
    engine.world_size = world_size
    engine.initialize()
    
    return engine


def run_parallel_inference(
    prompts: List[str],
    generate_fn: Callable,
    num_nodes: int = 1,
) -> List[str]:
    """Quick parallel inference."""
    engine = DistributedInferenceEngine()
    
    # Add local nodes
    for i in range(num_nodes):
        engine.add_node(NodeConfig(
            id=f"local_{i}",
            host="localhost",
            port=29500 + i,
            status=NodeStatus.ONLINE,
        ))
    
    return engine.parallel_generate(prompts, generate_fn)
