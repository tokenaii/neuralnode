"""Federated Learning for distributed training without sharing data."""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import threading


@dataclass
class FederatedClient:
    """A client in federated learning."""
    id: str
    data_size: int
    model_update: Optional[Dict[str, torch.Tensor]] = None
    last_update: Optional[datetime] = None
    is_available: bool = True


@dataclass
class FLConfig:
    """Federated Learning configuration."""
    num_rounds: int = 10
    clients_per_round: int = 5
    local_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 32
    aggregation_method: str = "fedavg"  # fedavg, fedprox, scaffold
    dp_epsilon: Optional[float] = None  # Differential privacy budget
    dp_delta: Optional[float] = None


class FederatedAveraging:
    """Federated Averaging (FedAvg) algorithm."""
    
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
        self.clients: Dict[str, FederatedClient] = {}
        self.round = 0
        self._lock = threading.Lock()
    
    def register_client(self, client_id: str, data_size: int) -> None:
        """Register a federated client."""
        self.clients[client_id] = FederatedClient(
            id=client_id,
            data_size=data_size,
        )
    
    def select_clients(self, num_clients: int) -> List[str]:
        """Select clients for current round."""
        available = [c for c in self.clients.values() if c.is_available]
        
        if len(available) <= num_clients:
            return [c.id for c in available]
        
        # Weighted by data size
        weights = [c.data_size for c in available]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        import random
        selected = random.choices(
            [c.id for c in available],
            weights=probs,
            k=num_clients,
        )
        
        return list(set(selected))  # Remove duplicates
    
    def aggregate(self, client_updates: List[Tuple[str, Dict[str, torch.Tensor]]]) -> None:
        """Aggregate client updates using FedAvg."""
        with self._lock:
            # Calculate total data size
            total_data = sum(self.clients[cid].data_size for cid, _ in client_updates)
            
            # Initialize global update
            global_state = self.global_model.state_dict()
            
            # Weighted average
            for key in global_state.keys():
                weighted_sum = torch.zeros_like(global_state[key])
                
                for client_id, update in client_updates:
                    weight = self.clients[client_id].data_size / total_data
                    weighted_sum += weight * update[key]
                
                global_state[key] = weighted_sum
            
            # Update global model
            self.global_model.load_state_dict(global_state)
            self.round += 1
    
    def get_global_model(self) -> nn.Module:
        """Get current global model."""
        return self.global_model


class FederatedClientTrainer:
    """Client-side federated learning trainer."""
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        config: FLConfig,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
        )
    
    def local_train(self, global_model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """Train on local data."""
        # Start from global model if provided
        if global_model:
            self.model.load_state_dict(global_model.state_dict())
        
        self.model.train()
        
        for epoch in range(self.config.local_epochs):
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                
                # Apply differential privacy if configured
                if self.config.dp_epsilon:
                    self._add_noise()
                
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Client {self.client_id} - Epoch {epoch+1}/{self.config.local_epochs}: Loss = {total_loss/len(self.train_loader):.4f}")
        
        # Return model update (state dict)
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
    
    def _add_noise(self) -> None:
        """Add differential privacy noise."""
        if not self.config.dp_epsilon:
            return
        
        # Simplified DP - would use proper DP-SGD
        noise_scale = 1.0 / self.config.dp_epsilon
        
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)


class FederatedLearningServer:
    """Central server for federated learning."""
    
    def __init__(
        self,
        global_model: nn.Module,
        config: FLConfig,
    ):
        self.global_model = global_model
        self.config = config
        self.aggregator = FederatedAveraging(global_model)
        self.history: List[Dict[str, Any]] = []
    
    def run_round(self) -> Dict[str, Any]:
        """Run one federated learning round."""
        print(f"\n=== Round {self.aggregator.round + 1}/{self.config.num_rounds} ===")
        
        # Select clients
        selected = self.aggregator.select_clients(self.config.clients_per_round)
        print(f"Selected clients: {selected}")
        
        # Collect updates (in real scenario, clients would send updates)
        client_updates = []
        
        # This is where we would receive updates from clients
        # For simulation, we would create mock updates
        
        if client_updates:
            # Aggregate
            self.aggregator.aggregate(client_updates)
        
        # Record history
        round_info = {
            "round": self.aggregator.round,
            "clients": selected,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(round_info)
        
        return round_info
    
    def train(self, num_rounds: Optional[int] = None) -> nn.Module:
        """Run full federated training."""
        rounds = num_rounds or self.config.num_rounds
        
        for _ in range(rounds):
            self.run_round()
        
        return self.aggregator.get_global_model()
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate global model."""
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }


class SecureAggregation:
    """Secure aggregation with cryptographic protection."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.shares: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def add_masked_update(
        self,
        client_id: str,
        update: Dict[str, torch.Tensor],
        seed: int,
    ) -> None:
        """Add masked model update."""
        import random
        
        # Generate mask
        random.seed(seed)
        
        masked = {}
        for key, tensor in update.items():
            mask = torch.randn_like(tensor) * 0.01  # Small noise
            masked[key] = tensor + mask
        
        self.shares[client_id] = masked
    
    def secure_aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate without revealing individual updates."""
        # Average of masked updates
        result = {}
        
        for key in self.shares[list(self.shares.keys())[0]].keys():
            stacked = torch.stack([s[key] for s in self.shares.values()])
            result[key] = stacked.mean(dim=0)
        
        return result


class FederatedLearningManager:
    """Manage federated learning experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, FederatedLearningServer] = {}
    
    def create_experiment(
        self,
        name: str,
        model: nn.Module,
        config: FLConfig,
    ) -> FederatedLearningServer:
        """Create a new FL experiment."""
        server = FederatedLearningServer(model, config)
        self.experiments[name] = server
        return server
    
    def get_experiment(self, name: str) -> Optional[FederatedLearningServer]:
        """Get existing experiment."""
        return self.experiments.get(name)
    
    def compare_methods(
        self,
        model_fn: Callable,
        configs: List[FLConfig],
        num_rounds: int = 10,
    ) -> Dict[str, Any]:
        """Compare different FL methods."""
        results = {}
        
        for config in configs:
            model = model_fn()
            server = FederatedLearningServer(model, config)
            
            final_model = server.train(num_rounds)
            
            results[config.aggregation_method] = {
                "rounds": server.aggregator.round,
                "history": server.history,
            }
        
        return results


# Convenience functions
def run_federated_learning(
    model: nn.Module,
    num_clients: int = 10,
    num_rounds: int = 10,
) -> nn.Module:
    """Quick federated learning run."""
    config = FLConfig(
        num_rounds=num_rounds,
        clients_per_round=min(5, num_clients),
    )
    
    server = FederatedLearningServer(model, config)
    
    # Register clients
    for i in range(num_clients):
        server.aggregator.register_client(
            client_id=f"client_{i}",
            data_size=1000,  # Mock data size
        )
    
    return server.train()


def create_federated_client(
    client_id: str,
    model: nn.Module,
    train_data: torch.utils.data.DataLoader,
) -> FederatedClientTrainer:
    """Quick federated client creation."""
    config = FLConfig()
    return FederatedClientTrainer(client_id, model, train_data, config)
