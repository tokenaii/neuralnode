"""Model Compression via Pruning and Knowledge Distillation."""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import copy


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    method: str = "pruning"  # pruning, quantization, distillation
    target_sparsity: float = 0.5  # For pruning: 0.0 to 1.0
    target_size_mb: Optional[float] = None
    structured: bool = False  # Structured vs unstructured pruning
    distillation_temperature: float = 2.0
    distillation_alpha: float = 0.7  # Weight for distillation loss


class ModelPruner:
    """Prune models to reduce size."""
    
    def __init__(self, model: nn.Module, config: Optional[CompressionConfig] = None):
        self.model = model
        self.config = config or CompressionConfig()
        self.original_params = sum(p.numel() for p in model.parameters())
    
    def unstructured_pruning(
        self,
        amount: float = 0.3,
        method: str = "l1_unstructured",
    ) -> nn.Module:
        """Unstructured pruning (individual weights)."""
        parameters_to_prune = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        if method == "l1_unstructured":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        elif method == "random_unstructured":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount,
            )
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return self.model
    
    def structured_pruning(
        self,
        amount: float = 0.3,
        n: int = 2,
        m: int = 4,
    ) -> nn.Module:
        """Structured pruning (entire channels/heads)."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # N:M structured sparsity
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=n,
                    m=m,
                    dim=0,
                )
        
        return self.model
    
    def magnitude_pruning(self, threshold: float = 0.01) -> nn.Module:
        """Prune weights below magnitude threshold."""
        with torch.no_grad():
            for param in self.model.parameters():
                mask = torch.abs(param) > threshold
                param.mul_(mask)
        
        return self.model
    
    def iterative_pruning(
        self,
        target_sparsity: float = 0.5,
        num_iterations: int = 5,
        finetune_fn: Optional[Callable] = None,
    ) -> nn.Module:
        """Iterative pruning with fine-tuning."""
        sparsity_per_iter = target_sparsity / num_iterations
        
        for i in range(num_iterations):
            print(f"Iteration {i+1}/{num_iterations}: Pruning {sparsity_per_iter:.2%}")
            
            # Prune
            self.unstructured_pruning(amount=sparsity_per_iter)
            
            # Fine-tune if function provided
            if finetune_fn:
                print("Fine-tuning after pruning...")
                finetune_fn(self.model)
        
        return self.model
    
    def get_sparsity(self) -> Dict[str, float]:
        """Calculate current sparsity."""
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return {
            "total_params": total_params,
            "zero_params": zero_params,
            "sparsity": zero_params / total_params if total_params > 0 else 0,
            "original_params": self.original_params,
            "compression_ratio": self.original_params / (total_params - zero_params) if (total_params - zero_params) > 0 else 1,
        }


class KnowledgeDistiller:
    """Knowledge Distillation from teacher to student model."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        config: Optional[CompressionConfig] = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config or CompressionConfig()
        self.teacher.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """Compute knowledge distillation loss."""
        # Soft targets from teacher
        soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)
        
        # Distillation loss (KL divergence)
        distillation_loss = nn.functional.kl_div(
            soft_prob,
            soft_targets,
            reduction='batchmean',
        ) * (temperature ** 2)
        
        # Hard target loss (standard cross-entropy)
        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        
        # Combined loss
        alpha = self.config.distillation_alpha
        return alpha * distillation_loss + (1 - alpha) * hard_loss
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
    ) -> float:
        """Train one epoch."""
        self.student.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(inputs)
                teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, 'logits') else teacher_outputs
            
            # Student forward
            student_outputs = self.student(inputs)
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else student_outputs
            
            # Compute loss
            loss = self.distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=self.config.distillation_temperature,
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def distill(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ) -> nn.Module:
        """Full distillation process."""
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        
        print(f"Starting distillation for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_dataloader, optimizer, device)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
        
        return self.student


class ModelCompressor:
    """Complete model compression pipeline."""
    
    def __init__(self):
        self.config = CompressionConfig()
    
    def compress(
        self,
        model: nn.Module,
        method: str = "pruning",
        target_reduction: float = 0.5,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model using specified method."""
        original_size = self._get_model_size(model)
        
        if method == "pruning":
            pruner = ModelPruner(model)
            compressed_model = pruner.iterative_pruning(
                target_sparsity=target_reduction,
            )
            stats = pruner.get_sparsity()
        
        elif method == "quantization":
            compressed_model = self._apply_quantization(model)
            stats = {"method": "quantization"}
        
        elif method == "distillation":
            # Would need teacher and student models
            compressed_model = model
            stats = {"method": "distillation", "note": "Requires separate student model"}
        
        else:
            raise ValueError(f"Unknown compression method: {method}")
        
        compressed_size = self._get_model_size(compressed_model)
        
        stats.update({
            "original_size_mb": original_size,
            "compressed_size_mb": compressed_size,
            "reduction_ratio": original_size / compressed_size if compressed_size > 0 else 1,
            "space_saved_percent": (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        })
        
        return compressed_model, stats
    
    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model


# Convenience functions
def prune_model(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
    """Quick model pruning."""
    pruner = ModelPruner(model)
    return pruner.unstructured_pruning(amount=sparsity)


def distill_model(
    teacher: nn.Module,
    student: nn.Module,
    train_data: torch.utils.data.DataLoader,
    num_epochs: int = 10,
) -> nn.Module:
    """Quick knowledge distillation."""
    distiller = KnowledgeDistiller(teacher, student)
    return distiller.distill(train_data, num_epochs=num_epochs)
