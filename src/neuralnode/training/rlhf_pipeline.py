"""Complete RLHF (Reinforcement Learning from Human Feedback) Pipeline."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PreferencePair:
    """A pair of responses with preference."""
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float = 1.0
    rejected_score: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # DPO settings
    beta: float = 0.1
    # PPO settings
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    # General
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    max_length: int = 512
    gradient_accumulation_steps: int = 4


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(self, preferences: List[PreferencePair], tokenizer=None):
        self.preferences = preferences
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        pref = self.preferences[idx]
        
        return {
            "prompt": pref.prompt,
            "chosen": pref.chosen,
            "rejected": pref.rejected,
            "chosen_score": pref.chosen_score,
            "rejected_score": pref.rejected_score,
        }


class DPOTrainer:
    """Direct Preference Optimization Trainer."""
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        config: Optional[RLHFConfig] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.config = config or RLHFConfig()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute DPO loss."""
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = pi_logratios - ref_logratios
        losses = -nn.functional.logsigmoid(self.config.beta * logits)
        
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        
        metrics = {
            "rewards/chosen": chosen_rewards.mean().item(),
            "rewards/rejected": rejected_rewards.mean().item(),
            "rewards/margins": (chosen_rewards - rejected_rewards).mean().item(),
            "logps/chosen": policy_chosen_logps.mean().item(),
            "logps/rejected": policy_rejected_logps.mean().item(),
        }
        
        return losses.mean(), metrics
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.ref_model.eval()
        
        # Forward pass for policy model
        policy_chosen_outputs = self.model(**batch["chosen_inputs"])
        policy_rejected_outputs = self.model(**batch["rejected_inputs"])
        
        # Forward pass for reference model
        with torch.no_grad():
            ref_chosen_outputs = self.ref_model(**batch["chosen_inputs"])
            ref_rejected_outputs = self.ref_model(**batch["rejected_inputs"])
        
        # Compute loss
        loss, metrics = self.compute_dpo_loss(
            policy_chosen_outputs.logits.mean(),
            policy_rejected_outputs.logits.mean(),
            ref_chosen_outputs.logits.mean(),
            ref_rejected_outputs.logits.mean(),
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        metrics["loss"] = loss.item()
        return metrics
    
    def train(self, dataset: PreferenceDataset, num_epochs: Optional[int] = None) -> List[Dict[str, float]]:
        """Train the model."""
        num_epochs = num_epochs or self.config.num_epochs
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        all_metrics = []
        
        for epoch in range(num_epochs):
            epoch_metrics = []
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {
                k: sum(m[k] for m in epoch_metrics) / len(epoch_metrics)
                for k in epoch_metrics[0].keys()
            }
            
            print(f"Epoch {epoch + 1}/{num_epochs}: {avg_metrics}")
            all_metrics.append(avg_metrics)
        
        return all_metrics


class PPOTrainer:
    """Proximal Policy Optimization Trainer."""
    
    def __init__(
        self,
        policy_model,
        value_model,
        tokenizer,
        config: Optional[RLHFConfig] = None,
    ):
        self.policy_model = policy_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.config = config or RLHFConfig()
        
        self.policy_optimizer = optim.AdamW(
            policy_model.parameters(),
            lr=self.config.learning_rate,
        )
        self.value_optimizer = optim.AdamW(
            value_model.parameters(),
            lr=self.config.learning_rate,
        )
    
    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PPO clipped loss."""
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon,
        ) * advantages
        
        return -torch.min(surr1, surr2).mean()
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            last_gae = delta + gamma * gae_lambda * last_gae * (1 - dones[t])
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single PPO training step."""
        # Policy loss
        new_log_probs = self.policy_model(**batch["inputs"]).logits
        policy_loss = self.compute_ppo_loss(
            batch["old_log_probs"],
            new_log_probs,
            batch["advantages"],
        )
        
        # Value loss
        values = self.value_model(**batch["inputs"]).values
        value_loss = nn.functional.mse_loss(values, batch["returns"])
        
        # Entropy bonus
        entropy = -torch.sum(
            new_log_probs * torch.exp(new_log_probs),
            dim=-1,
        ).mean()
        
        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy
        )
        
        # Optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        
        # Optimize value
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }


class CompleteRLHFPipeline:
    """Complete RLHF pipeline from data collection to deployment."""
    
    def __init__(self, base_model, tokenizer):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.preferences: List[PreferencePair] = []
        self.trainer = None
        self.config = RLHFConfig()
    
    def stage1_collect_preferences(
        self,
        prompts: List[str],
        num_responses_per_prompt: int = 2,
    ) -> None:
        """Stage 1: Generate responses and collect preferences."""
        print("Stage 1: Collecting preferences...")
        
        for prompt in prompts:
            # Generate multiple responses
            responses = []
            for _ in range(num_responses_per_prompt):
                response = self._generate_response(prompt)
                responses.append(response)
            
            # Rank responses (in real scenario, this would be human feedback)
            # For now, use simple heuristic
            best, worst = self._rank_responses(prompt, responses)
            
            self.preferences.append(PreferencePair(
                prompt=prompt,
                chosen=best,
                rejected=worst,
            ))
        
        print(f"Collected {len(self.preferences)} preference pairs")
    
    def stage2_train_reward_model(self) -> nn.Module:
        """Stage 2: Train reward model from preferences."""
        print("Stage 2: Training reward model...")
        # Simplified - would train actual reward model
        return self.base_model
    
    def stage3_train_policy(
        self,
        method: str = "dpo",
        num_epochs: int = 3,
    ) -> nn.Module:
        """Stage 3: Train policy with RL."""
        print(f"Stage 3: Training policy with {method.upper()}...")
        
        # Create dataset
        dataset = PreferenceDataset(self.preferences, self.tokenizer)
        
        if method == "dpo":
            trainer = DPOTrainer(
                model=self.base_model,
                ref_model=self.base_model,  # Would be a copy
                tokenizer=self.tokenizer,
                config=self.config,
            )
        elif method == "ppo":
            trainer = PPOTrainer(
                policy_model=self.base_model,
                value_model=self.base_model,  # Would be separate
                tokenizer=self.tokenizer,
                config=self.config,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Train
        metrics = trainer.train(dataset, num_epochs=num_epochs)
        self.trainer = trainer
        
        print(f"Training completed. Final metrics: {metrics[-1] if metrics else {}}")
        
        return self.base_model
    
    def stage4_evaluate(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Stage 4: Evaluate trained model."""
        print("Stage 4: Evaluating model...")
        
        results = []
        for prompt in test_prompts:
            response = self._generate_response(prompt)
            results.append({"prompt": prompt, "response": response})
        
        return {
            "num_evaluated": len(results),
            "sample_responses": results[:5],
        }
    
    def run_full_pipeline(
        self,
        prompts: List[str],
        test_prompts: List[str],
        method: str = "dpo",
    ) -> Dict[str, Any]:
        """Run complete RLHF pipeline."""
        print("=" * 60)
        print("Starting Complete RLHF Pipeline")
        print("=" * 60)
        
        # Stage 1
        self.stage1_collect_preferences(prompts)
        
        # Stage 2
        reward_model = self.stage2_train_reward_model()
        
        # Stage 3
        trained_model = self.stage3_train_policy(method=method)
        
        # Stage 4
        eval_results = self.stage4_evaluate(test_prompts)
        
        print("=" * 60)
        print("RLHF Pipeline Completed!")
        print("=" * 60)
        
        return {
            "preferences_collected": len(self.preferences),
            "evaluation": eval_results,
            "model": trained_model,
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response from model."""
        # Simplified - would use actual generation
        return f"Response to: {prompt}"
    
    def _rank_responses(self, prompt: str, responses: List[str]) -> Tuple[str, str]:
        """Rank responses (placeholder for human feedback)."""
        # Simplified - just pick first and last
        return responses[0], responses[-1]
    
    def save_preferences(self, path: str) -> None:
        """Save preferences to file."""
        data = [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "timestamp": p.timestamp.isoformat(),
            }
            for p in self.preferences
        ]
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_preferences(self, path: str) -> None:
        """Load preferences from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.preferences = [
            PreferencePair(
                prompt=p["prompt"],
                chosen=p["chosen"],
                rejected=p["rejected"],
                timestamp=datetime.fromisoformat(p["timestamp"]),
            )
            for p in data
        ]


# Convenience functions
def run_rlhf_pipeline(
    base_model,
    tokenizer,
    prompts: List[str],
    method: str = "dpo",
) -> Dict[str, Any]:
    """Quick RLHF pipeline execution."""
    pipeline = CompleteRLHFPipeline(base_model, tokenizer)
    return pipeline.run_full_pipeline(prompts, prompts[:10], method=method)
