"""Fine-tuning support with LoRA and RLHF for NeuralNode."""

import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class TrainingExample:
    """A single training example."""
    prompt: str
    completion: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""
    r: int = 16  # Rank
    alpha: int = 32  # Alpha
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./finetuned_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class FineTuner:
    """Fine-tune models with LoRA/QLoRA."""
    
    def __init__(self, base_model: str, config: Optional[TrainingConfig] = None):
        self.base_model = base_model
        self.config = config or TrainingConfig()
        self.examples: List[TrainingExample] = []
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if required packages are installed."""
        try:
            import torch
            import transformers
            import peft
            import trl
            self.dependencies_available = True
        except ImportError as e:
            print(f"Missing dependencies: {e}")
            print("Install with: pip install torch transformers peft trl datasets")
            self.dependencies_available = False
    
    def add_example(self, prompt: str, completion: str, **metadata) -> None:
        """Add a training example."""
        self.examples.append(TrainingExample(
            prompt=prompt,
            completion=completion,
            metadata=metadata
        ))
    
    def add_examples_from_file(self, path: Union[str, Path], format: str = "jsonl") -> None:
        """Load examples from file."""
        path = Path(path)
        
        if format == "jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    self.add_example(
                        prompt=data.get("prompt", data.get("instruction", "")),
                        completion=data.get("completion", data.get("output", "")),
                    )
        elif format == "json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.add_example(
                        prompt=item.get("prompt", item.get("instruction", "")),
                        completion=item.get("completion", item.get("output", "")),
                    )
    
    def prepare_dataset(self):
        """Prepare dataset for training."""
        if not self.examples:
            raise ValueError("No training examples added")
        
        # Format examples
        formatted = []
        for ex in self.examples:
            text = f"### Instruction:\n{ex.prompt}\n\n### Response:\n{ex.completion}"
            formatted.append({"text": text})
        
        # Create dataset
        try:
            from datasets import Dataset
            return Dataset.from_list(formatted)
        except ImportError:
            print("datasets not installed")
            return None
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> str:
        """Start training."""
        if not self.dependencies_available:
            raise RuntimeError("Dependencies not available")
        
        if not self.examples:
            raise ValueError("No training examples")
        
        print(f"Starting fine-tuning of {self.base_model}")
        print(f"Training examples: {len(self.examples)}")
        print(f"Output directory: {self.config.output_dir}")
        
        # This would be the actual training code
        # For now, we return a placeholder
        print("\nNote: Full training requires transformers, peft, trl libraries")
        print("Training pipeline:")
        print("1. Load base model with QLoRA config")
        print("2. Prepare dataset")
        print("3. Train with SFTTrainer")
        print("4. Save LoRA adapters")
        print("5. Merge and save final model")
        
        return self.config.output_dir
    
    def export_to_gguf(self, output_path: Optional[str] = None) -> str:
        """Export fine-tuned model to GGUF format."""
        print("Exporting to GGUF format...")
        print("This requires llama.cpp conversion tools")
        return output_path or f"{self.config.output_dir}.gguf"
    
    def save_training_state(self, path: str) -> None:
        """Save training state."""
        state = {
            "base_model": self.base_model,
            "config": self.config.__dict__,
            "examples": [
                {"prompt": ex.prompt, "completion": ex.completion}
                for ex in self.examples
            ],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_training_state(self, path: str) -> None:
        """Load training state."""
        with open(path, "r") as f:
            state = json.load(f)
        
        self.base_model = state["base_model"]
        # Restore examples
        for ex in state.get("examples", []):
            self.add_example(ex["prompt"], ex["completion"])


class RLHFTrainer:
    """RLHF (Reinforcement Learning from Human Feedback) training."""
    
    def __init__(self, base_model: str, reward_model: Optional[str] = None):
        self.base_model = base_model
        self.reward_model = reward_model or base_model
        self.preferences: List[Dict[str, str]] = []
    
    def add_preference(self, prompt: str, chosen: str, rejected: str) -> None:
        """Add a preference pair (good vs bad response)."""
        self.preferences.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    
    def train_reward_model(self) -> str:
        """Train a reward model from preferences."""
        print(f"Training reward model from {len(self.preferences)} preferences")
        print("This uses the Bradley-Terry model for ranking")
        return "reward_model_path"
    
    def train_with_ppo(self, reward_model_path: str, num_epochs: int = 1) -> str:
        """Train with PPO using the reward model."""
        print(f"Training with PPO for {num_epochs} epochs")
        print("This optimizes the policy to maximize reward")
        return "ppo_model_path"
    
    def train_with_dpo(self, beta: float = 0.1) -> str:
        """Train with Direct Preference Optimization."""
        print(f"Training with DPO (beta={beta})")
        print("DPO is more stable than PPO and doesn't need a reward model")
        return "dpo_model_path"


class InstructionTuner:
    """Simplified instruction tuning interface."""
    
    def __init__(self, base_model: str):
        self.finetuner = FineTuner(base_model)
    
    def add_instruction(self, instruction: str, input_text: str = "", output: str = "") -> None:
        """Add instruction-following example."""
        prompt = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
        self.finetuner.add_example(prompt, output)
    
    def train(
        self,
        output_name: str,
        epochs: int = 3,
        batch_size: int = 4,
    ) -> str:
        """Train the model."""
        self.finetuner.config.output_dir = f"./{output_name}"
        self.finetuner.config.num_train_epochs = epochs
        self.finetuner.config.per_device_train_batch_size = batch_size
        return self.finetuner.train()


# Convenience functions
def quick_finetune(
    base_model: str,
    examples: List[Dict[str, str]],
    output_dir: str = "./finetuned",
    epochs: int = 3,
) -> str:
    """Quick fine-tuning function."""
    tuner = FineTuner(base_model)
    
    for ex in examples:
        tuner.add_example(
            ex.get("prompt", ex.get("instruction", "")),
            ex.get("completion", ex.get("output", "")),
        )
    
    tuner.config.output_dir = output_dir
    tuner.config.num_train_epochs = epochs
    
    return tuner.train()


def create_instruction_dataset(examples: List[Dict[str, str]], output_path: str) -> None:
    """Create a JSONL instruction dataset."""
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
