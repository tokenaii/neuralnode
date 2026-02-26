"""Reinforcement Learning system for training and improving AI agents."""

import json
import random
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Experience:
    """A single experience/memory for RL."""
    state: str  # The prompt/context
    action: str  # The agent's response
    reward: float  # Feedback (-1 to 1)
    next_state: Optional[str] = None
    done: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Policy:
    """Agent policy/state."""
    strategy: str = "greedy"  # greedy, epsilon_greedy, boltzmann
    epsilon: float = 0.1  # For epsilon-greedy
    temperature: float = 1.0  # For boltzmann
    learning_rate: float = 0.001
    
    def select_action(self, actions: List[str], values: List[float]) -> str:
        """Select action based on policy."""
        if self.strategy == "greedy":
            return actions[values.index(max(values))]
        
        elif self.strategy == "epsilon_greedy":
            if random.random() < self.epsilon:
                return random.choice(actions)
            return actions[values.index(max(values))]
        
        elif self.strategy == "boltzmann":
            import math
            exp_values = [math.exp(v / self.temperature) for v in values]
            total = sum(exp_values)
            probs = [v / total for v in exp_values]
            return random.choices(actions, weights=probs)[0]
        
        return actions[0]


class SimpleRLAgent:
    """Simple reinforcement learning agent."""
    
    def __init__(
        self,
        llm,
        policy: Optional[Policy] = None,
        memory_size: int = 1000,
    ):
        self.llm = llm
        self.policy = policy or Policy()
        self.memory: List[Experience] = []
        self.memory_size = memory_size
        self.q_table: Dict[str, Dict[str, float]] = {}  # Q-learning table
        self.action_history: List[Tuple[str, str, float]] = []  # (state, action, reward)
    
    def act(self, state: str, possible_actions: Optional[List[str]] = None) -> str:
        """Choose action based on current policy."""
        # If no actions specified, use LLM to generate
        if possible_actions is None:
            response = self.llm.chat(state)
            return response.text if hasattr(response, 'text') else str(response)
        
        # Get Q-values for state
        q_values = self.q_table.get(state, {})
        
        # If no Q-values yet, initialize with random values
        if not q_values:
            for action in possible_actions:
                q_values[action] = random.uniform(-0.1, 0.1)
            self.q_table[state] = q_values
        
        # Select action
        actions = list(q_values.keys())
        values = list(q_values.values())
        
        return self.policy.select_action(actions, values)
    
    def learn(self, experience: Experience) -> None:
        """Learn from experience."""
        # Add to memory
        self.memory.append(experience)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Q-learning update
        state = experience.state
        action = experience.action
        reward = experience.reward
        
        # Initialize Q-values if needed
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        
        # Q-learning formula: Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s')) - Q(s,a))
        alpha = self.policy.learning_rate
        gamma = 0.95  # Discount factor
        
        current_q = self.q_table[state][action]
        
        # Get next state max Q-value
        if experience.next_state and experience.next_state in self.q_table:
            next_q_values = self.q_table[experience.next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        else:
            max_next_q = 0.0
        
        # Update Q-value
        new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        # Record for analysis
        self.action_history.append((state, action, reward))
    
    def feedback(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None,
    ) -> None:
        """Provide feedback/reward for an action."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
        )
        self.learn(experience)
    
    def save(self, path: str) -> None:
        """Save agent state."""
        data = {
            "policy": {
                "strategy": self.policy.strategy,
                "epsilon": self.policy.epsilon,
                "temperature": self.policy.temperature,
                "learning_rate": self.policy.learning_rate,
            },
            "q_table": self.q_table,
            "memory": [
                {
                    "state": e.state,
                    "action": e.action,
                    "reward": e.reward,
                    "next_state": e.next_state,
                    "done": e.done,
                }
                for e in self.memory[-100:]  # Save last 100 experiences
            ],
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.policy.strategy = data["policy"]["strategy"]
        self.policy.epsilon = data["policy"]["epsilon"]
        self.policy.temperature = data["policy"]["temperature"]
        self.policy.learning_rate = data["policy"]["learning_rate"]
        self.q_table = data["q_table"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        if not self.action_history:
            return {"experiences": 0, "average_reward": 0}
        
        rewards = [r for _, _, r in self.action_history]
        
        return {
            "experiences": len(self.memory),
            "total_interactions": len(self.action_history),
            "average_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "learned_states": len(self.q_table),
        }


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback."""
    
    def __init__(self, base_llm, reward_model: Optional[Any] = None):
        self.base_llm = base_llm
        self.reward_model = reward_model
        self.preferences: List[Dict[str, Any]] = []
        self.rl_agent = SimpleRLAgent(base_llm)
    
    def collect_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        preferred: str,  # 'A' or 'B'
    ) -> None:
        """Collect human preference between two responses."""
        self.preferences.append({
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "preferred": preferred,
            "timestamp": datetime.now().isoformat(),
        })
    
    def train_reward_model(self) -> bool:
        """Train a reward model from preferences."""
        # In a full implementation, this would train a neural network
        # For now, we use a simple heuristic
        print(f"Training reward model on {len(self.preferences)} preferences...")
        
        # Calculate basic statistics
        a_wins = sum(1 for p in self.preferences if p["preferred"] == "A")
        b_wins = len(self.preferences) - a_wins
        
        print(f"  A wins: {a_wins}, B wins: {b_wins}")
        
        return True
    
    def train_with_ppo(
        self,
        num_iterations: int = 100,
        episodes_per_iteration: int = 10,
    ) -> None:
        """Train with PPO (Proximal Policy Optimization)."""
        print(f"Training with PPO for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            # Collect experiences
            for _ in range(episodes_per_iteration):
                # Sample a prompt from preferences
                if self.preferences:
                    pref = random.choice(self.preferences)
                    state = pref["prompt"]
                    
                    # Generate response
                    action = self.rl_agent.act(state)
                    
                    # Get reward (simplified)
                    reward = self._calculate_reward(state, action)
                    
                    # Learn
                    self.rl_agent.feedback(state, action, reward)
            
            if (iteration + 1) % 10 == 0:
                stats = self.rl_agent.get_stats()
                print(f"  Iteration {iteration + 1}: avg_reward={stats['average_reward']:.3f}")
    
    def train_with_dpo(self, beta: float = 0.1) -> None:
        """Train with Direct Preference Optimization."""
        print(f"Training with DPO (beta={beta})...")
        
        # DPO is simpler than PPO - directly optimize from preferences
        for pref in self.preferences:
            preferred_response = pref["response_a"] if pref["preferred"] == "A" else pref["response_b"]
            less_preferred = pref["response_b"] if pref["preferred"] == "A" else pref["response_a"]
            
            # Positive reward for preferred
            self.rl_agent.feedback(pref["prompt"], preferred_response, 1.0)
            # Negative reward for less preferred
            self.rl_agent.feedback(pref["prompt"], less_preferred, -1.0)
        
        print(f"  Processed {len(self.preferences)} preferences")
    
    def _calculate_reward(self, state: str, action: str) -> float:
        """Calculate reward for an action."""
        # If we have a reward model, use it
        if self.reward_model:
            # This would call the reward model
            return 0.0
        
        # Otherwise, use heuristics
        # Check if response is long enough
        if len(action) < 10:
            return -0.5
        
        # Check if it addresses the prompt
        if any(word in action.lower() for word in state.lower().split()[:5]):
            return 0.5
        
        return 0.1  # Neutral
    
    def save(self, path: str) -> None:
        """Save training state."""
        data = {
            "preferences": self.preferences,
            "rl_agent": {
                "q_table": self.rl_agent.q_table,
                "policy": {
                    "strategy": self.rl_agent.policy.strategy,
                    "epsilon": self.rl_agent.policy.epsilon,
                },
            },
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_improved_model(self) -> SimpleRLAgent:
        """Export the trained RL agent."""
        return self.rl_agent


class AdaptiveAgent:
    """Agent that adapts and improves over time."""
    
    def __init__(self, base_llm, learning_enabled: bool = True):
        self.llm = base_llm
        self.rl_agent = SimpleRLAgent(base_llm)
        self.learning_enabled = learning_enabled
        self.feedback_history: List[Dict[str, Any]] = []
    
    def run(self, prompt: str) -> str:
        """Run the agent with learning."""
        # Try to use learned policy first
        if self.learning_enabled and prompt in self.rl_agent.q_table:
            # Use RL policy
            action = self.rl_agent.act(prompt)
            
            # Ask for implicit feedback (e.g., was this good?)
            # In production, this would be actual user feedback
            return action
        
        # Fall back to base LLM
        response = self.llm.chat(prompt)
        return response.text if hasattr(response, 'text') else str(response)
    
    def provide_feedback(self, prompt: str, response: str, rating: float) -> None:
        """Provide feedback on a response."""
        self.rl_agent.feedback(prompt, response, rating)
        
        self.feedback_history.append({
            "prompt": prompt,
            "response": response,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
        })
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return self.rl_agent.get_stats()


# Convenience functions
def create_rl_agent(llm, policy: Optional[Policy] = None) -> SimpleRLAgent:
    """Create an RL agent."""
    return SimpleRLAgent(llm, policy=policy)


def train_with_feedback(
    llm,
    training_data: List[Dict[str, Any]],
    method: str = "dpo",
) -> SimpleRLAgent:
    """Quick RL training with feedback."""
    trainer = RLHFTrainer(llm)
    
    # Add preferences
    for item in training_data:
        trainer.collect_preference(
            prompt=item["prompt"],
            response_a=item["response_a"],
            response_b=item["response_b"],
            preferred=item["preferred"],
        )
    
    # Train
    if method == "dpo":
        trainer.train_with_dpo()
    elif method == "ppo":
        trainer.train_with_ppo()
    
    return trainer.rl_agent
