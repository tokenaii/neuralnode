"""Agent Orchestrator for multi-agent collaboration."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class AgentRole(Enum):
    """Roles for agents in orchestration."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    CRITIC = "critic"
    PLANNER = "planner"
    EXECUTOR = "executor"
    COORDINATOR = "coordinator"


@dataclass
class Task:
    """A task for an agent."""
    id: str
    description: str
    role: AgentRole
    dependencies: List[str] = field(default_factory=list)
    output: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    agent_id: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for an agent in the orchestrator."""
    id: str
    name: str
    role: AgentRole
    llm_provider: str
    model: str
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None


class AgentOrchestrator:
    """Orchestrate multiple agents working together."""
    
    def __init__(self, llm_factory: Optional[Callable] = None):
        self.agents: Dict[str, Any] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.tasks: Dict[str, Task] = {}
        self.llm_factory = llm_factory
        self._executor = ThreadPoolExecutor(max_workers=5)
    
    def add_agent(self, config: AgentConfig, agent_instance: Optional[Any] = None) -> None:
        """Add an agent to the orchestrator."""
        self.agent_configs[config.id] = config
        
        if agent_instance:
            self.agents[config.id] = agent_instance
        elif self.llm_factory:
            # Create agent using factory
            self.agents[config.id] = self.llm_factory(
                provider=config.llm_provider,
                model=config.model,
            )
    
    def create_workflow(self, tasks: List[Task]) -> str:
        """Create a workflow with multiple tasks."""
        workflow_id = f"workflow_{len(self.tasks)}"
        
        for task in tasks:
            task.id = f"{workflow_id}_{task.id}"
            self.tasks[task.id] = task
        
        return workflow_id
    
    def execute_workflow(self, workflow_id: str, parallel: bool = True) -> Dict[str, Any]:
        """Execute a workflow."""
        # Get tasks for this workflow
        workflow_tasks = [
            task for task in self.tasks.values()
            if task.id.startswith(workflow_id)
        ]
        
        if parallel:
            return self._execute_parallel(workflow_tasks)
        else:
            return self._execute_sequential(workflow_tasks)
    
    def _execute_parallel(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks in parallel where possible."""
        completed = set()
        results = {}
        
        while len(completed) < len(tasks):
            # Find tasks that are ready (dependencies completed)
            ready_tasks = [
                task for task in tasks
                if task.status == "pending"
                and all(dep in completed for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                break
            
            # Submit ready tasks
            futures = {}
            for task in ready_tasks:
                task.status = "running"
                future = self._executor.submit(self._execute_task, task)
                futures[future] = task
            
            # Collect results
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    task.output = result
                    task.status = "completed"
                    results[task.id] = result
                    completed.add(task.id)
                except Exception as e:
                    task.status = "failed"
                    results[task.id] = f"Error: {str(e)}"
                    completed.add(task.id)
        
        return results
    
    def _execute_sequential(self, tasks: List[Task]) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        results = {}
        
        for task in sorted(tasks, key=lambda t: len(t.dependencies)):
            task.status = "running"
            
            try:
                result = self._execute_task(task)
                task.output = result
                task.status = "completed"
                results[task.id] = result
            except Exception as e:
                task.status = "failed"
                results[task.id] = f"Error: {str(e)}"
        
        return results
    
    def _execute_task(self, task: Task) -> str:
        """Execute a single task."""
        # Find best agent for this role
        agent_id = self._select_agent_for_role(task.role)
        
        if not agent_id or agent_id not in self.agents:
            return f"No agent available for role: {task.role.value}"
        
        agent = self.agents[agent_id]
        task.agent_id = agent_id
        
        # Execute based on role
        if hasattr(agent, 'agent'):
            # It's a NeuralNode with agent capability
            agent_instance = agent.agent()
            return agent_instance.run(task.description)
        elif hasattr(agent, 'chat'):
            # It's a basic LLM
            response = agent.chat(task.description)
            return response.text if hasattr(response, 'text') else str(response)
        else:
            return str(agent)
    
    def _select_agent_for_role(self, role: AgentRole) -> Optional[str]:
        """Select the best agent for a role."""
        candidates = [
            agent_id for agent_id, config in self.agent_configs.items()
            if config.role == role
        ]
        
        return candidates[0] if candidates else None
    
    def run_collaborative_task(
        self,
        task_description: str,
        agent_roles: List[AgentRole],
    ) -> Dict[str, Any]:
        """Run a task with multiple agents collaborating."""
        # Create subtasks for each role
        tasks = []
        
        for i, role in enumerate(agent_roles):
            task = Task(
                id=f"task_{i}",
                description=f"{role.value}: {task_description}",
                role=role,
                dependencies=[f"task_{i-1}"] if i > 0 else [],
            )
            tasks.append(task)
        
        # Create and execute workflow
        workflow_id = self.create_workflow(tasks)
        return self.execute_workflow(workflow_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "agents": len(self.agents),
            "tasks": {
                "total": len(self.tasks),
                "pending": sum(1 for t in self.tasks.values() if t.status == "pending"),
                "running": sum(1 for t in self.tasks.values() if t.status == "running"),
                "completed": sum(1 for t in self.tasks.values() if t.status == "completed"),
                "failed": sum(1 for t in self.tasks.values() if t.status == "failed"),
            },
        }


class SwarmOrchestrator(AgentOrchestrator):
    """Swarm-based multi-agent orchestration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.swarm_size = 3
    
    def swarm_execute(self, task: str, num_agents: int = 3) -> List[str]:
        """Execute task with multiple agents and vote on best result."""
        # Create temporary agents for swarm
        swarm_agents = []
        for i in range(num_agents):
            if self.llm_factory:
                agent = self.llm_factory()
                swarm_agents.append(agent)
        
        # Execute in parallel
        results = []
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [executor.submit(self._ask_agent, agent, task) for agent in swarm_agents]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {str(e)}")
        
        return results
    
    def _ask_agent(self, agent: Any, task: str) -> str:
        """Ask a single agent."""
        if hasattr(agent, 'chat'):
            response = agent.chat(task)
            return response.text if hasattr(response, 'text') else str(response)
        return str(agent)
    
    def vote_on_best_result(self, results: List[str], criteria: Optional[str] = None) -> str:
        """Vote on the best result from swarm."""
        if not results:
            return "No results to vote on"
        
        # Simple voting: return the most common result
        from collections import Counter
        
        # Count similar results (exact match)
        counts = Counter(results)
        most_common = counts.most_common(1)[0][0]
        
        return most_common


# Convenience functions
def create_research_team(llm_factory: Callable) -> AgentOrchestrator:
    """Create a research team orchestrator."""
    orchestrator = AgentOrchestrator(llm_factory)
    
    # Add research team agents
    orchestrator.add_agent(AgentConfig(
        id="researcher_1",
        name="Web Researcher",
        role=AgentRole.RESEARCHER,
        llm_provider="openai",
        model="gpt-4",
    ))
    
    orchestrator.add_agent(AgentConfig(
        id="writer_1",
        name="Content Writer",
        role=AgentRole.WRITER,
        llm_provider="openai",
        model="gpt-4",
    ))
    
    orchestrator.add_agent(AgentConfig(
        id="critic_1",
        name="Quality Critic",
        role=AgentRole.CRITIC,
        llm_provider="openai",
        model="gpt-4",
    ))
    
    return orchestrator
