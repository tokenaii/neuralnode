"""Visual Workflow Builder for creating AI workflows with nodes."""

from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from pathlib import Path


class NodeType(Enum):
    """Types of workflow nodes."""
    INPUT = "input"
    LLM = "llm"
    TOOL = "tool"
    CONDITION = "condition"
    LOOP = "loop"
    OUTPUT = "output"
    MERGE = "merge"
    SPLIT = "split"
    DELAY = "delay"
    WEBHOOK = "webhook"


@dataclass
class NodePort:
    """Input/output port for a node."""
    id: str
    name: str
    type: str  # string, number, boolean, object, any
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass
class WorkflowNode:
    """A node in the workflow."""
    id: str
    type: NodeType
    name: str
    position: Dict[str, float]  # x, y coordinates
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[NodePort] = field(default_factory=list)
    outputs: List[NodePort] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)  # IDs of connected nodes


@dataclass
class WorkflowConnection:
    """Connection between two nodes."""
    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    condition: Optional[str] = None  # For conditional connections


class VisualWorkflow:
    """A visual workflow composed of nodes and connections."""
    
    def __init__(self, name: str = "Untitled Workflow"):
        self.id = str(uuid.uuid4())
        self.name = name
        self.nodes: Dict[str, WorkflowNode] = {}
        self.connections: Dict[str, WorkflowConnection] = {}
        self.metadata: Dict[str, Any] = {
            "created": str(uuid.uuid1()),
            "version": "1.0",
        }
    
    def add_node(
        self,
        node_type: NodeType,
        name: str,
        position: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
    ) -> WorkflowNode:
        """Add a node to the workflow."""
        node_id = str(uuid.uuid4())
        
        # Create default ports based on node type
        inputs, outputs = self._create_default_ports(node_type)
        
        node = WorkflowNode(
            id=node_id,
            type=node_type,
            name=name,
            position=position,
            config=config or {},
            inputs=inputs,
            outputs=outputs,
        )
        
        self.nodes[node_id] = node
        return node
    
    def _create_default_ports(self, node_type: NodeType) -> tuple:
        """Create default input/output ports for a node type."""
        if node_type == NodeType.INPUT:
            return [], [NodePort("out", "output", "any")]
        
        elif node_type == NodeType.LLM:
            return [
                NodePort("prompt", "prompt", "string", required=True),
                NodePort("system", "system_prompt", "string", required=False),
            ], [
                NodePort("response", "response", "string"),
                NodePort("tokens", "token_count", "number"),
            ]
        
        elif node_type == NodeType.TOOL:
            return [
                NodePort("input", "input", "any"),
                NodePort("tool", "tool_name", "string", required=True),
            ], [
                NodePort("result", "result", "any"),
                NodePort("success", "success", "boolean"),
            ]
        
        elif node_type == NodeType.CONDITION:
            return [
                NodePort("input", "input", "any"),
                NodePort("condition", "condition", "string", required=True),
            ], [
                NodePort("true", "true_branch", "any"),
                NodePort("false", "false_branch", "any"),
            ]
        
        elif node_type == NodeType.OUTPUT:
            return [NodePort("input", "input", "any")], []
        
        else:
            return [NodePort("in", "input", "any")], [NodePort("out", "output", "any")]
    
    def connect(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
        condition: Optional[str] = None,
    ) -> WorkflowConnection:
        """Connect two nodes."""
        conn_id = str(uuid.uuid4())
        
        connection = WorkflowConnection(
            id=conn_id,
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
            condition=condition,
        )
        
        self.connections[conn_id] = connection
        
        # Update node connections
        if source_node in self.nodes:
            self.nodes[source_node].connections.append(conn_id)
        
        return connection
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its connections."""
        if node_id not in self.nodes:
            return False
        
        # Remove connections
        connections_to_remove = [
            conn_id for conn_id, conn in self.connections.items()
            if conn.source_node == node_id or conn.target_node == node_id
        ]
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        # Remove node
        del self.nodes[node_id]
        return True
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the workflow."""
        results = {}
        execution_order = self._topological_sort()
        
        for node_id in execution_order:
            node = self.nodes[node_id]
            result = self._execute_node(node, context or {})
            results[node_id] = result
        
        return results
    
    def _topological_sort(self) -> List[str]:
        """Get execution order using topological sort."""
        # Find start nodes (nodes with no incoming connections)
        incoming = {node_id: 0 for node_id in self.nodes}
        
        for conn in self.connections.values():
            incoming[conn.target_node] += 1
        
        # Start with nodes that have no dependencies
        queue = [node_id for node_id, count in incoming.items() if count == 0]
        order = []
        
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            
            # Find nodes that depend on this one
            for conn in self.connections.values():
                if conn.source_node == node_id:
                    incoming[conn.target_node] -= 1
                    if incoming[conn.target_node] == 0:
                        queue.append(conn.target_node)
        
        return order
    
    def _execute_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute a single node."""
        if node.type == NodeType.INPUT:
            return node.config.get("value", context.get("input"))
        
        elif node.type == NodeType.LLM:
            prompt = node.config.get("prompt", "")
            # Would call LLM here
            return {"response": f"LLM response to: {prompt}", "tokens": 100}
        
        elif node.type == NodeType.TOOL:
            tool_name = node.config.get("tool_name", "")
            # Would execute tool here
            return {"result": f"Tool {tool_name} executed", "success": True}
        
        elif node.type == NodeType.CONDITION:
            condition = node.config.get("condition", "")
            # Evaluate condition
            return {"condition": condition, "result": True}
        
        elif node.type == NodeType.OUTPUT:
            return node.config.get("value", "output")
        
        return None
    
    def save(self, filepath: str) -> None:
        """Save workflow to file."""
        data = {
            "id": self.id,
            "name": self.name,
            "metadata": self.metadata,
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type.value,
                    "name": n.name,
                    "position": n.position,
                    "config": n.config,
                    "inputs": [{"id": p.id, "name": p.name, "type": p.type} for p in n.inputs],
                    "outputs": [{"id": p.id, "name": p.name, "type": p.type} for p in n.outputs],
                }
                for n in self.nodes.values()
            ],
            "connections": [
                {
                    "id": c.id,
                    "source": c.source_node,
                    "source_port": c.source_port,
                    "target": c.target_node,
                    "target_port": c.target_port,
                    "condition": c.condition,
                }
                for c in self.connections.values()
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "VisualWorkflow":
        """Load workflow from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        workflow = cls(name=data.get("name", "Loaded Workflow"))
        workflow.id = data.get("id", str(uuid.uuid4()))
        workflow.metadata = data.get("metadata", {})
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node = WorkflowNode(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                name=node_data["name"],
                position=node_data["position"],
                config=node_data.get("config", {}),
                inputs=[NodePort(p["id"], p["name"], p["type"]) for p in node_data.get("inputs", [])],
                outputs=[NodePort(p["id"], p["name"], p["type"]) for p in node_data.get("outputs", [])],
            )
            workflow.nodes[node.id] = node
        
        # Load connections
        for conn_data in data.get("connections", []):
            conn = WorkflowConnection(
                id=conn_data["id"],
                source_node=conn_data["source"],
                source_port=conn_data["source_port"],
                target_node=conn_data["target"],
                target_port=conn_data["target_port"],
                condition=conn_data.get("condition"),
            )
            workflow.connections[conn.id] = conn
        
        return workflow


class WorkflowBuilderUI:
    """Web-based UI for building workflows."""
    
    def __init__(self):
        self.current_workflow: Optional[VisualWorkflow] = None
    
    def create_new(self, name: str = "New Workflow") -> VisualWorkflow:
        """Create a new workflow."""
        self.current_workflow = VisualWorkflow(name)
        return self.current_workflow
    
    def add_llm_node(
        self,
        name: str = "LLM",
        position: Optional[Dict[str, float]] = None,
        model: str = "gpt-4",
        prompt: str = "",
    ) -> WorkflowNode:
        """Add an LLM node."""
        if not self.current_workflow:
            self.create_new()
        
        pos = position or {"x": 100, "y": 100}
        
        return self.current_workflow.add_node(
            node_type=NodeType.LLM,
            name=name,
            position=pos,
            config={"model": model, "prompt": prompt},
        )
    
    def add_tool_node(
        self,
        tool_name: str,
        position: Optional[Dict[str, float]] = None,
    ) -> WorkflowNode:
        """Add a tool node."""
        if not self.current_workflow:
            self.create_new()
        
        pos = position or {"x": 300, "y": 100}
        
        return self.current_workflow.add_node(
            node_type=NodeType.TOOL,
            name=f"Tool: {tool_name}",
            position=pos,
            config={"tool_name": tool_name},
        )
    
    def connect_nodes(
        self,
        source: WorkflowNode,
        target: WorkflowNode,
        condition: Optional[str] = None,
    ) -> WorkflowConnection:
        """Connect two nodes."""
        if not self.current_workflow:
            raise ValueError("No workflow created")
        
        return self.current_workflow.connect(
            source_node=source.id,
            source_port=source.outputs[0].id if source.outputs else "out",
            target_node=target.id,
            target_port=target.inputs[0].id if target.inputs else "in",
            condition=condition,
        )
    
    def export_to_code(self) -> str:
        """Export workflow to Python code."""
        if not self.current_workflow:
            return ""
        
        code = ["# Generated from NeuralNode Workflow Builder", ""]
        code.append("import nn")
        code.append("")
        code.append("# Initialize AI")
        code.append("ai = nn.NeuralNode(provider=\"auto\")")
        code.append("")
        
        for node in self.current_workflow.nodes.values():
            if node.type == NodeType.LLM:
                code.append(f"# LLM: {node.name}")
                code.append(f'result_{node.id[:8]} = ai.chat("{node.config.get("prompt", "")}")')
                code.append("")
            
            elif node.type == NodeType.TOOL:
                code.append(f"# Tool: {node.name}")
                code.append(f'# Tool execution: {node.config.get("tool_name", "")}')
                code.append("")
        
        return "\n".join(code)


# Convenience functions
def create_workflow(name: str = "My Workflow") -> VisualWorkflow:
    """Quick workflow creation."""
    return VisualWorkflow(name)


def build_agent_workflow(tools: List[str]) -> VisualWorkflow:
    """Build a workflow for an agent with tools."""
    workflow = VisualWorkflow("Agent Workflow")
    
    # Input node
    input_node = workflow.add_node(
        NodeType.INPUT,
        "User Input",
        {"x": 50, "y": 200},
        config={"placeholder": "Enter task..."},
    )
    
    # LLM node
    llm_node = workflow.add_node(
        NodeType.LLM,
        "Agent",
        {"x": 250, "y": 200},
        config={"model": "gpt-4", "system_prompt": "You are an AI agent."},
    )
    
    # Connect
    workflow.connect(input_node.id, "out", llm_node.id, "prompt")
    
    # Add tool nodes
    y_pos = 100
    for tool in tools:
        tool_node = workflow.add_node(
            NodeType.TOOL,
            tool,
            {"x": 450, "y": y_pos},
            config={"tool_name": tool},
        )
        workflow.connect(llm_node.id, "response", tool_node.id, "input")
        y_pos += 100
    
    # Output node
    output_node = workflow.add_node(
        NodeType.OUTPUT,
        "Result",
        {"x": 650, "y": 200},
    )
    
    return workflow
