"""Central tools registry for NeuralNode."""

from .web import WebSearchTool
from .system.operations import DateTimeTool, SystemOperationsTool
from .advanced import PythonREPLTool, FileReadTool, FileWriteTool

# Aliases for compatibility with main __init__.py
FileReaderTool = FileReadTool
FileWriterTool = FileWriteTool

# ── Simple Calculator Tool ───────────────────────────────────────────────────
from ..core import BaseTool, ToolResult, ParameterSpec
import math


class CalculatorTool(BaseTool):
    """Perform mathematical calculations."""

    name = "calculator"
    description = "Perform mathematical calculations and evaluate expressions."
    parameters = {
        "expression": ParameterSpec(
            type="string",
            description="The math expression to evaluate (e.g., '2 + 2', 'math.sqrt(16)', 'math.pi * 5^2')",
            required=True,
        )
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            # Basic sanitization
            expr = expression.replace("^", "**")
            # Safe evaluation with math module
            allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed_names["abs"] = abs
            allowed_names["round"] = round
            allowed_names["min"] = min
            allowed_names["max"] = max

            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return ToolResult.ok(str(result), value=result)
        except Exception as e:
            return ToolResult.error(f"Calculation failed: {str(e)}")


__all__ = [
    "WebSearchTool",
    "FileReaderTool",
    "FileWriterTool",
    "CalculatorTool",
    "PythonREPLTool",
    "DateTimeTool",
    "SystemOperationsTool",
]
