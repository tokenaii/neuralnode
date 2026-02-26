"""Tools module for NeuralNode."""

from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod


class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass
    
    def to_definition(self) -> Dict[str, Any]:
        """Convert tool to function definition format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }


# Other imports
from .builtin import Calculator, WebSearch, CodeRunner, get_default_tools
from .browser_automation import BrowserController, BrowserTools
from .web_search import WebSearch, search, search_with_answer
from .system_control import (
    SystemController,
    FileManager,
    ProcessManager,
    CommandRunner,
    SystemInfo,
)
from .multimodal import (
    SpeechToText,
    TextToSpeech,
    VisionProcessor,
    MultiModalChat,
    transcribe_audio,
    speak_text,
    describe_image,
)
from .file_watcher import FileWatcher, watch_directory
from .email_tool import (
    EmailClient,
    GmailClient,
    OutlookClient,
    EmailTool,
    send_email,
    send_gmail,
)
from .calendar_integration import (
    CalendarEvent,
    GoogleCalendarClient,
    OutlookCalendarClient,
    CalendarTool,
    list_google_calendar_events,
    create_google_calendar_event,
)
from .code_interpreter import (
    CodeInterpreter,
    SandboxedInterpreter,
    JupyterStyleInterpreter,
    CodeResult,
    execute_python,
    execute_safely,
)
from .universal_tools import (
    ThinkingTool,
    VerifiedSearchTool,
    ChainOfThoughtTool,
    think_before_answer,
    search_verified,
    chain_of_thought,
)

__all__ = [
    # Base
    "Tool",
    # Built-in
    "WebSearch",
    "BrowserTools",
    "Calculator",
    "CodeRunner",
    "get_default_tools",
]
