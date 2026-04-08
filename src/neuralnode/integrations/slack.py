"""Slack Bot Integration for NeuralNode.

Phase 4: Ecosystem - Connect your AI agent to Slack.
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from ..core import BaseAgent
from ..utils.logger import get_logger


# Try to import slack_bolt
try:
    from slack_bolt import App
    from slack_sdk import WebClient
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False


@dataclass
class SlackSession:
    """User session in Slack."""
    user_id: str
    channel_id: str
    team_id: Optional[str] = None
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class SlackBot:
    """Slack bot integration for NeuralNode agents."""
    
    def __init__(
        self,
        token: str,
        signing_secret: str,
        agent: BaseAgent,
        app_token: Optional[str] = None
    ):
        if not SLACK_AVAILABLE:
            raise ImportError(
                "slack-bolt required. Install: pip install slack-bolt"
            )
        
        self.token = token
        self.signing_secret = signing_secret
        self.app_token = app_token
        self.agent = agent
        self.logger = get_logger()
        self.sessions: Dict[str, SlackSession] = {}
        
        # Create Slack app
        if app_token:
            # Socket mode
            self.app = App(token=token, signing_secret=signing_secret)
        else:
            # HTTP mode
            self.app = App(token=token, signing_secret=signing_secret)
        
        self._setup_events()
    
    def _setup_events(self):
        """Setup Slack event handlers."""
        
        @self.app.event("message")
        def handle_message(event, say):
            """Handle incoming messages."""
            user_id = event.get("user")
            text = event.get("text", "")
            channel_id = event.get("channel")
            
            # Skip bot messages
            if event.get("bot_id") or not user_id:
                return
            
            # Get or create session
            session_key = f"{user_id}:{channel_id}"
            if session_key not in self.sessions:
                self.sessions[session_key] = SlackSession(
                    user_id=user_id,
                    channel_id=channel_id
                )
            
            session = self.sessions[session_key]
            
            # Build context
            context = self._build_context(session)
            
            # Get response
            if context:
                task = f"Previous context:\n{context}\n\nUser: {text}"
            else:
                task = text
            
            try:
                response = self.agent.run(task)
                
                # Save to history
                session.chat_history.append({
                    "role": "user",
                    "content": text
                })
                session.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Send response (respect Slack's limits)
                self._send_response(say, response)
            
            except Exception as e:
                self.logger.error(f"Error handling Slack message: {e}")
                say(f"❌ Error: {str(e)[:1000]}")
        
        @self.app.command("/askai")
        def handle_command(ack, say, command):
            """Handle slash command."""
            ack()
            
            question = command.get("text", "")
            if not question:
                say("Please provide a question: `/askai What is Python?`")
                return
            
            try:
                response = self.agent.run(question)
                say(response[:3000])  # Slack block limit
            except Exception as e:
                say(f"Error: {str(e)[:1000]}")
        
        @self.app.command("/clearai")
        def handle_clear(ack, say, command):
            """Clear conversation history."""
            ack()
            
            user_id = command.get("user_id")
            channel_id = command.get("channel_id")
            session_key = f"{user_id}:{channel_id}"
            
            if session_key in self.sessions:
                self.sessions[session_key].chat_history.clear()
            
            say("🧹 Conversation history cleared!")
    
    def _build_context(self, session: SlackSession) -> str:
        """Build context from chat history."""
        if not session.chat_history:
            return ""
        
        recent = session.chat_history[-20:]
        lines = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:200]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def _send_response(self, say_func, text: str):
        """Send response respecting Slack limits."""
        max_length = 3000
        
        if len(text) <= max_length:
            say_func(text)
        else:
            # Split into chunks
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
            for i, chunk in enumerate(chunks, 1):
                say_func(f"({i}/{len(chunks)}) {chunk}")
    
    def start(self, port: int = 3000):
        """Start the Slack bot."""
        self.logger.info("Starting Slack bot...")
        
        if self.app_token:
            # Socket mode
            from slack_bolt.adapter.socket_mode import SocketModeHandler
            handler = SocketModeHandler(self.app, self.app_token)
            handler.start()
        else:
            # HTTP mode (built-in Bolt web server)
            self.logger.info(f"Starting Slack HTTP mode on port {port}")
            self.app.start(port=port)


def create_slack_bot(
    token: str,
    signing_secret: str,
    agent: BaseAgent,
    app_token: Optional[str] = None
) -> SlackBot:
    """Quick function to create Slack bot."""
    return SlackBot(
        token=token,
        signing_secret=signing_secret,
        agent=agent,
        app_token=app_token
    )
