"""WhatsApp Integration for NeuralNode.

Phase 4: Ecosystem - Connect your AI agent to WhatsApp.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from ..core import BaseAgent
from ..utils.logger import get_logger


# Try to import whatsapp libraries
try:
    # For WhatsApp Business API (via Twilio or direct)
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

try:
    # Alternative: whatsapp-web.py (unofficial, for development only)
    from whatsapp import Client as WhatsAppClient
    WHATSAPP_WEB_AVAILABLE = True
except ImportError:
    WHATSAPP_WEB_AVAILABLE = False


@dataclass
class WhatsAppSession:
    """User session in WhatsApp."""
    phone_number: str
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class TwilioWhatsAppBot:
    """WhatsApp bot via Twilio API."""
    
    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
        agent: BaseAgent
    ):
        if not TWILIO_AVAILABLE:
            raise ImportError(
                "twilio required. Install: pip install twilio"
            )
        
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.agent = agent
        self.logger = get_logger()
        self.sessions: Dict[str, WhatsAppSession] = {}
        
        # Create Twilio client
        self.client = TwilioClient(account_sid, auth_token)
    
    def handle_message(self, from_number: str, body: str) -> str:
        """Handle incoming WhatsApp message.
        
        This is typically called from a webhook endpoint.
        
        Args:
            from_number: Sender's phone number
            body: Message content
        
        Returns:
            Response message to send back
        """
        # Get or create session
        if from_number not in self.sessions:
            self.sessions[from_number] = WhatsAppSession(
                phone_number=from_number
            )
        
        session = self.sessions[from_number]
        session.last_activity = datetime.now()
        
        # Build context
        context = self._build_context(session)
        
        # Get response
        if context:
            task = f"Previous context:\n{context}\n\nUser: {body}"
        else:
            task = body
        
        try:
            response = self.agent.run(task)
            
            # Save to history
            session.chat_history.append({
                "role": "user",
                "content": body
            })
            session.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # Send response
            self.send_message(from_number, response)
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error handling WhatsApp message: {e}")
            error_msg = f"❌ Error: {str(e)[:500]}"
            self.send_message(from_number, error_msg)
            return error_msg
    
    def send_message(self, to_number: str, body: str):
        """Send WhatsApp message.
        
        Args:
            to_number: Recipient phone number (with country code)
            body: Message content
        """
        # Twilio has a 1600 character limit for WhatsApp
        max_length = 1600
        
        if len(body) > max_length:
            # Split into multiple messages
            chunks = [body[i:i+max_length] for i in range(0, len(body), max_length)]
            for i, chunk in enumerate(chunks, 1):
                message = self.client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    body=f"({i}/{len(chunks)}) {chunk}",
                    to=f"whatsapp:{to_number}"
                )
                self.logger.info(f"Sent WhatsApp message {i}/{len(chunks)}: {message.sid}")
        else:
            message = self.client.messages.create(
                from_=f"whatsapp:{self.from_number}",
                body=body,
                to=f"whatsapp:{to_number}"
            )
            self.logger.info(f"Sent WhatsApp message: {message.sid}")
    
    def _build_context(self, session: WhatsAppSession) -> str:
        """Build context from chat history."""
        if not session.chat_history:
            return ""
        
        recent = session.chat_history[-10:]  # Last 5 exchanges
        lines = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:150]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def clear_session(self, phone_number: str):
        """Clear session for a user."""
        if phone_number in self.sessions:
            self.sessions[phone_number].chat_history.clear()
            self.logger.info(f"Cleared session for {phone_number}")


class WhatsAppWebBot:
    """WhatsApp bot using whatsapp-web.py (unofficial).
    
    WARNING: This uses an unofficial library and may break with WhatsApp updates.
    Only for development/testing, not for production use.
    """
    
    def __init__(self, agent: BaseAgent):
        if not WHATSAPP_WEB_AVAILABLE:
            raise ImportError(
                "whatsapp-web.py not available. "
                "Note: This is an unofficial library. "
                "Install at your own risk: pip install whatsapp-web.py"
            )
        
        self.agent = agent
        self.logger = get_logger()
        self.sessions: Dict[str, WhatsAppSession] = {}
        self.client = None
    
    def start(self):
        """Start WhatsApp Web client.
        
        This will show a QR code to scan.
        """
        self.logger.info("Starting WhatsApp Web (unofficial)...")
        self.logger.warning("Using unofficial library - may break with WhatsApp updates")
        
        # Note: This is a placeholder as the actual implementation
        # depends on the specific library version and is subject to change
        # Users should refer to the library's documentation
        raise NotImplementedError(
            "WhatsApp Web integration requires manual setup. "
            "Please refer to whatsapp-web.py documentation."
        )


def create_twilio_whatsapp_bot(
    account_sid: str,
    auth_token: str,
    from_number: str,
    agent: BaseAgent
) -> TwilioWhatsAppBot:
    """Create WhatsApp bot via Twilio.
    
    Args:
        account_sid: Twilio Account SID
        auth_token: Twilio Auth Token
        from_number: Your Twilio WhatsApp number
        agent: NeuralNode agent
    
    Returns:
        Configured WhatsApp bot
    """
    return TwilioWhatsAppBot(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        agent=agent
    )


# Webhook handler example (for Flask/FastAPI)
def create_webhook_handler(bot: TwilioWhatsAppBot):
    """Create webhook handler for incoming messages.
    
    Example for Flask:
        from flask import Flask, request
        app = Flask(__name__)
        
        handler = create_webhook_handler(bot)
        
        @app.route("/whatsapp", methods=['POST'])
        def whatsapp():
            return handler(request)
    
    Example for FastAPI:
        from fastapi import Request
        
        handler = create_webhook_handler(bot)
        
        @app.post("/whatsapp")
        async def whatsapp(request: Request):
            return handler(request)
    """
    def handler(request):
        """Handle incoming webhook."""
        # Get message details from request
        from_number = request.values.get('From', '').replace('whatsapp:', '')
        body = request.values.get('Body', '')
        
        # Process message
        bot.handle_message(from_number, body)
        
        # Return empty response (Twilio doesn't need content)
        return '', 204
    
    return handler
