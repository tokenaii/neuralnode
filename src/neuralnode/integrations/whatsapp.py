"""WhatsApp Integration for NeuralNode (Twilio only)."""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from ..core import BaseAgent
from ..utils.logger import get_logger

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False


@dataclass
class WhatsAppSession:
    """User session in WhatsApp."""
    phone_number: str
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


class TwilioWhatsAppBot:
    """WhatsApp bot via Twilio API."""

    def __init__(self, account_sid: str, auth_token: str, from_number: str, agent: BaseAgent):
        if not TWILIO_AVAILABLE:
            raise ImportError("twilio required. Install: pip install twilio")
        self.account_sid = account_sid
        self.auth_token = auth_token
        self.from_number = from_number
        self.agent = agent
        self.logger = get_logger()
        self.sessions: Dict[str, WhatsAppSession] = {}
        self.client = TwilioClient(account_sid, auth_token)

    def handle_message(self, from_number: str, body: str) -> str:
        if from_number not in self.sessions:
            self.sessions[from_number] = WhatsAppSession(phone_number=from_number)
        session = self.sessions[from_number]
        session.last_activity = datetime.now()
        context = self._build_context(session)
        task = f"Previous context:\n{context}\n\nUser: {body}" if context else body
        try:
            response = self.agent.run(task)
            session.chat_history.append({"role": "user", "content": body})
            session.chat_history.append({"role": "assistant", "content": response})
            self.send_message(from_number, response)
            return response
        except Exception as e:
            self.logger.error(f"Error handling WhatsApp message: {e}")
            error_msg = f"Error: {str(e)[:500]}"
            self.send_message(from_number, error_msg)
            return error_msg

    def send_message(self, to_number: str, body: str):
        max_length = 1600
        if len(body) > max_length:
            chunks = [body[i:i+max_length] for i in range(0, len(body), max_length)]
            for i, chunk in enumerate(chunks, 1):
                self.client.messages.create(
                    from_=f"whatsapp:{self.from_number}",
                    body=f"({i}/{len(chunks)}) {chunk}",
                    to=f"whatsapp:{to_number}"
                )
        else:
            self.client.messages.create(
                from_=f"whatsapp:{self.from_number}",
                body=body,
                to=f"whatsapp:{to_number}"
            )

    def _build_context(self, session: WhatsAppSession) -> str:
        if not session.chat_history:
            return ""
        recent = session.chat_history[-10:]
        return "\n".join(
            f"{msg['role'].capitalize()}: {msg['content'][:150]}" for msg in recent
        )

    def clear_session(self, phone_number: str):
        if phone_number in self.sessions:
            self.sessions[phone_number].chat_history.clear()


def create_twilio_whatsapp_bot(
    account_sid: str,
    auth_token: str,
    from_number: str,
    agent: BaseAgent
) -> TwilioWhatsAppBot:
    return TwilioWhatsAppBot(account_sid=account_sid, auth_token=auth_token, from_number=from_number, agent=agent)


__all__ = ["WhatsAppSession", "TwilioWhatsAppBot", "create_twilio_whatsapp_bot", "TWILIO_AVAILABLE"]
