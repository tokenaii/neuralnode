"""WhatsApp Integration for NeuralNode.

Phase 4: Ecosystem - Connect your AI agent to WhatsApp.
"""

from __future__ import annotations

import os
import json
import logging
import time
import requests
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

from ..core import BaseAgent
from ..utils.logger import get_logger

# Try to import twilio for TwilioWhatsAppBot backward compatibility
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


# ==================== Twilio WhatsApp Bot (Legacy) ====================

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
        
        self.client = TwilioClient(account_sid, auth_token)
    
    def handle_message(self, from_number: str, body: str) -> str:
        if from_number not in self.sessions:
            self.sessions[from_number] = WhatsAppSession(
                phone_number=from_number
            )
        
        session = self.sessions[from_number]
        session.last_activity = datetime.now()
        
        context = self._build_context(session)
        
        if context:
            task = f"Previous context:\n{context}\n\nUser: {body}"
        else:
            task = body
        
        try:
            response = self.agent.run(task)
            
            session.chat_history.append({
                "role": "user",
                "content": body
            })
            session.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
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
        lines = []
        for msg in recent:
            role = msg["role"].capitalize()
            content = msg["content"][:150]
            lines.append(f"{role}: {content}")
        
        return "\n".join(lines)
    
    def clear_session(self, phone_number: str):
        if phone_number in self.sessions:
            self.sessions[phone_number].chat_history.clear()


# ==================== Evolution API WhatsApp Bot ====================

class WebhookHTTPHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP request handler for incoming Evolution API webhooks."""
    
    def log_message(self, format, *args):
        # Suppress logging to keep terminal output clean
        pass

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode('utf-8'))
            event = payload.get("event")
            
            # evolution-api sends messages via messages.upsert event
            if event in ["messages.upsert", "MESSAGES_UPSERT"]:
                data = payload.get("data", {})
                message = data.get("message", {})
                conversation = message.get("conversation") or message.get("extendedTextMessage", {}).get("text", "")
                
                key = data.get("key", {})
                from_me = key.get("fromMe", False)
                remote_jid = key.get("remoteJid")  # e.g., "1234567890@c.us"
                
                if not from_me and conversation and self.server.bot.agent:
                    instance = payload.get("instance")
                    # Route message to agent
                    response = self.server.bot.agent.run(conversation)
                    # Respond back to the user
                    self.server.bot.send_text(
                        instance_name=instance,
                        number=remote_jid,
                        text=response
                    )
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            
        except Exception as e:
            logger = get_logger()
            logger.error(f"Error handling webhook payload: {e}")
            self.send_response(500)
            self.end_headers()


class EvolutionWhatsAppBot:
    """WhatsApp integration powered by rebranded Evolution API."""
    
    def __init__(
        self,
        api_url: str = "http://localhost:8080",
        global_apikey: str = "my-secret-key",
        agent: Optional[BaseAgent] = None
    ):
        self.api_url = api_url.rstrip("/")
        self.global_apikey = global_apikey
        self.agent = agent
        self.logger = get_logger()
        self.headers = {
            "apikey": self.global_apikey,
            "Content-Type": "application/json"
        }
        self.gateway_process: Optional[subprocess.Popen] = None
        self.webhook_server: Optional[HTTPServer] = None
        self.webhook_thread: Optional[threading.Thread] = None
        
    def start_gateway(self, port: int = 8080, wait_seconds: int = 15):
        """Start the rebranded local Node.js Evolution API gateway."""
        gateway_dir = Path(__file__).parent / "whatsapp" / "evolution-api"
        if not gateway_dir.exists():
            raise FileNotFoundError(
                f"Evolution API directory not found at {gateway_dir}. "
                "Ensure it has been cloned and rebranded."
            )
            
        # Write local config .env if missing
        env_file = gateway_dir / ".env"
        if not env_file.exists():
            env_config = f"""PORT={port}
AUTHENTICATION_TYPE=apikey
AUTHENTICATION_API_KEY={self.global_apikey}
LOG_LEVEL=ERROR
DATABASE_ENABLED=false
"""
            with open(env_file, "w", encoding="utf-8") as f:
                f.write(env_config)
                
        self.logger.info("Installing gateway dependencies (npm install)...")
        # Run npm install
        subprocess.run(
            "npm install",
            shell=True,
            cwd=str(gateway_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        self.logger.info("Starting NeuralNode WhatsApp Gateway...")
        # Start server
        self.gateway_process = subprocess.Popen(
            "npm run dev" if (gateway_dir / "src").exists() else "npm start",
            shell=True,
            cwd=str(gateway_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for initialization
        time.sleep(wait_seconds)
        self.api_url = f"http://localhost:{port}"
        self.logger.info(f"Gateway is running at {self.api_url}")
        
    def stop_gateway(self):
        """Stop the local gateway process."""
        if self.gateway_process:
            self.gateway_process.terminate()
            self.gateway_process = None
            self.logger.info("Gateway process stopped.")

    def create_instance(self, instance_name: str, token: Optional[str] = None) -> Dict[str, Any]:
        """Create a new WhatsApp instance in the gateway."""
        url = f"{self.api_url}/instance/create"
        payload = {
            "instanceName": instance_name,
            "token": token or self.global_apikey,
            "qrcode": True,
            "integration": "WHATSAPP-BAILEYS"
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()
        
    def connect_qr(self, instance_name: str) -> str:
        """Create instance and generate connection QR code saved to a local image."""
        try:
            self.create_instance(instance_name)
        except Exception:
            # Might already exist
            pass
            
        url = f"{self.api_url}/instance/connect/{instance_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        base64_data = data.get("base64") or data.get("qrcode", {}).get("base64", "")
        if not base64_data:
            raise ValueError("No QR code base64 data returned from gateway.")
            
        # Strip header if present
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
            
        import base64
        image_data = base64.b64decode(base64_data)
        
        qr_file = Path("whatsapp_qr.png").resolve()
        with open(qr_file, "wb") as f:
            f.write(image_data)
            
        self.logger.info(f"QR code successfully saved to: {qr_file}")
        return str(qr_file)
        
    def connect_phone(self, instance_name: str, phone_number: str) -> str:
        """Create instance and request a pairing code to connect by typing it on the phone."""
        try:
            self.create_instance(instance_name)
        except Exception:
            pass
            
        # Clean phone number (digits only)
        phone_clean = "".join(filter(str.isdigit, phone_number))
        url = f"{self.api_url}/instance/connect/phoneCode/{instance_name}?number={phone_clean}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        code = data.get("code")
        if not code:
            raise ValueError("No pairing code returned from gateway.")
            
        self.logger.info(f"Enter this pairing code on your WhatsApp app: {code}")
        return code
        
    def connection_state(self, instance_name: str) -> str:
        """Get the current connection state of an instance (e.g., 'open', 'connecting')."""
        url = f"{self.api_url}/instance/connectionState/{instance_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json().get("instance", {}).get("state", "disconnected")

    def set_webhook(self, instance_name: str, webhook_url: str):
        """Configure webhook for messages on the specified instance."""
        url = f"{self.api_url}/webhook/set/{instance_name}"
        payload = {
            "webhook": {
                "enabled": True,
                "url": webhook_url,
                "byEvents": False,
                "base64": False,
                "events": [
                    "MESSAGES_UPSERT",
                    "CONNECTION_UPDATE"
                ]
            }
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        
    def send_text(self, instance_name: str, number: str, text: str) -> Dict[str, Any]:
        """Send a text message to a phone number."""
        url = f"{self.api_url}/message/sendText/{instance_name}"
        
        # Clean number and append suffix if missing
        number_clean = "".join(filter(str.isdigit, number))
        if not number_clean.endswith("@c.us"):
            number_clean = f"{number_clean}@c.us"
            
        payload = {
            "number": number_clean,
            "text": text,
            "delay": 1000,
            "linkPreview": True
        }
        
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def start_webhook_listener(self, port: int = 5000):
        """Start a zero-dependency local webhook listener to handle incoming messages."""
        self.webhook_server = HTTPServer(('0.0.0.0', port), WebhookHTTPHandler)
        self.webhook_server.bot = self
        
        self.webhook_thread = threading.Thread(
            target=self.webhook_server.serve_forever,
            daemon=True
        )
        self.webhook_thread.start()
        self.logger.info(f"Webhook listener started on port {port}.")

    def stop_webhook_listener(self):
        """Stop the webhook listener server."""
        if self.webhook_server:
            self.webhook_server.shutdown()
            self.webhook_server.server_close()
            self.webhook_server = None
            self.logger.info("Webhook listener stopped.")


# ==================== Backward Compatibility Helpers ====================

def create_twilio_whatsapp_bot(
    account_sid: str,
    auth_token: str,
    from_number: str,
    agent: BaseAgent
) -> TwilioWhatsAppBot:
    return TwilioWhatsAppBot(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        agent=agent
    )
