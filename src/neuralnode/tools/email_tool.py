"""Email tool for sending and receiving emails."""

import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import os


@dataclass
class EmailMessage:
    """Email message structure."""
    subject: str
    body: str
    from_addr: str
    to_addr: List[str]
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    attachments: Optional[List[str]] = None
    html_body: Optional[str] = None
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None


class EmailClient:
    """Email client for sending and receiving emails."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        imap_server: Optional[str] = None,
        imap_port: int = 993,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.imap_server = imap_server
        self.imap_port = imap_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
    
    def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html: Optional[str] = None,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
    ) -> bool:
        """Send an email."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.username
            
            # Handle recipients
            if isinstance(to, str):
                to = [to]
            msg['To'] = ', '.join(to)
            
            if cc:
                msg['Cc'] = ', '.join(cc)
            
            all_recipients = to + (cc or []) + (bcc or [])
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Add HTML if provided
            if html:
                msg.attach(MIMEText(html, 'html'))
            
            # Add attachments
            if attachments:
                for filepath in attachments:
                    if os.path.exists(filepath):
                        part = MIMEBase('application', 'octet-stream')
                        with open(filepath, 'rb') as f:
                            part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename={os.path.basename(filepath)}'
                        )
                        msg.attach(part)
            
            # Connect and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                server.sendmail(self.username, all_recipients, msg.as_string())
            
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def receive_emails(
        self,
        folder: str = "INBOX",
        limit: int = 10,
        unread_only: bool = False,
    ) -> List[EmailMessage]:
        """Receive emails from IMAP server."""
        if not self.imap_server:
            print("IMAP server not configured")
            return []
        
        try:
            # Connect to IMAP
            mail = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
            mail.login(self.username, self.password)
            mail.select(folder)
            
            # Search for emails
            if unread_only:
                status, messages = mail.search(None, 'UNSEEN')
            else:
                status, messages = mail.search(None, 'ALL')
            
            email_ids = messages[0].split()
            email_ids = email_ids[-limit:]  # Get last N emails
            
            emails = []
            for email_id in reversed(email_ids):
                status, msg_data = mail.fetch(email_id, '(RFC822)')
                
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Parse email
                        subject = email.header.decode_header(msg['Subject'])[0][0]
                        if isinstance(subject, bytes):
                            subject = subject.decode()
                        
                        from_addr = msg['From']
                        to_addr = msg['To']
                        date = msg['Date']
                        message_id = msg['Message-ID']
                        
                        # Get body
                        body = ""
                        html_body = None
                        
                        if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                
                                if content_type == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                elif content_type == "text/html":
                                    html_body = part.get_payload(decode=True).decode()
                        else:
                            body = msg.get_payload(decode=True).decode()
                        
                        emails.append(EmailMessage(
                            subject=subject,
                            body=body,
                            from_addr=from_addr,
                            to_addr=[to_addr] if to_addr else [],
                            html_body=html_body,
                            message_id=message_id,
                        ))
            
            mail.close()
            mail.logout()
            
            return emails
        except Exception as e:
            print(f"Error receiving emails: {e}")
            return []
    
    def search_emails(
        self,
        query: str,
        folder: str = "INBOX",
        limit: int = 10,
    ) -> List[EmailMessage]:
        """Search emails."""
        emails = self.receive_emails(folder, limit * 2)
        
        # Filter by query
        results = []
        query_lower = query.lower()
        
        for email in emails:
            if (query_lower in email.subject.lower() or
                query_lower in email.body.lower()):
                results.append(email)
                
                if len(results) >= limit:
                    break
        
        return results
    
    def mark_as_read(self, message_id: str) -> bool:
        """Mark an email as read."""
        # Implementation would use IMAP to mark message as seen
        return True


class GmailClient(EmailClient):
    """Gmail-specific email client."""
    
    def __init__(
        self,
        username: str,
        password: str,  # Use app password for Gmail
    ):
        super().__init__(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            imap_server="imap.gmail.com",
            imap_port=993,
            username=username,
            password=password,
            use_tls=True,
        )


class OutlookClient(EmailClient):
    """Outlook-specific email client."""
    
    def __init__(
        self,
        username: str,
        password: str,
    ):
        super().__init__(
            smtp_server="smtp.office365.com",
            smtp_port=587,
            imap_server="outlook.office365.com",
            imap_port=993,
            username=username,
            password=password,
            use_tls=True,
        )


class EmailTool:
    """Email tool for AI agents."""
    
    def __init__(self, client: Optional[EmailClient] = None):
        self.client = client
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for agent."""
        return {
            "name": "send_email",
            "description": "Send an email to one or more recipients",
            "parameters": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address(es), comma-separated"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content"
                },
            },
            "required": ["to", "subject", "body"]
        }
    
    def execute(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> str:
        """Execute email sending."""
        if not self.client:
            return "Error: Email client not configured"
        
        recipients = [e.strip() for e in to.split(',')]
        
        success = self.client.send_email(
            to=recipients,
            subject=subject,
            body=body,
        )
        
        if success:
            return f"Email sent successfully to {to}"
        else:
            return "Failed to send email"


# Convenience functions
def send_email(
    to: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
) -> bool:
    """Quick email sending."""
    client = EmailClient(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        username=username,
        password=password,
    )
    return client.send_email(to=to, subject=subject, body=body)


def send_gmail(
    to: str,
    subject: str,
    body: str,
    username: str,
    app_password: str,
) -> bool:
    """Quick Gmail sending."""
    client = GmailClient(username=username, password=app_password)
    return client.send_email(to=to, subject=subject, body=body)
