"""Email tool with OAuth2 support for secure authentication."""

import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)


def safe_decode(data):
    """Safely decode data whether it's bytes or string."""
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    return str(data) if data else ""


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
                                    payload = part.get_payload(decode=True)
                                    body = safe_decode(payload)
                                elif content_type == "text/html":
                                    payload = part.get_payload(decode=True)
                                    html_body = safe_decode(payload)
                        else:
                            payload = msg.get_payload(decode=True)
                            body = safe_decode(payload)
                        
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
    """Gmail client using Gmail API with OAuth2 (modern, secure)."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_file: str = "gmail_token.json",
    ):
        """
        Initialize Gmail client with OAuth2.
        
        Args:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            token_file: File to store tokens
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_file = token_file
        self._creds = None
        self._service = None
        
        # Initialize base with empty values (we use API instead of SMTP/IMAP)
        super().__init__(
            smtp_server="",
            smtp_port=0,
            imap_server="",
            imap_port=0,
        )
        
        # Authenticate and build service
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with Gmail API using OAuth2."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            # Gmail API scopes
            SCOPES = [
                'https://www.googleapis.com/auth/gmail.send',
                'https://www.googleapis.com/auth/gmail.readonly',
            ]
            
            # Load existing token
            if os.path.exists(self.token_file):
                self._creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            
            # If no valid credentials, run OAuth2 flow
            if not self._creds or not self._creds.valid:
                if self._creds and self._creds.expired and self._creds.refresh_token:
                    logger.info("Refreshing Gmail token...")
                    self._creds.refresh(Request())
                else:
                    logger.info("Starting Gmail OAuth2 flow...")
                    flow = InstalledAppFlow.from_client_config(
                        {
                            "installed": {
                                "client_id": self.client_id,
                                "client_secret": self.client_secret,
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                            }
                        },
                        SCOPES
                    )
                    self._creds = flow.run_local_server(port=0)
                
                # Save token
                with open(self.token_file, 'w') as token:
                    token.write(self._creds.to_json())
                logger.info(f"Gmail token saved to {self.token_file}")
            
            # Build Gmail API service
            self._service = build('gmail', 'v1', credentials=self._creds)
            
            # Get user email
            profile = self._service.users().getProfile(userId='me').execute()
            self.username = profile.get('emailAddress', 'me')
            
            logger.info(f"Gmail API authenticated as {self.username}")
            return True
            
        except ImportError:
            logger.error(
                "Google API libraries not installed. "
                "Install: pip install google-auth google-auth-oauthlib google-api-python-client"
            )
            return False
        except Exception as e:
            logger.error(f"Gmail authentication error: {e}", exc_info=True)
            return False
    
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
        """Send email using Gmail API."""
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            import base64
            
            if not self._service:
                logger.error("Gmail service not authenticated")
                return False
            
            # Create message
            message = MIMEMultipart() if attachments else MIMEText(html if html else body, 'html' if html else 'plain')
            
            if attachments:
                message['Subject'] = subject
                message.attach(MIMEText(html if html else body, 'html' if html else 'plain'))
                # Attachments would be added here
            else:
                message['Subject'] = subject
            
            # Recipients
            if isinstance(to, str):
                to = [to]
            message['To'] = ', '.join(to)
            message['From'] = self.username
            
            if cc:
                message['Cc'] = ', '.join(cc)
            
            # Encode message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            # Send via Gmail API
            result = self._service.users().messages().send(
                userId='me',
                body={'raw': encoded_message}
            ).execute()
            
            logger.info(f"Email sent via Gmail API: {result.get('id')}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending Gmail: {e}", exc_info=True)
            return False
    
    def receive_emails(
        self,
        folder: str = "inbox",
        limit: int = 10,
        unread_only: bool = False,
    ) -> List[EmailMessage]:
        """Receive emails using Gmail API."""
        try:
            if not self._service:
                logger.error("Gmail service not authenticated")
                return []
            
            # Build query
            query = ""
            if unread_only:
                query = "is:unread"
            
            # Search messages
            results = self._service.users().messages().list(
                userId='me',
                q=query,
                maxResults=limit
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for msg in messages:
                # Get full message details
                message = self._service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['Subject', 'From', 'Date']
                ).execute()
                
                headers = {h['name']: h['value'] for h in message.get('payload', {}).get('headers', [])}
                
                # Get message body snippet
                snippet = message.get('snippet', '')
                
                # Parse date
                date_str = headers.get('Date', '')
                try:
                    from email.utils import parsedate_to_datetime
                    timestamp = parsedate_to_datetime(date_str)
                except:
                    timestamp = datetime.now()
                
                emails.append(EmailMessage(
                    subject=headers.get('Subject', 'No subject'),
                    body=snippet,
                    from_addr=headers.get('From', ''),
                    to_addr=[self.username],
                    timestamp=timestamp,
                    message_id=msg['id'],
                    is_read='UNREAD' not in message.get('labelIds', [])
                ))
            
            logger.info(f"Retrieved {len(emails)} emails from Gmail")
            return emails
            
        except Exception as e:
            logger.error(f"Error receiving Gmail: {e}", exc_info=True)
            return []


class OutlookClient(EmailClient):
    """Outlook client using Microsoft Graph API (modern, secure)."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str = "common",
        token_file: str = "outlook_token.json",
    ):
        """
        Initialize Outlook client with MS Graph API.
        
        Args:
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID
            token_file: File to store tokens
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token_file = token_file
        self._access_token = None
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
        self._msal_app = None
        
        # Initialize base with dummy values
        super().__init__(
            smtp_server="",
            smtp_port=0,
            imap_server="",
            imap_port=0,
        )
        
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with MS Graph API."""
        try:
            import msal
            
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            
            self._msal_app = msal.PublicClientApplication(
                self.client_id,
                authority=authority,
            )
            
            # Try silent auth first
            accounts = self._msal_app.get_accounts()
            if accounts:
                result = self._msal_app.acquire_token_silent(
                    ["Mail.ReadWrite", "Mail.Send"],
                    account=accounts[0]
                )
                if result and 'access_token' in result:
                    self._access_token = result['access_token']
                    self.username = accounts[0].get('username', 'me')
                    logger.info("Outlook authentication successful (silent)")
                    return True
            
            # Interactive auth
            logger.info("Starting Outlook OAuth2 flow...")
            flow = self._msal_app.initiate_device_flow(scopes=["Mail.ReadWrite", "Mail.Send"])
            
            if 'user_code' not in flow:
                logger.error("Failed to create device flow")
                return False
            
            print(f"""
To sign in, use a web browser to open the page https://microsoft.com/devicelogin 
and enter the code {flow['user_code']} to authenticate.
""")
            
            result = self._msal_app.acquire_token_by_device_flow(flow)
            
            if 'access_token' in result:
                self._access_token = result['access_token']
                self.username = result.get('account', {}).get('username', 'me')
                logger.info("Outlook authentication successful")
                return True
            
            logger.error("Outlook authentication failed")
            return False
            
        except ImportError:
            logger.error("MSAL not installed. Run: pip install msal")
            return False
        except Exception as e:
            logger.error(f"Outlook authentication error: {e}", exc_info=True)
            return False
    
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
        """Send email using Microsoft Graph API."""
        try:
            import requests
            
            if not self._access_token:
                logger.error("Not authenticated with Outlook")
                return False
            
            # Prepare recipients
            recipients = []
            if isinstance(to, str):
                to = [to]
            for email in to:
                recipients.append({"emailAddress": {"address": email}})
            
            # Build message
            message = {
                "subject": subject,
                "body": {
                    "contentType": "HTML" if html else "Text",
                    "content": html if html else body
                },
                "toRecipients": recipients
            }
            
            if cc:
                message["ccRecipients"] = [{"emailAddress": {"address": e}} for e in cc]
            
            # Send
            headers = {
                'Authorization': f'Bearer {self._access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.graph_endpoint}/me/sendMail",
                headers=headers,
                json={"message": message}
            )
            
            if response.status_code in [200, 202]:
                logger.info(f"Email sent successfully via Outlook to {to}")
                return True
            elif response.status_code == 401:
                # Token expired
                if self._authenticate():
                    return self.send_email(to, subject, body, html, cc, bcc, attachments)
                return False
            else:
                logger.error(f"Send email failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Outlook email: {e}", exc_info=True)
            return False
    
    def receive_emails(
        self,
        folder: str = "inbox",
        limit: int = 10,
        unread_only: bool = False,
    ) -> List[EmailMessage]:
        """Receive emails using Microsoft Graph API."""
        try:
            import requests
            
            if not self._access_token:
                logger.error("Not authenticated with Outlook")
                return []
            
            headers = {
                'Authorization': f'Bearer {self._access_token}',
                'Content-Type': 'application/json'
            }
            
            # Build query
            params = {
                "$top": limit,
                "$select": "subject,bodyPreview,receivedDateTime,sender,id,hasAttachments"
            }
            
            if unread_only:
                params["$filter"] = "isRead eq false"
            
            response = requests.get(
                f"{self.graph_endpoint}/me/messages",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                emails = []
                
                for msg in data.get('value', []):
                    sender = msg.get('sender', {}).get('emailAddress', {}).get('address', '')
                    received = msg.get('receivedDateTime', '')
                    
                    try:
                        received_dt = datetime.fromisoformat(received.replace('Z', '+00:00'))
                    except:
                        received_dt = datetime.now()
                    
                    emails.append(EmailMessage(
                        subject=msg.get('subject', 'No subject'),
                        body=msg.get('bodyPreview', ''),
                        from_addr=sender,
                        to_addr=[self.username],
                        timestamp=received_dt,
                        message_id=msg.get('id'),
                        is_read=not msg.get('isRead', True)
                    ))
                
                logger.info(f"Retrieved {len(emails)} emails from Outlook")
                return emails
                
            elif response.status_code == 401:
                # Token expired
                if self._authenticate():
                    return self.receive_emails(folder, limit, unread_only)
                return []
            else:
                logger.error(f"Receive emails failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error receiving Outlook emails: {e}", exc_info=True)
            return []


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


class OAuth2GmailClient(EmailClient):
    """Gmail client with OAuth2 authentication (modern, secure)."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: Optional[str] = None,
        token_file: str = "gmail_token.json",
    ):
        """
        Initialize Gmail client with OAuth2.
        
        Args:
            client_id: Google OAuth2 client ID
            client_secret: Google OAuth2 client secret
            refresh_token: Optional existing refresh token
            token_file: File to store tokens
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.token_file = token_file
        self._access_token = None
        self._token_expiry = None
        
        # Initialize with dummy values, we'll use OAuth2 for auth
        super().__init__(
            smtp_server="smtp.gmail.com",
            smtp_port=587,
            imap_server="imap.gmail.com",
            imap_port=993,
        )
        
        # Load or obtain tokens
        self._load_or_refresh_token()
    
    def _load_or_refresh_token(self) -> bool:
        """Load existing token or get new one via OAuth2 flow."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            
            # Gmail scopes needed
            SCOPES = [
                'https://www.googleapis.com/auth/gmail.send',
                'https://www.googleapis.com/auth/gmail.readonly',
            ]
            
            creds = None
            
            # Load existing token
            if os.path.exists(self.token_file):
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            
            # If no valid credentials, run OAuth2 flow
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    logger.info("Refreshing Gmail token...")
                    creds.refresh(Request())
                else:
                    logger.info("Starting Gmail OAuth2 flow...")
                    flow = InstalledAppFlow.from_client_config(
                        {
                            "installed": {
                                "client_id": self.client_id,
                                "client_secret": self.client_secret,
                                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                                "token_uri": "https://oauth2.googleapis.com/token",
                            }
                        },
                        SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save token for future runs
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
                
                logger.info(f"Gmail token saved to {self.token_file}")
            
            self._access_token = creds.token
            self._token_expiry = creds.expiry
            
            # Get user email from token info
            if creds and creds.valid:
                # Use tokeninfo endpoint to get email
                import requests
                response = requests.get(
                    f"https://oauth2.googleapis.com/tokeninfo?access_token={creds.token}"
                )
                if response.status_code == 200:
                    info = response.json()
                    self.username = info.get('email', 'me')
                else:
                    self.username = 'me'
            
            return True
            
        except ImportError:
            logger.error(
                "Google auth libraries not installed. "
                "Install: pip install google-auth google-auth-oauthlib google-auth-httplib2"
            )
            return False
        except Exception as e:
            logger.error(f"Gmail OAuth2 error: {e}", exc_info=True)
            return False
    
    def _get_smtp_connection(self) -> Optional[smtplib.SMTP]:
        """Get authenticated SMTP connection using OAuth2."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            # Refresh token if needed
            if self._token_expiry and datetime.now() > self._token_expiry:
                self._load_or_refresh_token()
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            # Authenticate with OAuth2
            auth_string = f"user={self.username}\x01auth=Bearer {self._access_token}\x01\x01"
            server.docmd("AUTH", f"XOAUTH2 {auth_string}")
            
            return server
            
        except Exception as e:
            logger.error(f"SMTP OAuth2 connection failed: {e}")
            return None
    
    def _get_imap_connection(self) -> Optional[imaplib.IMAP4_SSL]:
        """Get authenticated IMAP connection using OAuth2."""
        try:
            import imaplib
            
            # Refresh token if needed
            if self._token_expiry and datetime.now() > self._token_expiry:
                self._load_or_refresh_token()
            
            mail = imaplib.IMAP4_SSL(self.imap_server)
            
            # Authenticate with OAuth2
            auth_string = f"user={self.username}\x01auth=Bearer {self._access_token}\x01\x01"
            mail.authenticate("XOAUTH2", lambda x: auth_string)
            
            return mail
            
        except Exception as e:
            logger.error(f"IMAP OAuth2 connection failed: {e}")
            return None


class MicrosoftGraphEmailClient(EmailClient):
    """Outlook/Office 365 email client using Microsoft Graph API (modern, secure)."""
    
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        tenant_id: str = "common",
        token_file: str = "outlook_token.json",
    ):
        """
        Initialize Outlook client with MS Graph API.
        
        Args:
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID (use "common" for personal accounts)
            token_file: File to store tokens
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.token_file = token_file
        self._access_token = None
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
        self._msal_app = None
        
        # Initialize parent with dummy values (we use Graph API instead)
        super().__init__(
            smtp_server="smtp.office365.com",
            smtp_port=587,
            imap_server="outlook.office365.com",
            imap_port=993,
        )
        
        self._authenticate()
    
    def _authenticate(self) -> bool:
        """Authenticate with MS Graph API using MSAL."""
        try:
            import msal
            import json
            
            authority = f"https://login.microsoftonline.com/{self.tenant_id}"
            
            self._msal_app = msal.PublicClientApplication(
                self.client_id,
                authority=authority,
            )
            
            # Load existing accounts
            accounts = self._msal_app.get_accounts()
            
            if accounts:
                # Try silent authentication
                result = self._msal_app.acquire_token_silent(
                    ["Mail.ReadWrite", "Mail.Send"],
                    account=accounts[0]
                )
                if result and 'access_token' in result:
                    self._access_token = result['access_token']
                    self.username = accounts[0].get('username', 'me')
                    logger.info("Outlook authentication successful (silent)")
                    return True
            
            # Interactive authentication
            logger.info("Starting Outlook OAuth2 flow...")
            flow = self._msal_app.initiate_device_flow(
                scopes=["Mail.ReadWrite", "Mail.Send"]
            )
            
            if 'user_code' not in flow:
                logger.error("Failed to create device flow")
                return False
            
            print(f"""
To sign in, use a web browser to open the page https://microsoft.com/devicelogin 
and enter the code {flow['user_code']} to authenticate.
""")
            
            result = self._msal_app.acquire_token_by_device_flow(
                flow,
                expires_at=flow.get('expires_at')
            )
            
            if 'access_token' in result:
                self._access_token = result['access_token']
                self.username = result.get('account', {}).get('username', 'me')
                logger.info("Outlook authentication successful")
                return True
            
            logger.error("Outlook authentication failed")
            return False
            
        except ImportError:
            logger.error("MSAL not installed. Run: pip install msal")
            return False
        except Exception as e:
            logger.error(f"Outlook authentication error: {e}", exc_info=True)
            return False
    
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
        """Send email using Microsoft Graph API."""
        try:
            import requests
            
            # Prepare recipients
            recipients = []
            if isinstance(to, str):
                to = [to]
            for email in to:
                recipients.append({
                    "emailAddress": {"address": email}
                })
            
            # Build message
            message = {
                "subject": subject,
                "body": {
                    "contentType": "HTML" if html else "Text",
                    "content": html if html else body
                },
                "toRecipients": recipients
            }
            
            if cc:
                message["ccRecipients"] = [
                    {"emailAddress": {"address": e}} for e in cc
                ]
            
            # Send via Graph API
            headers = {
                'Authorization': f'Bearer {self._access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.graph_endpoint}/me/sendMail",
                headers=headers,
                json={"message": message}
            )
            
            if response.status_code in [200, 202]:
                logger.info(f"Email sent successfully to {to}")
                return True
            elif response.status_code == 401:
                # Token expired, re-authenticate and retry
                if self._authenticate():
                    return self.send_email(to, subject, body, html, cc, bcc, attachments)
                return False
            else:
                logger.error(f"Send email failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Outlook email: {e}", exc_info=True)
            return False
    
    def receive_emails(
        self,
        folder: str = "inbox",
        limit: int = 10,
        unread_only: bool = False,
    ) -> List[EmailMessage]:
        """Receive emails using Microsoft Graph API."""
        try:
            import requests
            
            headers = {
                'Authorization': f'Bearer {self._access_token}',
                'Content-Type': 'application/json'
            }
            
            # Build query
            params = {
                "$top": limit,
                "$select": "subject,bodyPreview,receivedDateTime,sender,id,hasAttachments"
            }
            
            if unread_only:
                params["$filter"] = "isRead eq false"
            
            response = requests.get(
                f"{self.graph_endpoint}/me/messages",
                headers=headers,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                emails = []
                
                for msg in data.get('value', []):
                    sender = msg.get('sender', {}).get('emailAddress', {}).get('address', '')
                    received = msg.get('receivedDateTime', '')
                    
                    try:
                        received_dt = datetime.fromisoformat(received.replace('Z', '+00:00'))
                    except:
                        received_dt = datetime.now()
                    
                    emails.append(EmailMessage(
                        subject=msg.get('subject', 'No subject'),
                        body=msg.get('bodyPreview', ''),
                        from_addr=sender,
                        to_addr=[self.username],
                        timestamp=received_dt,
                        message_id=msg.get('id'),
                    ))
                
                logger.info(f"Retrieved {len(emails)} emails from Outlook")
                return emails
                
            elif response.status_code == 401:
                # Token expired
                if self._authenticate():
                    return self.receive_emails(folder, limit, unread_only)
                return []
            else:
                logger.error(f"Receive emails failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error receiving Outlook emails: {e}", exc_info=True)
            return []


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
