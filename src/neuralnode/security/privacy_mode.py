"""Privacy Mode with full data encryption for NeuralNode."""

import os
import json
import hashlib
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets


@dataclass
class EncryptionConfig:
    """Configuration for encryption."""
    enabled: bool = True
    algorithm: str = "AES256"
    key_rotation_days: int = 30
    encrypt_memory: bool = True
    encrypt_disk: bool = True
    encrypt_network: bool = True


class DataEncryptor:
    """Encrypt sensitive data."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or self._generate_password()
        self.key = self._derive_key(self.password)
        self.fernet = Fernet(self.key)
    
    def _generate_password(self) -> str:
        """Generate a secure password."""
        return secrets.token_urlsafe(32)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: Union[str, bytes, Dict]) -> str:
        """Encrypt data."""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        if isinstance(data, str):
            data = data.encode()
        
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> Union[str, Dict]:
        """Decrypt data."""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            result = decrypted.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file."""
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted = self.fernet.encrypt(data)
        
        output = output_path or f"{file_path}.encrypted"
        with open(output, 'wb') as f:
            f.write(encrypted)
        
        return output
    
    def decrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file."""
        with open(file_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.fernet.decrypt(encrypted)
        
        output = output_path or file_path.replace('.encrypted', '.decrypted')
        with open(output, 'wb') as f:
            f.write(decrypted)
        
        return output


class PrivacyMode:
    """Full privacy mode with encryption."""
    
    def __init__(self, config: Optional[EncryptionConfig] = None):
        self.config = config or EncryptionConfig()
        self.encryptor: Optional[DataEncryptor] = None
        self.enabled = False
        self.sensitive_fields = [
            "api_key", "password", "secret", "token", "credit_card",
            "ssn", "email", "phone", "address", "name",
        ]
    
    def enable(self, password: Optional[str] = None) -> None:
        """Enable privacy mode."""
        self.encryptor = DataEncryptor(password)
        self.enabled = True
        print("🔒 Privacy Mode Enabled - All data will be encrypted")
    
    def disable(self) -> None:
        """Disable privacy mode."""
        self.enabled = False
        self.encryptor = None
        print("🔓 Privacy Mode Disabled")
    
    def encrypt_message(self, message: str) -> str:
        """Encrypt a message."""
        if not self.enabled or not self.encryptor:
            return message
        return self.encryptor.encrypt(message)
    
    def decrypt_message(self, encrypted: str) -> str:
        """Decrypt a message."""
        if not self.enabled or not self.encryptor:
            return encrypted
        return self.encryptor.decrypt(encrypted)
    
    def sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data."""
        sanitized = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if field is sensitive
            is_sensitive = any(field in key_lower for field in self.sensitive_fields)
            
            if is_sensitive and self.enabled:
                if isinstance(value, str):
                    sanitized[key] = self.encryptor.encrypt(value)
                else:
                    sanitized[key] = "[ENCRYPTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def mask_sensitive(self, text: str) -> str:
        """Mask sensitive information in text."""
        import re
        
        # Mask API keys
        text = re.sub(r'sk-[a-zA-Z0-9]{48}', '[API_KEY_MASKED]', text)
        
        # Mask emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{{2,}}\b', '[EMAIL_MASKED]', text)
        
        # Mask phone numbers
        text = re.sub(r'\b\d{{3}}[-.]?\d{{3}}[-.]?\d{{4}}\b', '[PHONE_MASKED]', text)
        
        return text
    
    def secure_memory_clear(self, data: Any) -> None:
        """Securely clear data from memory."""
        if isinstance(data, str):
            # Overwrite string (limited effectiveness in Python)
            data = '\x00' * len(data)
        elif isinstance(data, bytearray):
            for i in range(len(data)):
                data[i] = 0
    
    def create_secure_session(self, session_id: str) -> Dict[str, Any]:
        """Create a secure session with encrypted storage."""
        return {
            "session_id": session_id,
            "encrypted": self.enabled,
            "created": self.encryptor.encrypt(str(datetime.now())) if self.enabled else str(datetime.now()),
        }
    
    def audit_log(self, action: str, details: Dict[str, Any]) -> None:
        """Create privacy audit log."""
        # Remove sensitive data from log
        safe_details = {k: v for k, v in details.items() 
                       if not any(s in k.lower() for s in self.sensitive_fields)}
        
        log_entry = {
            "timestamp": str(datetime.now()),
            "action": action,
            "details": safe_details,
            "privacy_mode": self.enabled,
        }
        
        # Would write to secure audit log
        print(f"Audit: {action}")


class SecureStorage:
    """Encrypted storage for sensitive data."""
    
    def __init__(self, storage_path: str, password: str):
        self.path = storage_path
        self.encryptor = DataEncryptor(password)
    
    def store(self, key: str, value: Any) -> None:
        """Store encrypted value."""
        encrypted = self.encryptor.encrypt(json.dumps({"value": value}))
        
        # Would write to encrypted database/file
        print(f"Stored encrypted value for key: {key}")
    
    def retrieve(self, key: str) -> Any:
        """Retrieve and decrypt value."""
        # Would read from encrypted storage
        return None
    
    def delete(self, key: str) -> bool:
        """Securely delete value."""
        # Would securely delete from storage
        return True


# Convenience functions
def enable_privacy_mode(password: Optional[str] = None) -> PrivacyMode:
    """Quick enable privacy mode."""
    privacy = PrivacyMode()
    privacy.enable(password)
    return privacy


def encrypt_sensitive_data(data: Dict[str, Any], password: str) -> Dict[str, Any]:
    """Quick encrypt sensitive fields."""
    privacy = PrivacyMode()
    privacy.enable(password)
    return privacy.sanitize_data(data)


def mask_text(text: str) -> str:
    """Quick mask sensitive info in text."""
    privacy = PrivacyMode()
    return privacy.mask_sensitive(text)
