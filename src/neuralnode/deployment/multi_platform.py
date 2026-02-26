"""Multi-platform deployment for NeuralNode."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import json


class Platform(Enum):
    """Supported deployment platforms."""
    MOBILE_IOS = "mobile_ios"
    MOBILE_ANDROID = "mobile_android"
    DESKTOP_WINDOWS = "desktop_windows"
    DESKTOP_MAC = "desktop_mac"
    DESKTOP_LINUX = "desktop_linux"
    CLOUD_DOCKER = "cloud_docker"
    CLOUD_KUBERNETES = "cloud_kubernetes"
    WEB = "web"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    platform: Platform
    api_endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    theme: str = "light"
    language: str = "en"
    offline_mode: bool = False
    cache_size_mb: int = 500


class MobileAppBuilder:
    """Build mobile apps for iOS and Android."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def build_ios(self, output_dir: str = "./mobile_ios") -> bool:
        """Build iOS app using React Native."""
        try:
            # Create React Native project
            print("Creating iOS app with React Native...")
            
            # Generate app structure
            self._generate_react_native_app(output_dir, "ios")
            
            # Build native code
            result = subprocess.run(
                ["npx", "react-native", "build-ios", "--release"],
                cwd=output_dir,
                capture_output=True,
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"iOS build failed: {e}")
            return False
    
    def build_android(self, output_dir: str = "./mobile_android") -> bool:
        """Build Android app using React Native."""
        try:
            print("Creating Android app with React Native...")
            
            self._generate_react_native_app(output_dir, "android")
            
            result = subprocess.run(
                ["npx", "react-native", "build-android", "--release"],
                cwd=output_dir,
                capture_output=True,
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Android build failed: {e}")
            return False
    
    def _generate_react_native_app(self, output_dir: str, platform: str) -> None:
        """Generate React Native app code."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # App.js
        app_js = f"""
import React, {{ useState }} from 'react';
import {{ View, Text, TextInput, Button, ScrollView, StyleSheet }} from 'react-native';

export default function App() {{
  const [message, setMessage] = useState('');
  const [response, setResponse] = useState('');
  
  const sendMessage = async () => {{
    try {{
      const res = await fetch('{self.config.api_endpoint}/chat', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ prompt: message }}),
      }});
      const data = await res.json();
      setResponse(data.text);
    }} catch (error) {{
      setResponse('Error: ' + error.message);
    }}
  }};
  
  return (
    <View style={{styles.container}}>
      <Text style={{styles.title}}>NeuralNode Mobile</Text>
      <TextInput
        style={{styles.input}}
        placeholder="Enter your message..."
        value={{message}}
        onChangeText={{setMessage}}
        multiline
      />
      <Button title="Send" onPress={{sendMessage}} />
      <ScrollView style={{styles.responseContainer}}>
        <Text>{{response}}</Text>
      </ScrollView>
    </View>
  );
}}

const styles = StyleSheet.create({{
  container: {{ flex: 1, padding: 20, backgroundColor: '#fff' }},
  title: {{ fontSize: 24, fontWeight: 'bold', marginBottom: 20 }},
  input: {{ borderWidth: 1, borderColor: '#ccc', padding: 10, marginBottom: 10, minHeight: 100 }},
  responseContainer: {{ marginTop: 20, padding: 10, backgroundColor: '#f0f0f0' }},
}});
"""
        
        with open(f"{output_dir}/App.js", "w") as f:
            f.write(app_js)


class DesktopGUIBuilder:
    """Build desktop apps for Windows, Mac, and Linux."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def build_electron(self, output_dir: str = "./desktop") -> bool:
        """Build cross-platform desktop app with Electron."""
        try:
            print("Creating Electron desktop app...")
            
            self._generate_electron_app(output_dir)
            
            # Build for current platform
            result = subprocess.run(
                ["npx", "electron-builder"],
                cwd=output_dir,
                capture_output=True,
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Electron build failed: {e}")
            return False
    
    def build_tauri(self, output_dir: str = "./desktop_tauri") -> bool:
        """Build with Tauri (Rust + Web)."""
        try:
            print("Creating Tauri desktop app...")
            
            self._generate_tauri_app(output_dir)
            
            result = subprocess.run(
                ["cargo", "tauri", "build"],
                cwd=output_dir,
                capture_output=True,
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Tauri build failed: {e}")
            return False
    
    def _generate_electron_app(self, output_dir: str) -> None:
        """Generate Electron app."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # package.json
        package_json = {
            "name": "neuralnode-desktop",
            "version": "1.0.0",
            "main": "main.js",
            "scripts": {
                "start": "electron .",
                "build": "electron-builder",
            },
            "dependencies": {
                "electron": "^28.0.0",
            },
            "devDependencies": {
                "electron-builder": "^24.0.0",
            },
            "build": {
                "appId": "com.neuralnode.desktop",
                "productName": "NeuralNode",
                "directories": {
                    "output": "dist",
                },
                "files": ["main.js", "renderer.js", "index.html", "package.json"],
            },
        }
        
        with open(f"{output_dir}/package.json", "w") as f:
            json.dump(package_json, f, indent=2)
        
        # main.js
        main_js = """
const { app, BrowserWindow } = require('electron');
const path = require('path');

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  });
  
  mainWindow.loadFile('index.html');
}

app.whenReady().then(createWindow);
app.on('window-all-closed', () => app.quit());
"""
        
        with open(f"{output_dir}/main.js", "w") as f:
            f.write(main_js)
        
        # index.html
        html = f"""
<!DOCTYPE html>
<html>
<head>
  <title>NeuralNode Desktop</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
    h1 {{ color: #333; margin-bottom: 20px; }}
    .chat-container {{ height: 500px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 20px; background: #fafafa; }}
    .input-area {{ display: flex; gap: 10px; }}
    textarea {{ flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; resize: vertical; min-height: 80px; }}
    button {{ padding: 10px 20px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background: #0056b3; }}
    .message {{ margin: 10px 0; padding: 10px; border-radius: 4px; }}
    .user {{ background: #e3f2fd; text-align: right; }}
    .ai {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>🧠 NeuralNode Desktop</h1>
    <div class="chat-container" id="chat"></div>
    <div class="input-area">
      <textarea id="input" placeholder="Enter your message..."></textarea>
      <button onclick="sendMessage()">Send</button>
    </div>
  </div>
  
  <script src="renderer.js"></script>
</body>
</html>
"""
        
        with open(f"{output_dir}/index.html", "w") as f:
            f.write(html)
        
        # renderer.js
        renderer_js = f"""
const API_ENDPOINT = '{self.config.api_endpoint}';

async function sendMessage() {{
  const input = document.getElementById('input');
  const chat = document.getElementById('chat');
  const message = input.value.trim();
  
  if (!message) return;
  
  // Add user message
  chat.innerHTML += '<div class="message user">' + escapeHtml(message) + '</div>';
  input.value = '';
  
  try {{
    const response = await fetch(API_ENDPOINT + '/chat', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ prompt: message }}),
    }});
    
    const data = await response.json();
    chat.innerHTML += '<div class="message ai">' + escapeHtml(data.text) + '</div>';
    chat.scrollTop = chat.scrollHeight;
  }} catch (error) {{
    chat.innerHTML += '<div class="message ai" style="color: red;">Error: ' + escapeHtml(error.message) + '</div>';
  }}
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

document.getElementById('input').addEventListener('keypress', (e) => {{
  if (e.key === 'Enter' && !e.shiftKey) {{
    e.preventDefault();
    sendMessage();
  }}
}});
"""
        
        with open(f"{output_dir}/renderer.js", "w") as f:
            f.write(renderer_js)
    
    def _generate_tauri_app(self, output_dir: str) -> None:
        """Generate Tauri app structure."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Would generate Rust + web code
        # Simplified for now
        pass


class CloudDeployer:
    """Deploy to cloud platforms."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_dockerfile(self, output_path: str = "./Dockerfile") -> bool:
        """Generate Dockerfile for deployment."""
        dockerfile = """
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "-m", "neuralnode.server", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        try:
            with open(output_path, "w") as f:
                f.write(dockerfile)
            return True
        except Exception as e:
            print(f"Dockerfile generation failed: {e}")
            return False
    
    def generate_kubernetes_yaml(self, output_path: str = "./k8s-deployment.yaml") -> bool:
        """Generate Kubernetes deployment files."""
        k8s_config = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuralnode
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuralnode
  template:
    metadata:
      labels:
        app: neuralnode
    spec:
      containers:
      - name: neuralnode
        image: neuralnode:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: neuralnode-service
spec:
  selector:
    app: neuralnode
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""
        
        try:
            with open(output_path, "w") as f:
                f.write(k8s_config)
            return True
        except Exception as e:
            print(f"Kubernetes config generation failed: {e}")
            return False
    
    def generate_terraform(self, output_dir: str = "./terraform") -> bool:
        """Generate Terraform for cloud deployment."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # main.tf
        main_tf = """
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_instance" "neuralnode" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.large"
  
  tags = {
    Name = "NeuralNode Server"
  }
  
  user_data = <<-EOF
              #!/bin/bash
              apt-get update
              apt-get install -y docker.io
              docker run -d -p 8000:8000 neuralnode:latest
              EOF
}

resource "aws_security_group" "neuralnode" {
  name_prefix = "neuralnode-"
  
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
"""
        
        with open(f"{output_dir}/main.tf", "w") as f:
            f.write(main_tf)
        
        # variables.tf
        variables_tf = """
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}
"""
        
        with open(f"{output_dir}/variables.tf", "w") as f:
            f.write(variables_tf)
        
        return True


class MultiPlatformDeployer:
    """Unified multi-platform deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.mobile_builder = MobileAppBuilder(config)
        self.desktop_builder = DesktopGUIBuilder(config)
        self.cloud_deployer = CloudDeployer(config)
    
    def deploy(self, platform: Platform, output_dir: str) -> bool:
        """Deploy to specified platform."""
        if platform == Platform.MOBILE_IOS:
            return self.mobile_builder.build_ios(output_dir)
        elif platform == Platform.MOBILE_ANDROID:
            return self.mobile_builder.build_android(output_dir)
        elif platform == Platform.DESKTOP_WINDOWS:
            return self.desktop_builder.build_electron(output_dir)
        elif platform == Platform.DESKTOP_MAC:
            return self.desktop_builder.build_electron(output_dir)
        elif platform == Platform.DESKTOP_LINUX:
            return self.desktop_builder.build_electron(output_dir)
        elif platform == Platform.CLOUD_DOCKER:
            return self.cloud_deployer.generate_dockerfile(f"{output_dir}/Dockerfile")
        elif platform == Platform.CLOUD_KUBERNETES:
            return self.cloud_deployer.generate_kubernetes_yaml(f"{output_dir}/k8s.yaml")
        else:
            print(f"Platform {platform} not yet supported")
            return False
    
    def deploy_all(self, output_base_dir: str = "./deployments") -> Dict[str, bool]:
        """Deploy to all supported platforms."""
        results = {}
        
        platforms = [
            Platform.MOBILE_ANDROID,
            Platform.DESKTOP_WINDOWS,
            Platform.CLOUD_DOCKER,
        ]
        
        for platform in platforms:
            output_dir = f"{output_base_dir}/{platform.value}"
            results[platform.value] = self.deploy(platform, output_dir)
        
        return results


# Convenience functions
def build_mobile_app(platform: str = "android", api_endpoint: str = "http://localhost:8000") -> bool:
    """Quick mobile app builder."""
    config = DeploymentConfig(
        platform=Platform.MOBILE_ANDROID if platform == "android" else Platform.MOBILE_IOS,
        api_endpoint=api_endpoint,
    )
    builder = MobileAppBuilder(config)
    
    if platform == "android":
        return builder.build_android()
    else:
        return builder.build_ios()


def build_desktop_app(api_endpoint: str = "http://localhost:8000") -> bool:
    """Quick desktop app builder."""
    config = DeploymentConfig(
        platform=Platform.DESKTOP_WINDOWS,
        api_endpoint=api_endpoint,
    )
    builder = DesktopGUIBuilder(config)
    return builder.build_electron()


def deploy_to_cloud(platform: str = "docker") -> bool:
    """Quick cloud deployment."""
    config = DeploymentConfig(
        platform=Platform.CLOUD_DOCKER,
    )
    deployer = CloudDeployer(config)
    
    if platform == "docker":
        return deployer.generate_dockerfile()
    elif platform == "kubernetes":
        return deployer.generate_kubernetes_yaml()
    elif platform == "terraform":
        return deployer.generate_terraform()
    return False
