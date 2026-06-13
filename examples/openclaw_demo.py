"""
NeuralNode OpenClaw Integration Demo
====================================

This script demonstrates how to download, setup, configure, start, and control
the OpenClaw gateway from within your NeuralNode framework.
"""

import time
import os
import neuralnode as nn
from neuralnode.integrations.openclaw import OpenClawManager

def main():
    print("🚀 NeuralNode OpenClaw Manager Demo")
    print("=" * 60)

    # 1. Initialize any NeuralNode LLM instance (cloud or local)
    # OpenClaw will route its API requests to this model.
    print("\n1. Initializing NeuralNode...")
    
    # We will use auto-select provider or default mock provider for demonstration.
    # In production, set your API keys (e.g. GEMINI_API_KEY, GROQ_API_KEY, etc.)
    try:
        ai = nn.NeuralNode(provider="auto")
    except Exception:
        # Fallback to a mock provider if no keys are set
        print("   ⚠️ No provider keys found. Falling back to ollama for demonstration.")
        ai = nn.NeuralNode(provider="ollama", model="llama3.2")

    print(f"   Using LLM Provider: {ai.provider_name} (Model: {ai.model})")

    # 2. Instantiate the OpenClawManager
    # This downloader and process runner resides in NeuralNode.
    print("\n2. Initializing OpenClawManager...")
    manager = OpenClawManager(
        node=ai,
        install_dir="./openclaw_runtime",  # Download path for OpenClaw repo
        gateway_port=18789,                # OpenClaw client gateway port
        api_server_port=18788,             # OpenAI-compatible API server port
        api_key="REDACTED_SHARED_TOKEN"    # Gateway authorization key
    )
    print(f"   Runtime Directory: {manager.install_dir}")

    # 3. Setup OpenClaw (Clone git repository & npm install dependencies)
    # Note: Ensure git and node/npm (or pnpm) are installed on your machine.
    print("\n3. Setting up OpenClaw gateway...")
    print("   (Cloning repo and installing Node dependencies; this can take a minute)...")
    try:
        # Check if already installed
        if not (manager.install_dir / "openclaw").exists():
            manager.setup()
            print("   ✅ OpenClaw successfully cloned and dependencies installed!")
        else:
            print("   ✅ OpenClaw runtime already exists.")
            # Ensure configurations are up to date
            manager.configure()
    except Exception as e:
        print(f"   ❌ Setup failed: {e}")
        print("   Make sure Node.js, npm/pnpm, and git are installed and on your PATH.")
        return

    # 4. Start OpenClaw Gateway (Runs uvicorn API and OpenClaw Node gateway process)
    print("\n4. Starting OpenClaw Gateway...")
    try:
        manager.start()
        print("   ✅ OpenClaw API server and Node gateway started successfully!")
    except Exception as e:
        print(f"   ❌ Failed to start: {e}")
        return

    # 5. Use OpenClawClient to control the running agent gateway
    print("\n5. Controlling OpenClaw Gateway...")
    client = manager.get_client()
    
    # Check gateway status
    if client.check_health():
        print("   ✅ OpenClaw Gateway is online and healthy.")
        
        # Create a new session
        session_id = "test-session-1"
        print(f"   Creating session '{session_id}'...")
        client.create_session(session_id)
        
        # Send a prompt/task to the agent
        task = "Introduce yourself in one sentence."
        print(f"   Sending task: '{task}'...")
        response = client.send_task(session_id, task)
        print(f"   Response: {response}")
        
        # Get status/logs of execution
        status = client.get_session_status(session_id)
        print(f"   Session Status: {status.get('status', 'N/A')}")
    else:
        print("   ❌ OpenClaw Gateway did not respond to health check.")

    # 6. Shutting down the gateway processes safely
    print("\n6. Stopping OpenClaw Gateway...")
    manager.stop()
    print("   ✅ OpenClaw stopped successfully.")
    print("=" * 60)
    print("🎉 Demo finished!")

if __name__ == "__main__":
    main()
