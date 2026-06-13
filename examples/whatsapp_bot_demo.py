"""
NeuralNode WhatsApp AI Chatbot & Agent Demo
==========================================

This script demonstrates how to start the WhatsApp Gateway (Evolution API),
link a WhatsApp account (via QR code or Pairing code), setup a local webhook
listener, and run an automated AI Agent chatbot on WhatsApp.
"""

import time
import os
import sys
from pathlib import Path
import neuralnode as nn
from neuralnode.integrations.whatsapp import EvolutionWhatsAppBot

def main():
    print("🚀 NeuralNode WhatsApp AI Chatbot Demo")
    print("=" * 60)

    # 1. Initialize the NeuralNode Agent
    print("\n1. Initializing AI Agent...")
    try:
        # Using auto-select provider or local Ollama
        ai = nn.NeuralNode(provider="auto")
    except Exception:
        print("   ⚠️ No provider API keys found. Falling back to ollama.")
        ai = nn.NeuralNode(provider="ollama", model="llama3.2")
        
    # Give the agent some tools/skills
    agent = ai.agent(skills=["web_search", "datetime"], system_prompt="You are a helpful WhatsApp AI Assistant.")
    print(f"   Agent initialized with capabilities: {ai.provider_name} (Model: {ai.model})")

    # 2. Initialize the Evolution WhatsApp Bot
    print("\n2. Initializing Evolution WhatsApp Bot...")
    bot = EvolutionWhatsAppBot(
        api_url="http://localhost:8080",
        global_apikey="my-secret-key",
        agent=agent
    )

    # 3. Start the local WhatsApp Gateway
    # Note: This will run `npm install` and start the gateway process in background.
    print("\n3. Starting WhatsApp Gateway...")
    print("   (This runs the Node.js Baileys gateway under the hood. Wait ~15s)...")
    try:
        bot.start_gateway(port=8080, wait_seconds=15)
        print("   ✅ WhatsApp Gateway is running.")
    except Exception as e:
        print(f"   ❌ Failed to start WhatsApp Gateway: {e}")
        print("   Please check that Node.js (v18+) is installed and on your PATH.")
        return

    instance_name = "neuralnode-bot"

    # 4. Linking your WhatsApp Account
    # You can connect via QR Code (scanning it on your phone) OR Pairing Code (entering it on your phone).
    print("\n4. Linking Account to WhatsApp...")
    print("   Choose your connection method:")
    print("   [1] QR Code (recommended - generates a PNG to scan)")
    print("   [2] Pairing Code (enter your phone number to get an 8-character code)")
    
    choice = input("Select connection method (1/2) [Default: 1]: ").strip() or "1"
    
    if choice == "2":
        phone_number = input("Enter your WhatsApp phone number with country code (e.g. +20123456789): ").strip()
        if not phone_number:
            print("   ❌ Phone number is required for pairing code. Exiting.")
            bot.stop_gateway()
            return
        
        print(f"   Requesting pairing code for {phone_number}...")
        try:
            pairing_code = bot.connect_phone(instance_name, phone_number)
            print("\n" + "*" * 50)
            print(f"👉 PAIRING CODE: {pairing_code}")
            print("To link your phone:")
            print("1. Open WhatsApp -> Linked Devices -> Link with Phone Number.")
            print(f"2. Enter the Pairing Code above: {pairing_code}")
            print("*" * 50 + "\n")
        except Exception as e:
            print(f"   ❌ Failed to get pairing code: {e}")
            bot.stop_gateway()
            return
    else:
        print("   Generating QR code...")
        try:
            qr_path = bot.connect_qr(instance_name)
            print("\n" + "*" * 50)
            print(f"👉 QR CODE SAVED TO: {qr_path}")
            print("To link your phone:")
            print("1. Open WhatsApp -> Linked Devices -> Link a Device.")
            print(f"2. Scan the QR code image saved at the path above.")
            print("*" * 50 + "\n")
        except Exception as e:
            print(f"   ❌ Failed to generate QR code: {e}")
            bot.stop_gateway()
            return

    # 5. Start the Webhook Listener & configure gateway webhook
    # Baileys gateway will POST incoming messages to this webhook,
    # which will then run the agent and automatically reply back.
    print("\n5. Starting Webhook Listener and Router...")
    webhook_port = 5000
    try:
        # Start a local HTTP server on port 5000
        bot.start_webhook_listener(port=webhook_port)
        
        # Configure the gateway to send updates to our listener
        # In a real environment, you'd use a public URL (e.g., ngrok or production domain)
        # For local development with docker/local processes, use localhost
        webhook_url = f"http://localhost:{webhook_port}"
        bot.set_webhook(instance_name, webhook_url)
        print(f"   ✅ Webhook listener online at {webhook_url}")
    except Exception as e:
        print(f"   ❌ Failed to initialize webhook: {e}")
        bot.stop_gateway()
        return

    # 6. Wait and handle chat chatbot conversations
    print("\n6. Bot is running! Waiting for messages...")
    print("   Send a message to your connected WhatsApp number from another phone.")
    print("   The chatbot agent will process it and respond automatically.")
    print("   Press Ctrl+C to terminate.")
    
    try:
        # Keep the script running to listen to webhook events
        while True:
            state = bot.connection_state(instance_name)
            print(f"   [Status Check] Current connection state: '{state}'", end="\r", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping WhatsApp Bot and Gateway...")
    finally:
        # Cleanup
        bot.stop_webhook_listener()
        bot.stop_gateway()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
