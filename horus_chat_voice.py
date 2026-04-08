"""
Horus Chat with Replica TTS (Female EN-US Voice)
Run: pip install edge-tts>=6.1.0
"""
import neuralnode as nn
import os
import platform

# Initialize Replica TTS with female EN-US voice
tts = nn.ReplicaTTS(voice_id="replic-aria-language{en-us}")

def play_audio(file_path):
    """Play audio file based on OS"""
    system = platform.system()
    try:
        if system == "Windows":
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif system == "Darwin":  # macOS
            os.system(f"afplay '{file_path}' &")
        else:  # Linux
            os.system(f"mpg321 '{file_path}' & 2>/dev/null || aplay '{file_path}' & 2>/dev/null || cvlc '{file_path}' --play-and-exit & 2>/dev/null")
    except Exception as e:
        print(f"⚠️ Could not play audio: {e}")
        print(f"📁 Audio saved to: {file_path}")

# Load Horus model
model = nn.HorusModel(
    model_id="tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"
).load()

messages = [
    {"role": "system", "content": "You're a medical AI assistant."}
]

print("Horus chat started. Type 'exit' to quit.\n")

while True:
    user_text = input("You: ").strip()
    if not user_text:
        continue
    if user_text.lower() in {"exit", "quit", "q"}:
        print("Bye.")
        break

    messages.append({"role": "user", "content": user_text})
    resp = model.chat(messages)
    assistant_text = resp.content.strip()

    print(f"Horus: {assistant_text}\n")
    
    # Generate and play audio
    print("🔊 Generating audio...")
    audio_file = tts.speak(assistant_text, blocking=True)
    print(f"🎵 Playing: {audio_file}")
    play_audio(audio_file)
    
    messages.append({"role": "assistant", "content": assistant_text})

# optional cleanup
model.unload()
