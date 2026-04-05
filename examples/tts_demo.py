import asyncio
from neuralnode.tts import NeuralTTS

async def main():
    # 1. Initialize NeuralNode TTS with a specific voice
    # List of available voices is in the documentation / README
    tts = NeuralTTS(voice="en-US-AriaNeural")

    # 2. Speak text directly (blocks until finished)
    print("Speaking...")
    tts.speak("Hello! I am NeuralNode 2.0. How can I help you today?")

    # 3. Save speech to a file (non-blocking / async support available)
    print("Saving to file...")
    tts.save("This is a test of the NeuralNode audio system.", "output.mp3")
    print("Saved as output.mp3")

    # 4. List all 20 curated voices
    print("\nAvailable voices:")
    for v in NeuralTTS.list_voices():
        print(f" - {v}")

if __name__ == "__main__":
    # If running in a script, use asyncio.run or just call functions
    # Note: tts.speak() handles the loop internally for convenience.
    asyncio.run(main())
