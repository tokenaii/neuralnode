import os
import sys

# Test imports
from neuralnode import NeuralNode
from neuralnode.tts import NeuralTTS, tts_list_voices
from neuralnode.speech import NeuralSpeech, SPEECH_LANGUAGES
from neuralnode import DateTime, SystemOps

print("--- Testing Imports and Instantiation ---")

# 1. NeuralNode Simplified API
nn = NeuralNode(provider="ollama", model="llama3.2", model_settings={"context_length": 8192, "gpu_layers": 35})
print("NeuralNode loaded:", nn.provider, nn.model, nn.model_settings)

# 2. Agent Skills
agent = nn.agent(skills=["datetime", "system_control"], system_prompt="Test.")
print("Agent tools loaded:", [t.name for t in agent.tools])

# 3. TTS
print("Voices count:", len(tts_list_voices()))
tts = NeuralTTS(voice="en-US-AriaNeural")
print("TTS Provider:", tts.provider)

# 4. Speech
print("Speech Languages count:", len(SPEECH_LANGUAGES))
sr = NeuralSpeech()

# 5. DateTime
dt = DateTime()
res = dt.execute()
print("DateTime:", res.content.split("\n")[0])

# 6. SystemOps
ops = SystemOps()
print("SystemOps count:", len(ops.OPERATIONS))

# 7. OpenAI Blocker
try:
    NeuralNode(provider="openai", api_key="test")
    print("FAILED: OpenAI was not blocked!")
except ValueError as e:
    print("SUCCESS: OpenAI blocked ->", str(e))

print("\nAll imports successful!")
