import sys
from pathlib import Path

# Add src to path
src_path = str(Path(r'd:\Work Space\NeuralNode\neuralnode\src'))
sys.path.insert(0, src_path)

print("Starting debug import...")

def try_import(name):
    print(f"Importing {name}...", end=" ", flush=True)
    try:
        __import__(f"neuralnode.{name}")
        print("Done.")
    except Exception as e:
        print(f"Failed: {e}")

submodules = [
    "utils.logger",
    "core",
    "providers",
    "memory",
    "rag",
    "agents",
    "tools",
    "vision",
    "tts",
    "speech",
    "integrations.telegram"
]

for sub in submodules:
    try_import(sub)

print("Attempting to import neuralnode top-level...")
import neuralnode
print("All imports successful!")
