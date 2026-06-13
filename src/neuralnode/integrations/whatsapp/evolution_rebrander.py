"""Programmatic rebrander for Evolution API integration in NeuralNode."""

import os
import re
from pathlib import Path

# Replacements mapping: (regex pattern, replacement, flags)
REPLACEMENTS = [
    (r"\bEvolution API\b", "NeuralNode WhatsApp Gateway"),
    (r"\bevolution-api\b", "neuralnode-whatsapp-gateway"),
    (r"\bEvolution Foundation\b", "NeuralNode Contributors"),
    (r"\bevolution-foundation\b", "neuralnode-contributors"),
]


def rebrand_gateway(target_dir: str):
    """Walk through the cloned directory and apply rebranding replacements."""
    print(f"Starting rebranding in {target_dir}...")
    
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"Target directory {target_dir} does not exist. Skipping.")
        return
        
    text_extensions = {".json", ".ts", ".js", ".md", ".html", ".css", ".env", ".yml", ".yaml", ".txt"}
    
    count = 0
    for root, dirs, files in os.walk(target_path):
        # Skip .git directory
        if ".git" in dirs:
            dirs.remove(".git")
            
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in text_extensions:
                continue
                
            try:
                # Read content
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    
                # Apply replacements
                modified = content
                for pattern, replacement in REPLACEMENTS:
                    modified = re.sub(pattern, replacement, modified)
                    
                # Write back if changed
                if modified != content:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(modified)
                    count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    print(f"Rebranding completed! Modified {count} files.")


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    rebrand_gateway(str(current_dir / "evolution-api"))
