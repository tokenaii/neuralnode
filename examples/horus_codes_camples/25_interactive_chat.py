"""
Interactive terminal chat with Horus model
Press Ctrl+C to exit
"""

import neuralnode as nn

# Load model (choose any available model)
MODEL_ID = "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"

print("="*50)
print("   Horus Chat Terminal")
print("="*50)
print(f"Loading model: {MODEL_ID}")
print("Type 'exit' to quit\n")

# Load model
model = nn.HorusModel(MODEL_ID).load()

# Chat history
history = []

while True:
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "خروج"]:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        print("Horus: ", end="", flush=True)
        
        # Generate response
        response = model.chat(
            [{"role": "user", "content": user_input}],
            max_new_tokens=512
        )
        
        print(response.content)
        
        # Save to history
        history.append(f"User: {user_input}")
        history.append(f"Horus: {response.content}")
        print()
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
