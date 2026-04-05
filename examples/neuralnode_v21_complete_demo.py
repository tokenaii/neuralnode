"""Current end-to-end NeuralNode demo."""

import neuralnode as nn


def main():
    ai = nn.NeuralNode(provider="ollama", model="llama3.2")
    print("Chat:", ai.chat("Introduce yourself briefly."))

    agent = ai.agent(skills=["datetime"], thinking=False)
    print("Agent:", agent.run("What time is it now?"))

    rag = ai.rag(store="memory")
    rag.add_text("NeuralNode includes Horus, Replica, Telegram, and TurboQuant support.")
    print("RAG:", rag.query("What are the main framework features?"))

    print("Replica voices:", nn.replica_voice_list()[:3])


if __name__ == "__main__":
    main()
