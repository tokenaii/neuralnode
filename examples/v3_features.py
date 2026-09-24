"""Current NeuralNode feature showcase."""

import neuralnode as nn


def provider_example():
    llm = nn.create_provider("groq", api_key="YOUR_GROQ_API_KEY", model="llama-3.1-8b-instant")
    response = llm.chat([{"role": "user", "content": "Say hello"}])
    print(response["content"] if isinstance(response, dict) else response.content)


def horus_example():
    model = nn.HorusModel("Horus-1.0-4B").load()
    response = model.chat([{"role": "user", "content": "Explain NeuralNode in one sentence."}])
    print(response.content)


def replica_example():
    print(nn.replica_voice_list())
    tts = nn.ReplicaTTS(voice_id="replic-aria-language{en-us}")
    tts.save_to_file("Hello from NeuralNode", "v3_features_voice.mp3")


def rag_example():
    ai = nn.NeuralNode(provider="ollama", model="llama3.2")
    rag = ai.rag(store="memory")
    rag.add_text("NeuralNode supports Horus, Replica, RAG, and Telegram.")
    print(rag.query("What does NeuralNode support?"))


if __name__ == "__main__":
    provider_example()
    horus_example()
    replica_example()
    rag_example()
