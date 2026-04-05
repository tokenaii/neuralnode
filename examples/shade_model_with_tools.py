"""Local model with tools example using the current API."""

import neuralnode as nn


def main():
    ai = nn.NeuralNode(provider="ollama", model="llama3.2")
    agent = ai.agent(
        skills=["web_search", "file", "datetime", "calculator"],
        thinking=False,
    )

    while True:
        task = input("> ").strip()
        if task.lower() in {"quit", "exit"}:
            break
        if not task:
            continue
        print(agent.run(task))


if __name__ == "__main__":
    main()
