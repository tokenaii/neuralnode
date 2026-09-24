"""Horus Cyper Nano 1.0 BETA Text-to-Text Early Access example."""

import os

import neuralnode as nn


def main() -> None:
    client = nn.NeuralNode(
        provider="horus-cyper-nano",
        model="horus-cyper-nano-1.0-beta",
        api_key=os.environ["HORUS_CYPER_NANO_API_KEY"],
    )

    answer = client.chat(
        "Review this authentication flow for defensive security weaknesses."
    )
    print(answer)


if __name__ == "__main__":
    main()
