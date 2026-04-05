"""Telegram bot example for the current NeuralNode API."""

import os

import neuralnode as nn


def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOTFATHER_TOKEN")
    if token == "YOUR_BOTFATHER_TOKEN":
        raise SystemExit("Set TELEGRAM_BOT_TOKEN before running this example.")

    ai = nn.NeuralNode(
        provider="horus",
        model="tokenaii/horus/Horus-1.0-4B",
    )

    agent = ai.agent(agent_type="simple", thinking=False)

    bot = nn.TelegramBot(
        token=token,
        agent=agent,
        config=nn.TelegramBotConfig(
            token=token,
            enable_voice=True,
            enable_documents=True,
            reply_mode="both",  # "text" | "voice" | "both"
            voice_reply_voice_id="replic-aria-language{en-us}",
            persistence_path="telegram_sessions.json",
        ),
    )

    bot.start()


if __name__ == "__main__":
    main()
