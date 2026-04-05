# Telegram Guide

This guide reflects the current NeuralNode Telegram integration.

## Requirements

Install Telegram support:

```bash
pip install "neuralnode[telegram]"
```

Create a bot token:
1. Open Telegram
2. Message `@BotFather`
3. Run `/newbot`
4. Copy the token

## Basic Setup

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="anthropic",
    model="claude-3-haiku-20240307",
    api_key="YOUR_ANTHROPIC_API_KEY",
)

agent = ai.agent(agent_type="simple", thinking=False)

bot = nn.TelegramBot(
    token="YOUR_BOTFATHER_TOKEN",
    agent=agent,
    config=nn.TelegramBotConfig(token="YOUR_BOTFATHER_TOKEN"),
)

bot.start()
```

## Horus + Telegram

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="horus",
    model="tokenaii/horus/Horus-1.0-4B",
)

agent = ai.agent(agent_type="simple", thinking=False)

bot = nn.TelegramBot(
    token="YOUR_BOTFATHER_TOKEN",
    agent=agent,
    config=nn.TelegramBotConfig(
        token="YOUR_BOTFATHER_TOKEN",
        enable_voice=True,
        enable_documents=True,
        reply_mode="both",
        voice_reply_voice_id="replic-salma-language{ar-eg}",
    ),
)

bot.start()
```

## Features

Current Telegram support includes:
- text chat
- persistent chat sessions
- voice download and transcription
- optional Replica voice replies
- document download and analysis

## Voice Handling

Telegram voice support uses:
- `SpeechRecognition`
- `pydub`

NeuralNode downloads the voice note, converts it when needed, transcribes it, then sends the transcript to the agent.

## Document Handling

Document support uses:
- `pypdf` for PDF files
- `python-docx` for DOCX files
- built-in text loaders for `.txt`, `.md`, `.json`, `.csv`, and code files

Uploaded documents are downloaded locally, parsed, then summarized/analyzed by the agent.

## Configuration

```python
config = nn.TelegramBotConfig(
    token="YOUR_BOTFATHER_TOKEN",
    allowed_users=[123456789],
    enable_voice=True,
    enable_documents=True,
    reply_mode="voice",  # text | voice | both
    voice_input_language="en-US",
    voice_reply_voice_id="replic-aria-language{en-us}",
    persistence_path="telegram_sessions.json",
)
```

## Replica Voice Replies

```python
import neuralnode as nn

for voice in nn.replica_voice_list():
    print(voice["id"])
```

Use one of the custom `replic-*` IDs in `voice_reply_voice_id`.

## Troubleshooting

- If voice transcription fails, install `SpeechRecognition` and `pydub`.
- If voice replies fail, install `edge-tts`.
- If PDF or DOCX parsing fails, install `pypdf` or `python-docx`.
- If using Horus GGUF, install `llama-cpp-python`.
