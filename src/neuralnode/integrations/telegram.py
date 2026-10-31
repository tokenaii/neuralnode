"""Telegram Bot Integration for NeuralNode.

Connect an agent to Telegram using a BotFather token, with support for:
- text conversations
- voice message transcription
- optional voice replies through Replica TTS
- document ingestion and analysis
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents import _strip_thinking
from ..core import BaseAgent, ensure_llm_response
from ..rag import load_documents
from ..utils.dependencies import ensure_feature_dependencies

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False


def _refresh_telegram_imports() -> None:
    global TELEGRAM_AVAILABLE, Update, Application, CommandHandler, ContextTypes, MessageHandler, filters
    try:
        from telegram import Update as _Update
        from telegram.ext import (
            Application as _Application,
            CommandHandler as _CommandHandler,
            ContextTypes as _ContextTypes,
            MessageHandler as _MessageHandler,
            filters as _filters,
        )

        Update = _Update
        Application = _Application
        CommandHandler = _CommandHandler
        ContextTypes = _ContextTypes
        MessageHandler = _MessageHandler
        filters = _filters
        TELEGRAM_AVAILABLE = True
    except ImportError:
        TELEGRAM_AVAILABLE = False


@dataclass
class TelegramSession:
    """User session in Telegram bot."""

    user_id: int
    username: Optional[str]
    first_name: str
    chat_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    authorized: bool = True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelegramBotConfig:
    """Configuration for Telegram bot."""

    token: str
    allowed_users: Optional[List[int]] = None
    admin_users: List[int] = field(default_factory=list)
    max_message_length: int = 4096
    enable_voice: bool = True
    enable_documents: bool = True
    response_timeout: int = 60
    show_typing: bool = True
    persistence_path: Optional[str] = None
    reply_mode: str = "text"
    voice_input_language: str = "en-US"
    voice_reply: bool = False
    voice_reply_voice_id: str = "replic-aria-language{en-us}"
    documents_use_rag: bool = True
    document_chunk_limit: int = 6
    max_document_chars: int = 16000
    download_dir: Optional[str] = None
    validate_token_on_start: bool = True
    auto_install_deps: bool = True
    suppress_library_logs: bool = True
    debug_updates: bool = False


class TelegramBot:
    """Telegram Bot integration for NeuralNode agents."""

    def __init__(
        self,
        token: str,
        agent: BaseAgent,
        config: Optional[TelegramBotConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        self.config = config or TelegramBotConfig(token=token)
        if not TELEGRAM_AVAILABLE and self.config.auto_install_deps:
            ensure_feature_dependencies("telegram", auto_install=True)
            _refresh_telegram_imports()
        if not TELEGRAM_AVAILABLE:
            raise ImportError(
                "python-telegram-bot required. Install with: pip install python-telegram-bot"
            )

        self.token = token
        self.agent = agent
        self.system_prompt = system_prompt or "You are a helpful AI assistant on Telegram."

        self.sessions: Dict[int, TelegramSession] = {}
        self.application: Optional[Application] = None
        self._running = False

        if self.config.persistence_path:
            self._load_sessions()

    def _load_sessions(self):
        path = Path(self.config.persistence_path)
        if not path.exists():
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for user_id, session_data in data.items():
                self.sessions[int(user_id)] = TelegramSession(
                    user_id=int(user_id),
                    username=session_data.get("username"),
                    first_name=session_data.get("first_name", "Unknown"),
                    chat_history=session_data.get("chat_history", []),
                    authorized=session_data.get("authorized", True),
                )
        except Exception as exc:
            print(f"Error loading sessions: {exc}")

    def _save_sessions(self):
        if not self.config.persistence_path:
            return

        try:
            data = {}
            for user_id, session in self.sessions.items():
                data[str(user_id)] = {
                    "username": session.username,
                    "first_name": session.first_name,
                    "chat_history": session.chat_history[-50:],
                    "authorized": session.authorized,
                }

            path = Path(self.config.persistence_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"Error saving sessions: {exc}")

    def _get_download_dir(self) -> Path:
        target = Path(
            self.config.download_dir or (Path(tempfile.gettempdir()) / "neuralnode_telegram")
        )
        target.mkdir(parents=True, exist_ok=True)
        return target

    def validate_token(self, timeout: int = 10) -> bool:
        """Validate BotFather token against Telegram `getMe` endpoint."""
        url = f"https://api.telegram.org/bot{self.token}/getMe"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
            return bool(payload.get("ok"))
        except (urllib.error.URLError, TimeoutError, ValueError):
            return False

    async def _download_telegram_file(self, file_id: str, filename: str) -> Path:
        telegram_file = await self.application.bot.get_file(file_id)
        target = self._get_download_dir() / filename
        await telegram_file.download_to_drive(custom_path=str(target))
        return target

    def _get_or_create_session(self, user) -> TelegramSession:
        user_id = user.id
        if user_id not in self.sessions:
            authorized = True
            if self.config.allowed_users:
                authorized = user_id in self.config.allowed_users
            self.sessions[user_id] = TelegramSession(
                user_id=user_id,
                username=user.username,
                first_name=user.first_name or "User",
                authorized=authorized,
            )

        session = self.sessions[user_id]
        session.last_activity = datetime.now()
        return session

    def _build_chat_context(self, session: TelegramSession) -> str:
        if not session.chat_history:
            return ""
        recent = session.chat_history[-20:]
        parts = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _build_user_task(self, session: TelegramSession, user_message: str) -> str:
        chat_context = self._build_chat_context(session)
        if chat_context:
            return f"Previous context:\n{chat_context}\n\nUser message: {user_message}"
        return user_message

    def _configure_runtime_logging(self) -> None:
        if not self.config.suppress_library_logs:
            return
        for logger_name in ("httpx", "httpcore", "telegram", "telegram.ext"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def _run_agent(self, task: str) -> str:
        response = self.agent.run(task)
        if hasattr(response, "output"):
            response = response.output
        else:
            try:
                response = ensure_llm_response(response).content
            except Exception:
                pass

        text = _strip_thinking(str(response))
        text = re.sub(r"Thought:.*?\n", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"REASONING:.*?\n", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    async def _run_agent_with_timeout(self, task: str) -> str:
        timeout = max(1, int(self.config.response_timeout or 60))
        return await asyncio.wait_for(asyncio.to_thread(self._run_agent, task), timeout=timeout)

    def _record_exchange(self, session: TelegramSession, user_message: str, response: str):
        session.chat_history.append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now().isoformat(),
            }
        )
        session.chat_history.append(
            {
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
            }
        )
        if len(session.chat_history) > 100:
            session.chat_history = session.chat_history[-100:]
        self._save_sessions()

    def _normalized_reply_mode(self) -> str:
        # Backward compatibility: old `voice_reply=True` means both text + voice.
        if self.config.voice_reply and self.config.reply_mode == "text":
            return "both"
        mode = (self.config.reply_mode or "text").strip().lower()
        if mode not in {"text", "voice", "both"}:
            return "text"
        return mode

    async def _send_long_message(self, update: Update, text: str):
        max_length = self.config.max_message_length
        if len(text) <= max_length:
            await update.message.reply_text(text)
            return

        chunks = []
        while text:
            if len(text) <= max_length:
                chunks.append(text)
                break
            split_at = text.rfind("\n", 0, max_length)
            if split_at == -1:
                split_at = max_length
            chunks.append(text[:split_at])
            text = text[split_at:].strip()

        for index, chunk in enumerate(chunks, 1):
            header = f"Part {index}/{len(chunks)}:\n" if len(chunks) > 1 else ""
            await update.message.reply_text(header + chunk)

    async def _send_voice_reply(self, update: Update, text: str):
        try:
            from ..replica import ReplicaTTS
        except ImportError:
            return

        output_path = self._get_download_dir() / (
            f"telegram_reply_{update.effective_user.id}_{int(datetime.now().timestamp())}.mp3"
        )
        tts = ReplicaTTS(voice_id=self.config.voice_reply_voice_id)
        tts.save_to_file(text, str(output_path))
        with open(output_path, "rb") as audio_file:
            await update.message.reply_voice(voice=audio_file)

    async def _send_response(self, update: Update, response: str):
        mode = self._normalized_reply_mode()
        if mode in {"text", "both"}:
            await self._send_long_message(update, response)
        if mode in {"voice", "both"}:
            try:
                await self._send_voice_reply(update, response)
            except Exception as exc:
                # If voice-only mode fails, fall back to text so the user still gets a reply.
                await update.message.reply_text(f"Voice reply failed: {exc}")
                if mode == "voice":
                    await self._send_long_message(update, response)

    def _convert_audio_for_transcription(self, path: Path) -> Path:
        if path.suffix.lower() in {".wav", ".aiff", ".aif", ".flac", ".mp3"}:
            return path

        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise ImportError(
                "pydub is required for Telegram voice transcription. Install with: pip install pydub"
            ) from exc

        converted_path = path.with_suffix(".mp3")
        audio = AudioSegment.from_file(str(path))
        audio.export(str(converted_path), format="mp3")
        return converted_path

    def _extract_document_text(self, file_path: Path) -> str:
        docs = load_documents([file_path])
        if not docs:
            raise ValueError(f"No readable content extracted from {file_path.name}")

        text = "\n\n".join(doc.content for doc in docs if doc.content.strip())
        if not text.strip():
            raise ValueError(f"Document {file_path.name} did not contain readable text")
        return text[: self.config.max_document_chars]

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_or_create_session(update.effective_user)
        if not session.authorized:
            await update.message.reply_text(
                "You are not authorized to use this bot. Contact the bot administrator for access."
            )
            return

        await update.message.reply_text(
            (
                f"Hello {session.first_name}!\n\n"
                "I'm an AI assistant powered by NeuralNode.\n\n"
                "Commands:\n"
                "/help - Show help\n"
                "/clear - Clear chat history\n"
                "/status - Show bot status\n"
                "/about - About this bot"
            )
        )

    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            (
                "Help - Available Commands:\n\n"
                "/start - Start the bot\n"
                "/help - Show this help message\n"
                "/clear - Clear your conversation history\n"
                "/status - Check bot status\n"
                "/about - About this bot\n\n"
                "Send a document to analyze it.\n"
                "Send a voice message to transcribe it.\n"
                "Send normal text to chat with the model."
            )
        )

    async def _clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user.id in self.sessions:
            self.sessions[user.id].chat_history.clear()
            self._save_sessions()
        await update.message.reply_text("Conversation history cleared.")

    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        session = self._get_or_create_session(update.effective_user)
        await update.message.reply_text(
            (
                "Bot Status:\n\n"
                f"Name: {session.first_name}\n"
                f"ID: {session.user_id}\n"
                f"Messages: {len(session.chat_history)}\n"
                f"Agent Type: {type(self.agent).__name__}\n"
                f"Voice enabled: {self.config.enable_voice}\n"
                f"Documents enabled: {self.config.enable_documents}\n"
                f"Reply mode: {self._normalized_reply_mode()}\n"
                f"Last activity: {session.last_activity.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        )

    async def _about_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            (
                "About This Bot\n\n"
                "Powered by NeuralNode.\n"
                "Supports chat, document analysis, and optional voice workflows."
            )
        )

    async def _handle_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        session = self._get_or_create_session(user)
        if not session.authorized:
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        user_message = update.message.text or ""
        if self.config.debug_updates:
            print(f"[Telegram][text] user={user.id} message={user_message[:200]}")
        if self.config.show_typing:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            task = self._build_user_task(session, user_message)
            response = await self._run_agent_with_timeout(task)
            self._record_exchange(session, user_message, response)
            await self._send_response(update, response)
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "Model response timed out. Try a shorter prompt or reduce max tokens."
            )
        except Exception as exc:
            await update.message.reply_text(f"Error processing your message: {str(exc)[:200]}")

    async def _handle_voice_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config.enable_voice:
            await update.message.reply_text("Voice messages are disabled.")
            return

        user = update.effective_user
        session = self._get_or_create_session(user)
        if not session.authorized:
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        try:
            from ..speech import NeuralSpeech
        except ImportError:
            await update.message.reply_text(
                "Speech recognition dependencies are missing. Install SpeechRecognition and pydub."
            )
            return

        if self.config.show_typing:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        voice = update.message.voice
        file_name = f"voice_{user.id}_{voice.file_unique_id}.ogg"
        if self.config.debug_updates:
            print(f"[Telegram][voice] user={user.id} file={file_name}")

        try:
            downloaded = await self._download_telegram_file(voice.file_id, file_name)
            transcribable = self._convert_audio_for_transcription(downloaded)
            speech = NeuralSpeech(language=self.config.voice_input_language)
            transcript = speech.transcribe(
                str(transcribable),
                language=self.config.voice_input_language,
            )

            task = self._build_user_task(session, transcript)
            response = await self._run_agent_with_timeout(task)
            self._record_exchange(session, transcript, response)

            await update.message.reply_text(f"Transcript: {transcript}")
            await self._send_response(update, response)
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "Model response timed out after transcription. Try shorter audio or a shorter prompt."
            )
        except Exception as exc:
            await update.message.reply_text(f"Voice transcription failed: {str(exc)[:300]}")

    async def _handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.config.enable_documents:
            await update.message.reply_text("Document handling is disabled.")
            return

        user = update.effective_user
        session = self._get_or_create_session(user)
        if not session.authorized:
            await update.message.reply_text("You are not authorized to use this bot.")
            return

        if self.config.show_typing:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        document = update.message.document
        safe_name = document.file_name or f"document_{document.file_unique_id}"
        if self.config.debug_updates:
            print(f"[Telegram][document] user={user.id} file={safe_name}")
        try:
            downloaded = await self._download_telegram_file(document.file_id, safe_name)
            extracted_text = self._extract_document_text(downloaded)

            task = (
                f"Analyze the uploaded document `{safe_name}`.\n\n"
                f"Document content:\n{extracted_text}\n\n"
                "Provide a concise summary, key points, risks, and actionable insights."
            )
            response = await self._run_agent_with_timeout(task)
            self._record_exchange(session, f"[Document upload] {safe_name}", response)

            header = f"Document received: {safe_name}\nSize: {document.file_size / 1024:.1f} KB\n\n"
            await self._send_response(update, header + response)
        except asyncio.TimeoutError:
            await update.message.reply_text(
                "Model response timed out while analyzing document. Try a smaller file."
            )
        except Exception as exc:
            await update.message.reply_text(f"Document analysis failed: {str(exc)[:300]}")

    def _setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("clear", self._clear_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(CommandHandler("about", self._about_command))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text_message)
        )
        if self.config.enable_voice:
            self.application.add_handler(MessageHandler(filters.VOICE, self._handle_voice_message))
        if self.config.enable_documents:
            self.application.add_handler(
                MessageHandler(filters.Document.ALL, self._handle_document)
            )
        self.application.add_error_handler(self._handle_error)

    async def _handle_error(self, update: object, context: ContextTypes.DEFAULT_TYPE):
        print(f"Telegram bot error: {context.error}")
        if self.config.debug_updates:
            logging.exception("Telegram update error", exc_info=context.error)

    def start(self, poll_interval: float = 1.0):
        if self._running:
            print("Bot is already running.")
            return
        if self.config.validate_token_on_start and not self.validate_token():
            raise ValueError(
                "Invalid Telegram bot token. Verify your BotFather token and network connectivity."
            )

        self._configure_runtime_logging()
        self.application = Application.builder().token(self.token).build()
        self._setup_handlers()
        self._running = True
        self.application.run_polling(poll_interval=poll_interval)

    def start_async(self) -> asyncio.Task:
        if self._running:
            return None
        self._running = True

        async def run_bot():
            if self.config.validate_token_on_start and not self.validate_token():
                raise ValueError(
                    "Invalid Telegram bot token. Verify your BotFather token and network connectivity."
                )
            self._configure_runtime_logging()
            self.application = Application.builder().token(self.token).build()
            self._setup_handlers()
            await self.application.initialize()
            await self.application.start()
            await self.application.updater.start_polling()

        return asyncio.create_task(run_bot())

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self.application:
            asyncio.create_task(self.application.stop())
        self._save_sessions()

    def send_message(self, user_id: int, message: str) -> bool:
        if not self.application:
            return False

        async def send():
            try:
                await self.application.bot.send_message(
                    chat_id=user_id,
                    text=message[: self.config.max_message_length],
                )
                return True
            except Exception:
                return False

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(send())

    def broadcast_message(self, message: str) -> Dict[int, bool]:
        results = {}
        for user_id, session in self.sessions.items():
            if session.authorized:
                results[user_id] = self.send_message(user_id, message)
        return results


class TelegramBotBuilder:
    """Builder for creating Telegram bots with agents."""

    def __init__(self):
        self.token: Optional[str] = None
        self.agent: Optional[BaseAgent] = None
        self.config = TelegramBotConfig(token="")
        self.system_prompt: Optional[str] = None

    def with_token(self, token: str) -> "TelegramBotBuilder":
        self.token = token
        self.config.token = token
        return self

    def with_token_from_env(self, env_var: str = "TELEGRAM_BOT_TOKEN") -> "TelegramBotBuilder":
        self.token = os.getenv(env_var)
        if not self.token:
            raise ValueError(f"Environment variable {env_var} not set")
        self.config.token = self.token
        return self

    def with_agent(self, agent: BaseAgent) -> "TelegramBotBuilder":
        self.agent = agent
        return self

    def allow_users(self, user_ids: List[int]) -> "TelegramBotBuilder":
        self.config.allowed_users = user_ids
        return self

    def with_persistence(self, path: str) -> "TelegramBotBuilder":
        self.config.persistence_path = path
        return self

    def with_system_prompt(self, prompt: str) -> "TelegramBotBuilder":
        self.system_prompt = prompt
        return self

    def build(self) -> TelegramBot:
        if not self.token:
            raise ValueError("Bot token required. Get one from @BotFather")
        if not self.agent:
            raise ValueError("Agent required")
        return TelegramBot(
            token=self.token,
            agent=self.agent,
            config=self.config,
            system_prompt=self.system_prompt,
        )


def create_telegram_bot(
    token: str,
    agent: BaseAgent,
    allowed_users: Optional[List[int]] = None,
) -> TelegramBot:
    config = TelegramBotConfig(token=token, allowed_users=allowed_users)
    return TelegramBot(token=token, agent=agent, config=config)


TelegramAgent = TelegramBot
