"""Integration probe for Horus + Replica + Telegram configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import neuralnode as nn
from neuralnode.providers.horus import HORUS_CONTEXT_WINDOW, UNIFIED_CHAT_TEMPLATE_NAME


def run_probe() -> dict:
    report = {
        "horus": {
            "chat_template": UNIFIED_CHAT_TEMPLATE_NAME,
            "context_window": HORUS_CONTEXT_WINDOW,
            "available_models": list(nn.HorusModel.list_available_models().keys()),
        },
        "replica": {
            "voice_count": len(nn.replica_voice_list()),
            "sample_voices": nn.replica_voice_list()[:3],
        },
        "telegram": {
            "config_fields": sorted(nn.TelegramBotConfig.__dataclass_fields__.keys()),
            "reply_modes_supported": ["text", "voice", "both"],
        },
    }

    if os.getenv("NEURALNODE_LIVE_HORUS", "0") == "1":
        model_id = os.getenv(
            "NEURALNODE_HORUS_MODEL_ID",
            "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf",
        )
        try:
            model = nn.HorusModel(model_id=model_id, max_new_tokens=16, local_files_only=False).load()
            response = model.chat([{"role": "user", "content": "Say hello in one line."}])
            report["live_horus"] = {
                "status": "ok",
                "model_id": model_id,
                "response_preview": response.content[:200],
            }
        except Exception as exc:
            report["live_horus"] = {
                "status": "failed",
                "model_id": model_id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }

    return report


if __name__ == "__main__":
    print(json.dumps(run_probe(), ensure_ascii=False, indent=2))
