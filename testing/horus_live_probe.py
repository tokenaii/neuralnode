import sys
import traceback

import neuralnode as nn


def main() -> int:
    model_id = "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"
    print(f"probe_start model_id={model_id}", flush=True)
    try:
        model = nn.HorusModel(model_id=model_id, local_files_only=False, max_new_tokens=8)
        print("probe_constructed", flush=True)
        model.load()
        print("probe_loaded", flush=True)
        response = model.chat([{"role": "user", "content": "Say hello in one line."}])
        print("probe_chat_ok", response.content[:200], flush=True)
        return 0
    except Exception as exc:
        print(f"probe_failed type={type(exc).__name__} msg={str(exc)[:400]}", flush=True)
        traceback.print_exc(limit=1)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
