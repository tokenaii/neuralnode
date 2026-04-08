"""Horus ready-GGUF usage (including TQ* variants).

Run:
    python examples/horus_tq_ready_gguf.py
"""

from __future__ import annotations

import json

import neuralnode as nn


def main() -> None:
    # Example TQ model id (replace with the real published TQ GGUF file when available)
    tq_model_id = "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-TQ3_1S.gguf"
    q4_model_id = "tokenaii/Hours-1.0-4B-GGUF/Horus-1.0-4B-Q4_K_M.gguf"

    # Optional: register TQ in Horus listed models for nicer discovery.
    nn.HorusModel.register_gguf_variant(
        model_id=tq_model_id,
        name="Horus 1.0 4B (TurboQuant TQ3_1S)",
        size="4B",
    )

    print("Registered models:")
    for model_id in [tq_model_id, q4_model_id]:
        print("-", model_id)

    # Use any ready GGUF model directly by model_id.
    model = nn.HorusModel(
        model_id=q4_model_id,
        cache_dir=r"D:\Work Space\NeuralNode\hf_cache",
        local_files_only=True,
        suppress_warnings=True,
        suppress_native_output=True,
    ).load()

    response = model.chat([{"role": "user", "content": "Say: Horus ready GGUF works."}])
    print("\nChat preview:")
    print(response.content[:200])
    model.unload()

    # Quick benchmark helper (latency-oriented).
    benchmark = nn.HorusModel.benchmark_variants(
        model_ids=[q4_model_id],
        prompt="Explain TurboQuant in one line.",
        runs=2,
        cache_dir=r"D:\Work Space\NeuralNode\hf_cache",
        local_files_only=True,
        max_new_tokens=48,
    )
    print("\nBenchmark:")
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

