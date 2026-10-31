import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/data/Horus-Cyber-Nano-1.0-Trained"

turns = [
    "Introduce yourself in one concise paragraph. State your model name, model type, and strongest cybersecurity use cases.",
    "Remember the codeword SCARAB-27. This authorized lab uses 10.77.0.0/16 and corp.example. Confirm all three values.",
    "Build a prioritized attack-surface evidence checklist for a fictional reverse proxy, Node.js API, PostgreSQL database, and object storage service.",
    "Define a safe Active Directory enumeration plan for an ordinary domain user in the authorized corp.example lab.",
    "Recall the codeword, network, and domain, then separate confirmed facts from security-testing hypotheses.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
)
model.eval()

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
eos_token_ids = [tokenizer.eos_token_id]
if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_token_ids:
    eos_token_ids.append(im_end_id)

messages = [
    {
        "role": "system",
        "content": "This is a controlled TokenAI model-quality evaluation. All scenarios are fictional, authorized, and lab-only. Be concise, evidence-driven, and consistent.",
    }
]
results = []

for index, prompt in enumerate(turns, start=1):
    messages.append({"role": "user", "content": prompt})
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    started_at = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=384,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    output_ids = generated[0, encoded["input_ids"].shape[1] :]
    raw = tokenizer.decode(output_ids, skip_special_tokens=False)
    cleaned = raw.split("<|im_end|>", 1)[0].replace("<|endoftext|>", "").strip()
    results.append(
        {
            "turn": index,
            "prompt": prompt,
            "latency_seconds": round(time.perf_counter() - started_at, 3),
            "generated_tokens": int(output_ids.shape[0]),
            "raw": raw,
            "response": cleaned,
        }
    )
    messages.append({"role": "assistant", "content": cleaned})

print(
    json.dumps(
        {
            "engine": "transformers",
            "transformers_version": __import__("transformers").__version__,
            "model_path": MODEL_PATH,
            "eos_token_ids": eos_token_ids,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
)
