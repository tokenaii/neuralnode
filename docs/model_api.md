# NeuralNode Model API

NeuralNode can expose a vLLM-compatible API that authenticates `nn-` access tokens and forwards chat completions to a model server on another machine. Tokens contain 24 characters: the `nn-` prefix and 21 cryptographically random alphanumeric characters. SQLite stores SHA-256 token hashes; the full token is shown only when created.

## Start H-Preview on its GPU server

Install vLLM on the remote inference server, then start the Hugging Face model with its public repository ID:

```bash
pip install vllm
vllm serve DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU \
  --host 0.0.0.0 --port 8000
```

Keep that port private to the NeuralNode gateway using a private network, firewall, or TLS reverse proxy. The Hugging Face model card also documents a vLLM-compatible `/v1/chat/completions` endpoint.

## Start the NeuralNode gateway

Install the API extra on the gateway machine:

```bash
pip install "neuralnode[api]"
neuralnode-api create-admin
```

Save the one-time admin token securely. The owner admin token is unique to the database and exempt from RPM limits. Use it to issue `basic` or `elite` user tokens through `POST /v1/admin/tokens`. Basic tokens have a configurable RPM limit (60 by default); Elite tokens have no application RPM limit and their model requests are not recorded in the RPM counter. Both tiers can be revoked by the admin. Set the gateway environment and start it:

```bash
H_PREVIEW_UPSTREAM_URL=http://GPU_SERVER:8000/v1
H_PREVIEW_UPSTREAM_MODEL=DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU
NEURALNODE_API_DB=/var/lib/neuralnode/api.sqlite3
neuralnode-api serve --host 0.0.0.0 --port 8080
```

On PowerShell, use `$env:H_PREVIEW_UPSTREAM_URL`, `$env:H_PREVIEW_UPSTREAM_MODEL`, and `$env:NEURALNODE_API_DB` to set those values before starting the server. If the upstream server requires authentication, set `H_PREVIEW_UPSTREAM_API_KEY` on the gateway only.

The gateway exposes authenticated `GET /v1/models` and `POST /v1/chat/completions`. Requests accept model name `h-preview` and are forwarded to the configured upstream model. Streaming chat completions are passed through as SSE.

## Create tokens and call from Python

Create a Basic token with an RPM limit or an Elite token without one:

```python
import requests

ADMIN_TOKEN = "paste-the-one-time-admin-token-here"
result = requests.post(
    "https://api.example.com/v1/admin/tokens",
    headers={"Authorization": "Bearer " + ADMIN_TOKEN},
    json={"tier": "basic", "rpm_limit": 120},
).json()
basic_token = result["token"]

elite = requests.post(
    "https://api.example.com/v1/admin/tokens",
    headers={"Authorization": "Bearer " + ADMIN_TOKEN},
    json={"tier": "elite"},
).json()
elite_token = elite["token"]
```

Use the generated user token through NeuralNode:

```python
import neuralnode as nn

ai = nn.NeuralNode(
    provider="h-preview",
    model="h-preview",
    api_key=elite_token,
    base_url="https://api.example.com/v1",
)
print(ai.chat("Say hello in one sentence."))
```

Alternatively set `NEURALNODE_API_KEY` and `NEURALNODE_API_BASE_URL` so the client reads them automatically. Only the owner should hold the admin token, since it can manage user credentials. Elite removes NeuralNode's per-token RPM accounting; it does not remove GPU capacity, server availability, or upstream infrastructure limits.
