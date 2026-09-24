# Horus Cyper Nano model workers

The production model host runs two independent CUDA-backed `llama-server`
workers from the Q4_K_M GGUF. Both workers use the same private authentication
key and listen only on the model host's VPC address:

- `172.31.28.151:8011`
- `172.31.28.151:8012`

Each worker has one inference slot and a 32,768-token context. The Early Access
gateway distributes requests across healthy workers and remains responsible for
public Access Token authentication, per-token RPM enforcement, concurrency
control, and durable conversation logging.

The gateway currently reaches the workers through a hardened systemd SSH
transport bound only to `172.18.0.1` on the gateway host. Strict host-key
checking, a dedicated identity, fixed forwards, and systemd sandboxing are
enabled. This transport must be replaced with direct VPC connectivity by
applying the source-security-group rule in `direct-vpc-cutover.md`; the worker
ports must never be opened to the public internet.

This host is also the encrypted off-instance backup destination. The RSA private
key never leaves the model host, and the forced-command backup account can only
receive, validate, decrypt, and inspect backup archives under
`/data/early-access-backups`.

`tokenai-model-metrics` exposes worker, GPU, temperature, utilization, and disk
status only to the authenticated gateway monitor.

The CUDA runtime uses llama.cpp `b10298` with the response-parser fallback patch
stored beside these deployment files. NeuralNode parses tool-call XML after the
worker returns raw model content.

The secret file `/etc/tokenai/horus-cyper-model.env` must be owned by root with
mode `0600` and contain `MODEL_INTERNAL_API_KEY`. Never place that value in the
repository.
