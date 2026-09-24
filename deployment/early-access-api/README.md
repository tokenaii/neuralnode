# Horus Cyper Nano Early Access Gateway

This module is installed beside the existing TokenAI Early Access API on AWS.
It reuses the existing Access Token authentication and PostgreSQL pool. The
deployed API entrypoint is mirrored in `server.production.js`.

The public API contract is:

- `GET /v1/access`
- `GET /v1/models`
- `POST /v1/chat/completions`

Production guarantees:

- One active Access Token may exist for each Early Access user.
- Accepted requests are limited by the per-key RPM value stored in PostgreSQL.
- Every successful completion stores the customer messages and matching model
  response in the same conversation record.
- Agent Mode accepts function tools, tool choices, assistant tool calls, tool
  results, and buffered SSE responses while retaining the complete interaction.
- Production inference uses two CUDA workers with a 32,768-token context. The
  public base URL is `https://api.tokenai.llc/v1`.
- A PostgreSQL-backed FIFO queue limits global model concurrency to two running
  requests. A full queue returns HTTP 503 with `Retry-After` instead of
  overloading the GPU workers.
- Every command is evaluated for Knowledge Distillation and model-output
  training-data generation. Matches retain the user message and model response
  for manual review; detection never disables an Access Token automatically.
- Raw role markers and duplicate output are rejected and retried against the
  alternate worker before the gateway fails closed.
- Deleting a user removes their application, email, token, and personal data so
  the email can register again. Conversation and usage audit records are kept
  and shown as belonging to a deleted identity.

Database setup runs migrations `001` through `005`. They add the one-token
unique index, durable rate-limit events, encrypted Access Token recovery,
concurrency and queue leases, manual distillation review events, public endpoint
throttling, and audit-log retention behavior.

`tokenai-early-access-backup.timer` creates and validates a PostgreSQL custom
dump every day. The dump is encrypted with a unique AES-256 key, the AES key is
wrapped by an RSA public key, and the archive is copied to the model host. Only
the off-instance destination has the RSA private key. The receiver verifies the
checksum, decrypts the dump, and runs `pg_restore --list` before accepting it.
Local encrypted archives are retained for 14 days and off-instance archives for
30 days.

`docker-compose.aws.yml` mirrors the AWS deployment. The permanent named
Cloudflare Tunnel publishes only the private API origin. The edge Worker reaches
that origin through a Cloudflare VPC Service binding; no Quick Tunnel or public
origin address is used.

`tokenai-early-access-monitor.timer` checks the API, database, edge path, model
transport, both workers, queue state, GPU health, latency, disk usage, and backup
age every minute. Alert and recovery messages are delivered through the
configured TokenAI mailbox.

Required model-server configuration:

```dotenv
MODEL_UPSTREAM_URLS=http://model-worker-1/v1,http://model-worker-2/v1
MODEL_UPSTREAM_MODEL=deployed-model-id
MODEL_UPSTREAM_MODE=completion
MODEL_UPSTREAM_API_KEY=
MODEL_UPSTREAM_AUTH_HEADER=Authorization
MODEL_UPSTREAM_AUTH_SCHEME=Bearer
MODEL_UPSTREAM_TIMEOUT_MS=120000
```

`MODEL_UPSTREAM_URLS` accepts a comma-separated list of API base URLs or complete
`/chat/completions` endpoints. Requests are distributed in round-robin order and
retry another worker after a network error or an HTTP 5xx response. The singular
`MODEL_UPSTREAM_URL` remains supported for one-worker deployments. Until either
setting is configured, authenticated requests fail closed with HTTP 503 and are
recorded as failed usage events.

The production-ready transport configuration targets the model host private VPC
address on ports 8011 and 8012. `../model-server/direct-vpc-cutover.md` contains
the single security-group rule that restricts those ports to the Early Access
API security group. Keep the existing pinned SSH transport active until that
rule is applied and both private worker health checks pass.
