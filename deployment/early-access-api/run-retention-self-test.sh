#!/bin/sh
set -eu

cleanup() {
  kill "${api_pid:-}" "${model_pid:-}" 2>/dev/null || true
}

trap cleanup EXIT

node mock-model-server.js >/tmp/horus-mock-model.log 2>&1 &
model_pid=$!

PORT=18081 \
MODEL_UPSTREAM_URL=http://127.0.0.1:19090/v1 \
MODEL_UPSTREAM_MODEL=horus-cyper-nano-1.0-beta \
SMTP_HOST= SMTP_USER= SMTP_PASS= SMTP_FROM_EMAIL= \
node server.js >/tmp/horus-secondary-api.log 2>&1 &
api_pid=$!

node retention-rpm-self-test.js
node agent-mode-self-test.js
node security-hardening-self-test.js
