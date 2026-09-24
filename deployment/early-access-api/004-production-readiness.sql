BEGIN;

ALTER TABLE api_keys
  ADD COLUMN IF NOT EXISTS safety_strikes INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS disabled_reason TEXT;

CREATE TABLE IF NOT EXISTS inference_queue_leases (
  id UUID PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE,
  api_key_id BIGINT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  special_request_id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('waiting', 'running')),
  queued_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS inference_queue_leases_status_time_idx
  ON inference_queue_leases (status, queued_at, expires_at);

CREATE TABLE IF NOT EXISTS api_safety_events (
  id BIGSERIAL PRIMARY KEY,
  api_key_id BIGINT REFERENCES api_keys(id) ON DELETE SET NULL,
  special_request_id TEXT NOT NULL,
  request_id TEXT NOT NULL,
  conversation_id UUID,
  severity TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  action TEXT NOT NULL CHECK (action IN ('observed', 'blocked', 'token_disabled')),
  categories TEXT[] NOT NULL DEFAULT '{}',
  matched_rules TEXT[] NOT NULL DEFAULT '{}',
  content_sha256 TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_safety_events_request_time_idx
  ON api_safety_events (special_request_id, created_at DESC);

COMMIT;
