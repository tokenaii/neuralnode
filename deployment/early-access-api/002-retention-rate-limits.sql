BEGIN;

ALTER TABLE api_conversations
  DROP CONSTRAINT IF EXISTS api_conversations_special_request_id_fkey;

ALTER TABLE api_usage_events
  DROP CONSTRAINT IF EXISTS api_usage_events_special_request_id_fkey;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM api_keys
    GROUP BY special_request_id
    HAVING count(*) > 1
  ) THEN
    RAISE EXCEPTION 'Cannot enforce one Access Token per user while duplicate API keys exist.';
  END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS api_keys_one_per_user_idx
  ON api_keys (special_request_id);

CREATE TABLE IF NOT EXISTS api_request_rate_events (
  id BIGSERIAL PRIMARY KEY,
  api_key_id BIGINT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  special_request_id TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_request_rate_events_key_time_idx
  ON api_request_rate_events (api_key_id, created_at DESC);

COMMIT;
