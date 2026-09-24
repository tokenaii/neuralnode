BEGIN;

ALTER TABLE api_keys
  ADD COLUMN IF NOT EXISTS key_ciphertext TEXT,
  ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'api_keys_rpm_limit_check'
  ) THEN
    ALTER TABLE api_keys
      ADD CONSTRAINT api_keys_rpm_limit_check CHECK (rpm_limit BETWEEN 1 AND 600);
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint WHERE conname = 'access_requests_rpm_limit_check'
  ) THEN
    ALTER TABLE access_requests
      ADD CONSTRAINT access_requests_rpm_limit_check CHECK (rpm_limit BETWEEN 1 AND 600);
  END IF;
END
$$;

CREATE TABLE IF NOT EXISTS public_request_rate_events (
  id BIGSERIAL PRIMARY KEY,
  fingerprint TEXT NOT NULL,
  event_type TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS public_request_rate_events_lookup_idx
  ON public_request_rate_events (fingerprint, event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS api_inflight_requests (
  id UUID PRIMARY KEY,
  api_key_id BIGINT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_inflight_requests_key_expiry_idx
  ON api_inflight_requests (api_key_id, expires_at);

COMMIT;
