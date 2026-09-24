BEGIN;

ALTER TABLE api_safety_events
  ADD COLUMN IF NOT EXISTS user_message TEXT,
  ADD COLUMN IF NOT EXISTS model_response TEXT,
  ADD COLUMN IF NOT EXISTS review_status TEXT NOT NULL DEFAULT 'pending',
  ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS reviewed_by TEXT,
  ADD COLUMN IF NOT EXISTS warning_sent_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS warning_email_id BIGINT REFERENCES email_outbox(id) ON DELETE SET NULL;

UPDATE api_safety_events
SET action = 'observed'
WHERE action = 'blocked';

UPDATE api_safety_events
SET review_status = CASE
  WHEN action = 'token_disabled' THEN 'token_disabled'
  ELSE 'pending'
END
WHERE review_status = 'pending';

ALTER TABLE api_safety_events
  DROP CONSTRAINT IF EXISTS api_safety_events_action_check,
  DROP CONSTRAINT IF EXISTS api_safety_events_review_status_check;

ALTER TABLE api_safety_events
  ADD CONSTRAINT api_safety_events_action_check
    CHECK (action IN ('observed', 'warning_sent', 'token_disabled', 'dismissed')),
  ADD CONSTRAINT api_safety_events_review_status_check
    CHECK (review_status IN ('pending', 'warning_sent', 'token_disabled', 'dismissed'));

CREATE UNIQUE INDEX IF NOT EXISTS api_safety_events_request_id_idx
  ON api_safety_events (request_id);

CREATE INDEX IF NOT EXISTS api_safety_events_review_time_idx
  ON api_safety_events (review_status, created_at DESC);

COMMIT;
