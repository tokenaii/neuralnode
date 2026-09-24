SELECT
  metadata->>'upstreamWorker' AS worker,
  metadata->>'upstreamMode' AS mode,
  status_code,
  metadata->>'latencyMs' AS latency_ms,
  created_at
FROM api_usage_events
WHERE event_type = 'chat_completion'
ORDER BY created_at DESC
LIMIT 6;

SELECT
  cm.role,
  left(cm.content, 300) AS content,
  cm.latency_ms,
  cm.created_at
FROM conversation_messages cm
JOIN api_conversations ac ON ac.id = cm.conversation_id
WHERE ac.client_name = 'NeuralNode Gateway Self Test'
ORDER BY cm.created_at DESC
LIMIT 6;
