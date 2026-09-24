import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg

const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 2 })
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const specialRequestId = `HCN10-INTEGRATION-${suffix}`
const tokenSuffix = crypto.randomBytes(12).toString("base64url").replace(/[^A-Za-z0-9]/g, "").slice(0, 16)
const apiKey = `horusNN-${tokenSuffix.padEnd(16, "A")}`
const keyPrefix = `${apiKey.slice(0, 12)}...`
const keyHash = crypto
  .createHash("sha256")
  .update(`${apiKey}:${process.env.API_KEY_PEPPER}`)
  .digest("hex")

let result
try {
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url, country_or_region,
       use_cases, responsible_use_accepted, responsible_use_accepted_at, status,
       api_key_prefix, api_key_hash, source
     ) VALUES (
       $1, 'Gateway', 'Self Test', 'Gateway Self Test', 'TokenAI',
       'https://tokenai.llc', 'Integration test', 'developer_researcher', 20, $2,
       'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test', '', 'Internal',
       'Temporary automated gateway verification record.', true, now(), 'approved',
       $3, $4, 'gateway-self-test'
     )`,
    [specialRequestId, `gateway-${suffix.toLowerCase()}@tokenai.llc`, keyPrefix, keyHash],
  )
  await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit)
     VALUES ($1, $2, $3, 20)`,
    [specialRequestId, keyPrefix, keyHash],
  )

  const accessResponse = await fetch("http://127.0.0.1:8080/v1/access", {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  const accessBody = await accessResponse.json()

  const chatResponse = await fetch("http://127.0.0.1:8080/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-Client-Name": "NeuralNode Gateway Self Test",
    },
    body: JSON.stringify({
      model: "horus-cyper-nano-1.0-beta",
      messages: [{ role: "user", content: "Gateway storage verification message." }],
      max_tokens: 32,
      stream: false,
    }),
  })
  const chatBody = await chatResponse.json()

  const counts = await pool.query(
    `SELECT
       (SELECT count(*)::int FROM api_conversations WHERE special_request_id = $1) AS conversations,
       (SELECT count(*)::int FROM conversation_messages cm
          JOIN api_conversations ac ON ac.id = cm.conversation_id
        WHERE ac.special_request_id = $1) AS messages,
       (SELECT count(*)::int FROM api_usage_events WHERE special_request_id = $1) AS usage_events`,
    [specialRequestId],
  )

  result = {
    accessStatus: accessResponse.status,
    accessRpmLimit: accessBody.rpm_limit,
    chatStatus: chatResponse.status,
    chatErrorCode: chatBody?.error?.code,
    requestIdPresent: Boolean(chatBody.request_id),
    conversationIdPresent: Boolean(chatBody.conversation_id),
    assistantPreview: String(chatBody?.choices?.[0]?.message?.content || "").slice(0, 240),
    stored: counts.rows[0],
  }
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id = $1", [specialRequestId])
  const cleanup = await pool.query(
    "SELECT count(*)::int AS remaining FROM access_requests WHERE special_request_id = $1",
    [specialRequestId],
  )
  result = { ...(result || {}), cleanupRemaining: cleanup.rows[0].remaining }
  await pool.end()
}

console.log(JSON.stringify(result))
