import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 2 })
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const specialRequestId = `HCN10-RETENTION-${suffix}`
const replacementRequestId = `HCN10-REAPPLY-${suffix}`
const email = `retention-${suffix.toLowerCase()}@tokenai.llc`
const tokenSuffix = crypto.randomBytes(12).toString("base64url").replace(/[^A-Za-z0-9]/g, "").slice(0, 16)
const apiKey = `horusNN-${tokenSuffix.padEnd(16, "A")}`
const keyPrefix = `${apiKey.slice(0, 12)}...`
const keyHash = crypto.createHash("sha256").update(`${apiKey}:${process.env.API_KEY_PEPPER}`).digest("hex")
const conversationId = crypto.randomUUID()
const apiBase = "http://127.0.0.1:18081"

async function waitForApi() {
  for (let attempt = 0; attempt < 30; attempt += 1) {
    try {
      const response = await fetch(`${apiBase}/health`)
      if (response.ok) return
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 200))
  }
  throw new Error("Secondary API did not start.")
}

async function insertRequest(id) {
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url, country_or_region,
       use_cases, responsible_use_accepted, responsible_use_accepted_at, status,
       source
     ) VALUES (
       $1, 'Retention', 'Self Test', 'Retention Self Test', 'TokenAI',
       'https://tokenai.llc', 'Integration test', 'developer_researcher', 2, $2,
       'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test', '', 'Internal',
       'Temporary automated retention and rate-limit verification.', true, now(), 'approved',
       'retention-rpm-self-test'
     )`,
    [id, email],
  )
}

async function chat(message) {
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-Client-Name": "Retention RPM Self Test",
    },
    body: JSON.stringify({
      model: "horus-cyper-nano-1.0-beta",
      conversation_id: conversationId,
      messages: [{ role: "user", content: message }],
      max_tokens: 32,
      stream: false,
    }),
  })
  return { status: response.status, body: await response.json(), headers: response.headers }
}

let result = {}
try {
  await waitForApi()
  await insertRequest(specialRequestId)
  await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit)
     VALUES ($1, $2, $3, 2)`,
    [specialRequestId, keyPrefix, keyHash],
  )

  let duplicateKeyBlocked = false
  try {
    await pool.query(
      `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit)
       VALUES ($1, 'duplicate...', $2, 2)`,
      [specialRequestId, crypto.randomBytes(32).toString("hex")],
    )
  } catch (error) {
    duplicateKeyBlocked = error.code === "23505"
  }

  const accessResponse = await fetch(`${apiBase}/v1/access`, {
    headers: { Authorization: `Bearer ${apiKey}` },
  })
  const accessBody = await accessResponse.json()
  const first = await chat("First stored customer message.")
  const second = await chat("Second stored customer message.")
  const third = await chat("This request must be rate limited.")

  const beforeDelete = await pool.query(
    `SELECT
       count(*) FILTER (WHERE cm.role = 'user')::int AS user_messages,
       count(*) FILTER (WHERE cm.role = 'assistant')::int AS assistant_messages,
       count(*)::int AS total_messages
     FROM conversation_messages cm
     WHERE cm.conversation_id = $1`,
    [conversationId],
  )

  const deleteResponse = await fetch(`${apiBase}/v1/admin/users/${specialRequestId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${process.env.ADMIN_TOKEN}` },
  })
  const deleteBody = await deleteResponse.json()

  const retained = await pool.query(
    `SELECT
       (SELECT count(*)::int FROM access_requests WHERE special_request_id = $1) AS users,
       (SELECT count(*)::int FROM api_keys WHERE special_request_id = $1) AS keys,
       (SELECT count(*)::int FROM api_conversations WHERE special_request_id = $1) AS conversations,
       (SELECT count(*)::int FROM conversation_messages WHERE conversation_id = $2) AS messages`,
    [specialRequestId, conversationId],
  )

  const dashboardResponse = await fetch(
    `${apiBase}/v1/admin/conversations?specialRequestId=${encodeURIComponent(specialRequestId)}`,
    { headers: { Authorization: `Bearer ${process.env.ADMIN_TOKEN}` } },
  )
  const dashboardBody = await dashboardResponse.json()

  await insertRequest(replacementRequestId)
  const emailReused = await pool.query(
    "SELECT count(*)::int AS count FROM access_requests WHERE lower(business_mail) = lower($1)",
    [email],
  )

  result = {
    access: { status: accessResponse.status, rpmLimit: accessBody.rpm_limit },
    chatStatuses: [first.status, second.status, third.status],
    rateHeaders: {
      limit: third.headers.get("x-ratelimit-limit"),
      remaining: third.headers.get("x-ratelimit-remaining"),
      retryAfterPresent: Boolean(third.headers.get("retry-after")),
    },
    duplicateKeyBlocked,
    storedBeforeDelete: beforeDelete.rows[0],
    delete: { status: deleteResponse.status, ...deleteBody },
    retainedAfterDelete: retained.rows[0],
    dashboardAfterDelete: {
      status: dashboardResponse.status,
      count: dashboardBody.conversations?.length || 0,
      ownerDeleted: dashboardBody.conversations?.[0]?.owner_deleted,
      displayedName: dashboardBody.conversations?.[0]?.full_name,
    },
    emailReusable: emailReused.rows[0].count === 1,
  }
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id IN ($1, $2)", [
    specialRequestId,
    replacementRequestId,
  ])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = $1", [specialRequestId])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = $1", [specialRequestId])
  await pool.end()
}

console.log(JSON.stringify(result))
