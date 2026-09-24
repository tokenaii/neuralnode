import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 4 })
const apiBase = "http://127.0.0.1:18081"
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const approvedRequestId = `HCN10-SECURE-${suffix}`
const unverifiedRequestId = `HCN10-UNVERIFIED-${suffix}`
const secondRequestId = `HCN10-OWNER-${suffix}`
const fixedMessage = "Security hardening self-test response."

function hashKey(key) {
  return crypto.createHash("sha256").update(`${key}:${process.env.API_KEY_PEPPER}`).digest("hex")
}

async function insertRequest(id, email, socialStatus = "verified") {
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url, country_or_region,
       use_cases, responsible_use_accepted, responsible_use_accepted_at, status,
       social_post_status, social_post_url, source
     ) VALUES (
       $1, 'Security', 'Self Test', 'Security Self Test', 'TokenAI',
       'https://tokenai.llc', 'Integration test', 'developer_researcher', 3, $2,
       'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test', '', 'Internal',
       'Temporary automated security and access-control verification.', true, now(), 'pending',
       $3, $4, 'security-hardening-self-test'
     )`,
    [
      id,
      email,
      socialStatus,
      socialStatus === "verified" ? "https://linkedin.com/posts/tokenai-security-self-test" : null,
    ],
  )
}

async function chat(apiKey, payload = {}) {
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "horus-cyper-nano-1.0-beta",
      messages: [{ role: "user", content: "Run the security self-test." }],
      max_tokens: 16,
      ...payload,
    }),
  })
  return { status: response.status, body: await response.json() }
}

let result = {}
try {
  await insertRequest(approvedRequestId, `secure-${suffix.toLowerCase()}@tokenai.llc`)
  await insertRequest(unverifiedRequestId, `unverified-${suffix.toLowerCase()}@tokenai.llc`, "awaiting-applicant-reply")

  const unverifiedApproval = await fetch(`${apiBase}/v1/admin/access-requests/${unverifiedRequestId}/approve`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.ADMIN_TOKEN}`, "Content-Type": "application/json" },
    body: "{}",
  })

  const approval = await fetch(`${apiBase}/v1/admin/access-requests/${approvedRequestId}/approve`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.ADMIN_TOKEN}`, "Content-Type": "application/json" },
    body: JSON.stringify({ reviewedBy: "security-self-test" }),
  })
  const approvalBody = await approval.json()
  const apiKey = approvalBody.apiKey

  await pool.query(
    "UPDATE api_keys SET metadata = $2, rpm_limit = 3 WHERE special_request_id = $1",
    [approvedRequestId, { fixedResponse: fixedMessage, internalTest: true }],
  )

  const storage = await pool.query(
    `SELECT
       count(*)::int AS key_count,
       count(*) FILTER (WHERE key_ciphertext IS NOT NULL)::int AS encrypted_key_count,
       count(*) FILTER (WHERE key_hash = $2)::int AS matching_hash_count
     FROM api_keys WHERE special_request_id = $1`,
    [approvedRequestId, hashKey(apiKey)],
  )
  const rawOutbox = await pool.query(
    "SELECT count(*)::int AS count FROM email_outbox WHERE special_request_id = $1 AND body_text ~ 'horusNN-[A-Za-z0-9]{16}'",
    [approvedRequestId],
  )

  const duplicateApproval = await fetch(`${apiBase}/v1/admin/access-requests/${approvedRequestId}/approve`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.ADMIN_TOKEN}`, "Content-Type": "application/json" },
    body: "{}",
  })

  await insertRequest(secondRequestId, `owner-${suffix.toLowerCase()}@tokenai.llc`)
  await pool.query("UPDATE access_requests SET status = 'approved' WHERE special_request_id = $1", [secondRequestId])
  const secondKey = `horusNN-${crypto.randomBytes(12).toString("base64url").replace(/[^A-Za-z0-9]/g, "").slice(0, 16).padEnd(16, "A")}`
  await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit, metadata)
     VALUES ($1, $2, $3, 10, $4)`,
    [
      secondRequestId,
      `${secondKey.slice(0, 12)}...`,
      hashKey(secondKey),
      { fixedResponse: fixedMessage, clineCompletion: true },
    ],
  )
  const foreignConversationId = crypto.randomUUID()
  const ownerRequest = await chat(secondKey, { conversation_id: foreignConversationId })
  const ownerMismatch = await chat(apiKey, { conversation_id: foreignConversationId })
  const clineCompatibleRequest = await chat(secondKey, {
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "Run the Cline compatibility self-test." }],
      },
    ],
    stream: false,
    tools: [
      {
        type: "function",
        function: {
          name: "attempt_completion",
          description: "Present the final result.",
          parameters: {
            type: "object",
            properties: { result: { type: "string" } },
            required: ["result"],
          },
        },
      },
    ],
  })
  const clineXmlRequest = await chat(secondKey, {
    messages: [{ role: "user", content: "Complete the Cline XML compatibility self-test." }],
  })

  await pool.query(
    "DELETE FROM api_request_rate_events WHERE api_key_id = (SELECT id FROM api_keys WHERE special_request_id = $1)",
    [approvedRequestId],
  )
  const concurrent = await Promise.all(Array.from({ length: 20 }, () => chat(apiKey)))
  const statusCounts = concurrent.reduce((counts, item) => {
    counts[item.status] = (counts[item.status] || 0) + 1
    return counts
  }, {})

  const stored = await pool.query(
    `SELECT count(cm.id)::int AS messages
     FROM conversation_messages cm
     JOIN api_conversations ac ON ac.id = cm.conversation_id
     WHERE ac.special_request_id = $1`,
    [approvedRequestId],
  )

  result = {
    unverifiedApprovalStatus: unverifiedApproval.status,
    approvalStatus: approval.status,
    keyFormatValid: /^horusNN-[A-Za-z0-9]{16}$/.test(apiKey),
    storage: storage.rows[0],
    rawTokenRowsInOutbox: rawOutbox.rows[0].count,
    duplicateApprovalStatus: duplicateApproval.status,
    ownerRequestStatus: ownerRequest.status,
    ownerMismatchStatus: ownerMismatch.status,
    clineCompatibleRequestStatus: clineCompatibleRequest.status,
    clineCompatibleToolCallName:
      clineCompatibleRequest.body?.choices?.[0]?.message?.tool_calls?.[0]?.function?.name || null,
    clineXmlCompletionPresent: String(clineXmlRequest.body?.choices?.[0]?.message?.content || "").includes(
      "<attempt_completion>",
    ),
    concurrentStatusCounts: statusCounts,
    successfulRequestsWithinRpm: Number(statusCounts[200] || 0) <= 3,
    storedMessages: stored.rows[0].messages,
  }
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id = ANY($1)", [
    [approvedRequestId, unverifiedRequestId, secondRequestId],
  ])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = ANY($1)", [
    [approvedRequestId, unverifiedRequestId, secondRequestId],
  ])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = ANY($1)", [
    [approvedRequestId, unverifiedRequestId, secondRequestId],
  ])
  await pool.end()
}

console.log(JSON.stringify(result))
