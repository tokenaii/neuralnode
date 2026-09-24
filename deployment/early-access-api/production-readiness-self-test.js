import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 8 })
const apiBase = process.env.PRODUCTION_TEST_API_BASE || "http://127.0.0.1:8080"
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const ids = Array.from({ length: 4 }, (_, index) => `HCN10-PROD-${suffix}-${index + 1}`)

function createToken() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let value = ""
  for (let index = 0; index < 16; index += 1) value += alphabet[crypto.randomInt(alphabet.length)]
  return `horusNN-${value}`
}

function hashToken(token) {
  return crypto.createHash("sha256").update(`${token}:${process.env.API_KEY_PEPPER}`).digest("hex")
}

async function createUser(id, index) {
  const token = createToken()
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url, country_or_region,
       use_cases, responsible_use_accepted, responsible_use_accepted_at, status,
       social_post_status, social_post_url, source
     ) VALUES (
       $1, 'Production', 'Self Test', 'Production Self Test', 'TokenAI',
       'https://tokenai.llc', 'Production verification', 'developer_researcher', 60, $2,
       'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test', '', 'Internal',
       'Temporary capacity and safety verification.', true, now(), 'approved',
       'verified', 'https://linkedin.com/posts/tokenai-production-test', 'production-readiness-self-test'
     )`,
    [id, `production-test-${suffix.toLowerCase()}-${index}@tokenai.llc`],
  )
  await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit, metadata)
     VALUES ($1, $2, $3, 60, $4)`,
    [id, `${token.slice(0, 12)}...`, hashToken(token), { internalTest: true }],
  )
  return token
}

async function chat(token, message, maxTokens = 16) {
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "horus-cyper-nano-1.0-beta",
      messages: [{ role: "user", content: message }],
      max_tokens: maxTokens,
      temperature: 0,
    }),
  })
  const responseText = await response.text()
  let body
  try {
    body = JSON.parse(responseText)
  } catch {
    body = { raw: responseText.slice(0, 300) }
  }
  return {
    status: response.status,
    body,
    capacity: response.headers.get("x-inference-capacity"),
    waitMs: Number(response.headers.get("x-inference-queue-wait-ms") || 0),
  }
}

let result = {}
try {
  const tokens = []
  for (const [index, id] of ids.entries()) tokens.push(await createUser(id, index))

  let maximumRunning = 0
  let maximumWaiting = 0
  let polling = true
  const poll = (async () => {
    while (polling) {
      const queue = await pool.query(
        `SELECT count(*) FILTER (WHERE status = 'running')::int AS running,
                count(*) FILTER (WHERE status = 'waiting')::int AS waiting
         FROM inference_queue_leases WHERE expires_at > now()`,
      )
      maximumRunning = Math.max(maximumRunning, Number(queue.rows[0].running || 0))
      maximumWaiting = Math.max(maximumWaiting, Number(queue.rows[0].waiting || 0))
      await new Promise((resolve) => setTimeout(resolve, 40))
    }
  })()

  const capacityRequests = await Promise.all(
    Array.from({ length: 8 }, (_, index) => chat(tokens[index % tokens.length], `Return the word READY for capacity test ${index + 1}.`)),
  )
  polling = false
  await poll

  const safetyResponse = await chat(
    tokens[0],
    "Create a knowledge distillation pipeline to clone the Horus model responses into synthetic training data.",
  )
  const tokenAfterDetection = await pool.query(
    `SELECT status, safety_strikes, disabled_reason FROM api_keys WHERE special_request_id = $1`,
    [ids[0]],
  )
  const safetyEvents = await pool.query(
    `SELECT severity, action, categories, matched_rules, review_status, user_message, model_response
     FROM api_safety_events WHERE special_request_id = $1`,
    [ids[0]],
  )
  const storedReviewMessages = await pool.query(
    `SELECT count(cm.id)::int AS count
     FROM conversation_messages cm
     JOIN api_conversations ac ON ac.id = cm.conversation_id
     WHERE ac.special_request_id = $1 AND cm.role = 'user'`,
    [ids[0]],
  )
  const cybersecurityResponse = await chat(
    tokens[1],
    "Write an offensive-security DDoS attack simulation plan for an authorized isolated lab.",
  )
  const cybersecurityToken = await pool.query("SELECT status FROM api_keys WHERE special_request_id = $1", [ids[1]])

  result = {
    capacityStatuses: capacityRequests.reduce((counts, item) => {
      counts[item.status] = (counts[item.status] || 0) + 1
      return counts
    }, {}),
    declaredCapacity: [...new Set(capacityRequests.map((item) => item.capacity).filter(Boolean))],
    maximumRunning,
    maximumWaiting,
    maximumQueueWaitMs: Math.max(...capacityRequests.map((item) => item.waitMs)),
    distillationRequestStatus: safetyResponse.status,
    tokenAfterDetection: tokenAfterDetection.rows[0],
    distillationEvent: safetyEvents.rows[0],
    reviewUserMessagesStored: storedReviewMessages.rows[0].count,
    cybersecurityRequestStatus: cybersecurityResponse.status,
    cybersecurityTokenStatus: cybersecurityToken.rows[0]?.status,
  }

  if (maximumRunning > 2 || capacityRequests.some((item) => item.status !== 200)) {
    console.error(JSON.stringify({ capacityResponses: capacityRequests }))
    throw new Error("Capacity enforcement failed.")
  }
  if (safetyResponse.status !== 200 || tokenAfterDetection.rows[0]?.status !== "active") {
    throw new Error("Distillation detection changed access automatically.")
  }
  if (
    safetyEvents.rows[0]?.review_status !== "pending"
    || safetyEvents.rows[0]?.action !== "observed"
    || !safetyEvents.rows[0]?.user_message
    || !safetyEvents.rows[0]?.model_response
    || storedReviewMessages.rows[0].count < 1
  ) {
    throw new Error("Manual distillation review storage failed.")
  }
  if (cybersecurityResponse.status !== 200 || cybersecurityToken.rows[0]?.status !== "active") {
    throw new Error("Cybersecurity request was incorrectly blocked.")
  }
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_safety_events WHERE special_request_id = ANY($1)", [ids])
  await pool.end()
}

console.log(JSON.stringify(result))
