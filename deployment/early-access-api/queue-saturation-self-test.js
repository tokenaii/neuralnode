import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 12 })
const apiBase = process.env.PRODUCTION_TEST_API_BASE || "http://127.0.0.1:8080"
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const requestCount = Number(process.env.QUEUE_TEST_REQUESTS || 45)
const ids = Array.from({ length: requestCount }, (_, index) => `HCN10-QUEUE-${suffix}-${index + 1}`)

function createToken() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let value = ""
  for (let index = 0; index < 16; index += 1) value += alphabet[crypto.randomInt(alphabet.length)]
  return `horusNN-${value}`
}

async function createUser(id, index) {
  const token = createToken()
  const hash = crypto.createHash("sha256").update(`${token}:${process.env.API_KEY_PEPPER}`).digest("hex")
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name, company_website,
       job, applicant_type, rpm_limit, business_mail, linkedin_profile_url, github_profile_url,
       portfolio_url, country_or_region, use_cases, responsible_use_accepted,
       responsible_use_accepted_at, status, social_post_status, social_post_url, source
     ) VALUES (
       $1, 'Queue', 'Test', 'Queue Test', 'TokenAI', 'https://tokenai.llc', 'Capacity test',
       'developer_researcher', 60, $2, 'https://linkedin.com/in/tokenai-test',
       'https://github.com/tokenai-test', '', 'Internal', 'Bounded queue verification.', true,
       now(), 'approved', 'verified', 'https://linkedin.com/posts/tokenai-queue-test', 'queue-saturation-self-test'
     )`,
    [id, `queue-${suffix.toLowerCase()}-${index}@tokenai.llc`],
  )
  await pool.query(
    "INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit, metadata) VALUES ($1, $2, $3, 60, $4)",
    [id, `${token.slice(0, 12)}...`, hash, { internalTest: true }],
  )
  return token
}

async function chat(token, index) {
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "horus-cyper-nano-1.0-beta",
      messages: [{ role: "user", content: `Write a short defensive checklist for software review. Queue test ${index}.` }],
      max_tokens: 128,
      temperature: 0,
    }),
  })
  const body = await response.json()
  return { status: response.status, code: body?.error?.code || null }
}

let maximumRunning = 0
let maximumWaiting = 0
let polling = true
try {
  const tokens = []
  for (const [index, id] of ids.entries()) tokens.push(await createUser(id, index))
  const poll = (async () => {
    while (polling) {
      const queue = await pool.query(
        `SELECT count(*) FILTER (WHERE status = 'running')::int AS running,
                count(*) FILTER (WHERE status = 'waiting')::int AS waiting
         FROM inference_queue_leases WHERE expires_at > now()`,
      )
      maximumRunning = Math.max(maximumRunning, Number(queue.rows[0].running || 0))
      maximumWaiting = Math.max(maximumWaiting, Number(queue.rows[0].waiting || 0))
      await new Promise((resolve) => setTimeout(resolve, 25))
    }
  })()

  const responses = await Promise.all(tokens.map((token, index) => chat(token, index + 1)))
  polling = false
  await poll
  const statuses = responses.reduce((counts, item) => {
    const key = `${item.status}:${item.code || "none"}`
    counts[key] = (counts[key] || 0) + 1
    return counts
  }, {})
  const result = { requestCount, statuses, maximumRunning, maximumWaiting }
  console.log(JSON.stringify(result))

  const queueFull = responses.filter((item) => item.status === 503 && item.code === "inference_queue_full").length
  const unexpected = responses.filter((item) => item.status !== 200 && !(item.status === 503 && item.code === "inference_queue_full"))
  if (maximumRunning > 2 || maximumWaiting > 40 || queueFull < 1 || unexpected.length) process.exitCode = 1
} finally {
  polling = false
  await pool.query("DELETE FROM access_requests WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = ANY($1)", [ids])
  await pool.end()
}
