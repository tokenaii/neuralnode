import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 6 })
const apiBase = process.env.SOAK_API_BASE || "http://127.0.0.1:8080"
const durationSeconds = Math.max(60, Number(process.env.SOAK_DURATION_SECONDS || 1800))
const requestIntervalMs = Math.max(1000, Number(process.env.SOAK_REQUEST_INTERVAL_MS || 10000))
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const ids = [`HCN10-SOAK-${suffix}-1`, `HCN10-SOAK-${suffix}-2`]

function createToken() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let value = ""
  for (let index = 0; index < 16; index += 1) value += alphabet[crypto.randomInt(alphabet.length)]
  return `horusNN-${value}`
}

function percentile(values, ratio) {
  if (!values.length) return 0
  const sorted = [...values].sort((left, right) => left - right)
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * ratio))]
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
       $1, 'Soak', 'Test', 'Soak Test', 'TokenAI', 'https://tokenai.llc', 'Reliability test',
       'developer_researcher', 60, $2, 'https://linkedin.com/in/tokenai-test',
       'https://github.com/tokenai-test', '', 'Internal', 'Production soak verification.', true,
       now(), 'approved', 'verified', 'https://linkedin.com/posts/tokenai-soak-test', 'production-soak-test'
     )`,
    [id, `soak-${suffix.toLowerCase()}-${index}@tokenai.llc`],
  )
  await pool.query(
    "INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit, metadata) VALUES ($1, $2, $3, 60, $4)",
    [id, `${token.slice(0, 12)}...`, hash, { internalTest: true }],
  )
  return token
}

async function request(token, sequence) {
  const started = Date.now()
  try {
    const response = await fetch(`${apiBase}/v1/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "horus-cyper-nano-1.0-beta",
        messages: [{ role: "user", content: `Reply with READY for reliability check ${sequence}.` }],
        max_tokens: 16,
        temperature: 0,
      }),
    })
    const body = await response.json()
    const content = String(body?.choices?.[0]?.message?.content || "")
    return {
      status: response.status,
      latencyMs: Date.now() - started,
      empty: response.status === 200 && !content,
      leakedRole: /<\|im_(?:start|end)\|>|(?:^|\n)(?:assistant|user|system|tool)\s*:?\s*$/imu.test(content),
    }
  } catch (error) {
    return { status: 0, latencyMs: Date.now() - started, error: String(error) }
  }
}

const tokens = []
const results = []
const startedAt = Date.now()
try {
  for (const [index, id] of ids.entries()) tokens.push(await createUser(id, index))
  let sequence = 0
  while (Date.now() - startedAt < durationSeconds * 1000) {
    const batch = await Promise.all(tokens.map((token) => request(token, ++sequence)))
    results.push(...batch)
    const remaining = requestIntervalMs - Math.max(...batch.map((item) => item.latencyMs))
    if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining))
  }

  const workers = await pool.query(
    `SELECT metadata->>'upstreamWorker' AS worker, count(*)::int AS count
     FROM api_usage_events WHERE special_request_id = ANY($1) AND event_type = 'chat_completion'
     GROUP BY metadata->>'upstreamWorker' ORDER BY worker`,
    [ids],
  )
  const statuses = results.reduce((counts, item) => {
    counts[item.status] = (counts[item.status] || 0) + 1
    return counts
  }, {})
  const latencies = results.filter((item) => item.status === 200).map((item) => item.latencyMs)
  const summary = {
    durationSeconds: Math.round((Date.now() - startedAt) / 1000),
    requests: results.length,
    statuses,
    successRate: results.length ? Number(((statuses[200] || 0) / results.length).toFixed(5)) : 0,
    p50LatencyMs: percentile(latencies, 0.5),
    p95LatencyMs: percentile(latencies, 0.95),
    p99LatencyMs: percentile(latencies, 0.99),
    emptyResponses: results.filter((item) => item.empty).length,
    leakedRoleResponses: results.filter((item) => item.leakedRole).length,
    workerDistribution: workers.rows,
  }
  console.log(JSON.stringify(summary))
  if ((statuses[200] || 0) !== results.length || summary.emptyResponses || summary.leakedRoleResponses) process.exitCode = 1
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = ANY($1)", [ids])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = ANY($1)", [ids])
  await pool.end()
}
