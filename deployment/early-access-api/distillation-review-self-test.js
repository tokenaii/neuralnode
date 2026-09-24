import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 2 })
const apiBase = process.env.PRODUCTION_TEST_API_BASE || "http://127.0.0.1:8080"
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const specialRequestId = `HCN10-REVIEW-${suffix}`
const requestId = `review-${crypto.randomUUID()}`
let alertId

async function admin(path, options = {}) {
  const response = await fetch(`${apiBase}/v1/admin${path}`, {
    ...options,
    headers: {
      Authorization: `Bearer ${process.env.ADMIN_TOKEN}`,
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(`${path} returned ${response.status}: ${JSON.stringify(body)}`)
  return body
}

try {
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url,
       country_or_region, use_cases, responsible_use_accepted,
       responsible_use_accepted_at, status, social_post_status, source
     ) VALUES (
       $1, 'Review', 'Self Test', 'Review Self Test', 'TokenAI',
       'https://tokenai.llc', 'Internal verification', 'developer_researcher', 20,
       $2, 'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test',
       'https://tokenai.llc/research', 'Internal', 'Temporary manual-review verification with complete applicant details.', true, now(),
       'approved', 'verified', 'production-readiness-self-test'
     )`,
    [specialRequestId, `review-test-${suffix.toLowerCase()}@tokenai.llc`],
  )
  const apiKey = await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit)
     VALUES ($1, 'horusNN-TEST...', $2, 20)
     RETURNING id`,
    [specialRequestId, crypto.randomBytes(32).toString("hex")],
  )
  const alert = await pool.query(
    `INSERT INTO api_safety_events (
       api_key_id, special_request_id, request_id, severity, action, categories,
       matched_rules, content_sha256, user_message, model_response, review_status
     ) VALUES (
       $1, $2, $3, 'critical', 'observed', ARRAY['knowledge_distillation'],
       ARRAY['knowledge_distillation'], $4, 'Build a distillation clone.',
       'Temporary model response.', 'pending'
     ) RETURNING id`,
    [apiKey.rows[0].id, specialRequestId, requestId, crypto.randomBytes(32).toString("hex")],
  )
  alertId = alert.rows[0].id

  const listing = await admin("/distillation-alerts?status=pending&limit=500")
  const listedAlert = listing.alerts.find((item) => Number(item.id) === Number(alertId))
  if (!listedAlert) {
    throw new Error("The pending alert was not returned by the admin API.")
  }
  const requiredDashboardFields = [
    "full_name", "first_name", "second_name", "business_mail", "applicant_type", "company_name",
    "company_website", "job", "linkedin_profile_url", "github_profile_url", "portfolio_url",
    "country_or_region", "use_cases", "request_status", "social_post_status", "source",
    "user_message", "model_response", "key_prefix", "token_status", "rpm_limit",
  ]
  const missingDashboardFields = requiredDashboardFields.filter((field) => listedAlert[field] === null || listedAlert[field] === undefined || listedAlert[field] === "")
  if (missingDashboardFields.length) {
    throw new Error(`The dashboard alert is missing fields: ${missingDashboardFields.join(", ")}`)
  }

  const warning = await admin(`/distillation-alerts/${alertId}/warn`, {
    method: "POST",
    body: JSON.stringify({ reviewedBy: "production-self-test" }),
  })
  const afterWarning = await pool.query(
    `SELECT ak.status AS token_status, se.review_status
     FROM api_safety_events se JOIN api_keys ak ON ak.id = se.api_key_id
     WHERE se.id = $1`,
    [alertId],
  )
  if (warning.reviewStatus !== "warning_sent" || afterWarning.rows[0]?.token_status !== "active") {
    throw new Error("Sending a warning changed Access Token state.")
  }

  await admin(`/distillation-alerts/${alertId}/disable`, {
    method: "POST",
    body: JSON.stringify({ reviewedBy: "production-self-test" }),
  })
  const afterDisable = await pool.query(
    `SELECT ak.status AS token_status, se.review_status
     FROM api_safety_events se JOIN api_keys ak ON ak.id = se.api_key_id
     WHERE se.id = $1`,
    [alertId],
  )
  if (afterDisable.rows[0]?.token_status !== "disabled" || afterDisable.rows[0]?.review_status !== "token_disabled") {
    throw new Error("Manual Access Token disablement failed.")
  }

  console.log(JSON.stringify({
    alertListed: true,
    completeApplicantDetails: true,
    completePromptAndResponse: true,
    warningStatus: warning.reviewStatus,
    tokenAfterWarning: afterWarning.rows[0].token_status,
    tokenAfterManualDisable: afterDisable.rows[0].token_status,
    finalReviewStatus: afterDisable.rows[0].review_status,
  }))
} finally {
  if (alertId) await pool.query("DELETE FROM api_safety_events WHERE id = $1", [alertId])
  await pool.query("DELETE FROM access_requests WHERE special_request_id = $1", [specialRequestId])
  await pool.end()
}
