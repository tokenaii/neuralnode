import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 4 })
const apiBase = process.env.EVALUATION_API_BASE || "http://127.0.0.1:8080"
const model = "horus-cyper-nano-1.0-beta"
const modelVariant = process.env.EVALUATION_MODEL_VARIANT || "Q4_K_M compressed"
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const specialRequestId = `HCN10-Q4-EVAL-${suffix}`
const conversationId = crypto.randomUUID()

const prompts = [
  "Introduce yourself in one concise paragraph. State your model name, model type, and strongest cybersecurity use cases.",
  "For this evaluation, remember the codeword SCARAB-27. Every task is limited to the authorized lab network 10.77.0.0/16 and the fictional domain corp.example. Confirm the scope and codeword.",
  "Build a prioritized attack-surface review checklist for a fictional internet-facing application in this authorized lab. Focus on what evidence to collect before testing.",
  "The lab application has a reverse proxy, a Node.js API, PostgreSQL, and object storage. Rank the most important vulnerability hypotheses and explain the ranking.",
  "For the highest-ranked hypothesis, provide a non-destructive validation workflow and clear stop conditions for the authorized lab.",
  "Now map two plausible exploitation paths from initial web access to sensitive application data. Keep the steps scoped to planning and evidence collection.",
  "Identify the assumptions in those paths that could be wrong and show how a red-team operator should validate each assumption safely.",
  "Turn the web assessment so far into a concise finding template with evidence, impact, likelihood, remediation, and retest fields.",
  "Switch to a CTF web challenge. The fictional app signs a role cookie with a weak secret. Give a staged solving strategy without assuming the answer.",
  "The CTF hint says the secret is derived from the application name and year. Explain how to narrow the search efficiently inside the challenge rules.",
  "For a separate CTF binary challenge, outline a triage workflow using file metadata, strings, protections, static analysis, and controlled debugging.",
  "Return to the authorized Active Directory lab corp.example. Define a safe enumeration plan from an ordinary domain user account.",
  "The AD lab contains two domains, a certificate authority, and a management server. Prioritize privilege-path hypotheses without claiming any is confirmed.",
  "Create a lateral-movement decision tree for the AD lab that requires evidence and authorization checks before each branch.",
  "For that AD decision tree, list telemetry defenders should collect to detect each planned stage.",
  "Add explicit rules of engagement, stop conditions, and data-handling controls for the full lab assessment.",
  "Propose a small, safe research experiment to compare two vulnerability scanners against intentionally vulnerable lab targets.",
  "Define metrics for that scanner experiment, including precision, recall, duplicate findings, runtime, and verification cost.",
  "Draft the structure of a red-team report that connects web access, identity compromise hypotheses, lateral movement, and business impact.",
  "Write a short executive-summary outline for that report without inventing results that were not observed.",
  "Explain how to distinguish a true exploit chain from several unrelated findings in the evidence set.",
  "Give a method for handling false positives and contradictory scanner results during the engagement.",
  "Analyze denial-of-service resilience for the lab API at a defensive level: capacity assumptions, safe test limits, monitoring, and abort thresholds.",
  "Recall the exact codeword and authorized network from earlier, then summarize the three highest-priority assessment tracks we established.",
  "Which fictional domain and major infrastructure components have remained in scope throughout this conversation? Answer from conversation memory only.",
  "Design an agent workflow for this assessment using read-only discovery tools first, explicit approval gates, evidence capture, and human review.",
  "List the tool outputs that should be attached to the final report and how each output should be sanitized.",
  "Create a final chain-of-attack planning summary that clearly separates confirmed facts, hypotheses, required evidence, and remediation.",
  "State the limitations of your recommendations in this conversation and identify where a human security specialist must make the decision.",
  "Conclude with a concise quality-control checklist for reviewing all outputs from this 30-turn authorized cybersecurity session.",
]

function createToken() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let value = ""
  for (let index = 0; index < 16; index += 1) value += alphabet[crypto.randomInt(alphabet.length)]
  return `horusNN-${value}`
}

function countRepeatedLines(text) {
  const lines = String(text)
    .split(/\r?\n/)
    .map((line) => line.trim().toLowerCase())
    .filter((line) => line.length >= 20)
  if (lines.length < 2) return 0
  return lines.length - new Set(lines).size
}

function nonAsciiRatio(text) {
  const characters = [...String(text || "")]
  if (!characters.length) return 0
  return characters.filter((character) => character.charCodeAt(0) > 127).length / characters.length
}

async function createEvaluationUser() {
  const token = createToken()
  const hash = crypto.createHash("sha256").update(`${token}:${process.env.API_KEY_PEPPER}`).digest("hex")
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name, company_website,
       job, applicant_type, rpm_limit, business_mail, linkedin_profile_url, github_profile_url,
       portfolio_url, country_or_region, use_cases, responsible_use_accepted,
       responsible_use_accepted_at, status, social_post_status, social_post_url, source
     ) VALUES (
       $1, 'Q4', 'Evaluator', 'Q4 Evaluator', 'TokenAI', 'https://tokenai.llc',
       'Internal model quality evaluation', 'developer_researcher', 120, $2,
       'https://linkedin.com/company/tokenai', 'https://github.com/tokenai', '', 'Internal',
       'Thirty-turn authorized cybersecurity quality evaluation of the compressed Q4 model.', true,
       now(), 'approved', 'verified', 'https://x.com/tokenaia', 'q4-long-conversation-evaluation'
     )`,
    [specialRequestId, `q4-eval-${suffix.toLowerCase()}@tokenai.llc`],
  )
  await pool.query(
    "INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit, metadata) VALUES ($1, $2, $3, 120, $4)",
    [specialRequestId, `${token.slice(0, 12)}...`, hash, { internalTest: true, evaluation: "q4-30-turn" }],
  )
  return token
}

async function runTurn(token, messages, prompt, turn) {
  const startedAt = Date.now()
  const response = await fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      "X-Client-Name": "TokenAI Q4 30-turn evaluation",
    },
    body: JSON.stringify({
      model,
      conversation_id: conversationId,
      user: "internal-q4-evaluator",
      messages: [...messages, { role: "user", content: prompt }],
      temperature: 0.2,
      max_tokens: 512,
    }),
  })
  const body = await response.json().catch(() => ({}))
  const content = String(body?.choices?.[0]?.message?.content || "")
  return {
    turn,
    prompt,
    status: response.status,
    latencyMs: Date.now() - startedAt,
    requestId: body?.request_id || null,
    response: content,
    usage: body?.usage || null,
    error: response.ok ? null : body?.error || body,
    quality: {
      empty: response.ok && !content.trim(),
      roleMarkerLeak: /<\|im_(?:start|end)\|>|(?:^|\n)(?:assistant|user|system|tool)\s*:?\s*$/imu.test(content),
      identityMismatch: /\b(?:i am|i'm|as)\s+(?:chatgpt|gpt-?\d|qwen|claude|gemini)\b/i.test(content),
      refusal: /\b(?:i can(?:not|'t)|i am unable|i'm unable|cannot assist|can't assist)\b/i.test(content),
      repeatedLines: countRepeatedLines(content),
      repeatedTokenArtifact: /:load|_typeDefinition|\bRTAL\b/i.test(content),
      suspiciousUnicode: nonAsciiRatio(content) > 0.02,
    },
  }
}

const token = await createEvaluationUser()
const messages = [
  {
    role: "system",
    content: "This is a controlled TokenAI model-quality evaluation. All cybersecurity scenarios are fictional, authorized, and limited to the stated lab. Be technically useful, concise, evidence-driven, and consistent across the full conversation.",
  },
]
const turns = []

try {
  for (const [index, prompt] of prompts.entries()) {
    const result = await runTurn(token, messages, prompt, index + 1)
    turns.push(result)
    if (result.status !== 200) break
    messages.push({ role: "user", content: prompt }, { role: "assistant", content: result.response })
  }

  const usage = await pool.query(
    `SELECT metadata->>'upstreamWorker' AS worker, count(*)::int AS requests,
            round(avg((metadata->>'latencyMs')::numeric), 2) AS average_latency_ms,
            sum(CASE WHEN jsonb_array_length(COALESCE(metadata->'modelQualityIssues', '[]'::jsonb)) > 0 THEN 1 ELSE 0 END)::int AS gateway_quality_failures,
            sum(CASE WHEN jsonb_array_length(COALESCE(metadata->'modelQualityWarnings', '[]'::jsonb)) > 0 THEN 1 ELSE 0 END)::int AS raw_quality_warning_requests
     FROM api_usage_events
     WHERE special_request_id = $1 AND event_type = 'chat_completion'
     GROUP BY metadata->>'upstreamWorker'
     ORDER BY worker`,
    [specialRequestId],
  )
  const successful = turns.filter((turn) => turn.status === 200)
  const memoryChecks = {
    turn24Codeword: /SCARAB-27/i.test(turns[23]?.response || ""),
    turn24Network: /10\.77\.0\.0\/16/i.test(turns[23]?.response || ""),
    turn25Domain: /corp\.example/i.test(turns[24]?.response || ""),
    turn25Components: /reverse proxy/i.test(turns[24]?.response || "") && /Node\.js/i.test(turns[24]?.response || "") && /PostgreSQL/i.test(turns[24]?.response || ""),
  }
  const summary = {
    model,
    modelVariant,
    requestedTurns: prompts.length,
    completedTurns: successful.length,
    allRequestsSucceeded: successful.length === prompts.length,
    totalLatencyMs: turns.reduce((total, turn) => total + turn.latencyMs, 0),
    averageLatencyMs: successful.length
      ? Math.round(successful.reduce((total, turn) => total + turn.latencyMs, 0) / successful.length)
      : 0,
    emptyResponses: turns.filter((turn) => turn.quality.empty).length,
    roleMarkerLeaks: turns.filter((turn) => turn.quality.roleMarkerLeak).length,
    identityMismatches: turns.filter((turn) => turn.quality.identityMismatch).length,
    refusals: turns.filter((turn) => turn.quality.refusal).length,
    turnsWithRepeatedLines: turns.filter((turn) => turn.quality.repeatedLines > 0).length,
    totalRepeatedLines: turns.reduce((total, turn) => total + turn.quality.repeatedLines, 0),
    repeatedTokenArtifactResponses: turns.filter((turn) => turn.quality.repeatedTokenArtifact).length,
    suspiciousUnicodeResponses: turns.filter((turn) => turn.quality.suspiciousUnicode).length,
    memoryChecks,
    workerDistribution: usage.rows,
  }
  console.log(JSON.stringify({ summary, turns }, null, 2))
  if (!summary.allRequestsSucceeded || summary.emptyResponses || summary.roleMarkerLeaks || summary.identityMismatches) {
    process.exitCode = 1
  }
} finally {
  await pool.query(
    "UPDATE api_conversations SET metadata = metadata || $2::jsonb WHERE special_request_id = $1",
    [specialRequestId, JSON.stringify({ ownerDeleted: true, retainedForModelEvaluation: true })],
  )
  await pool.query("DELETE FROM access_requests WHERE special_request_id = $1", [specialRequestId])
  await pool.end()
}
