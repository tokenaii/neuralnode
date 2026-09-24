import crypto from "node:crypto"
import pg from "pg"

const { Pool } = pg
const pool = new Pool({ connectionString: process.env.DATABASE_URL, max: 2 })
const suffix = crypto.randomBytes(5).toString("hex").toUpperCase()
const specialRequestId = `HCN10-AGENT-${suffix}`
const apiKey = `horusNN-${crypto.randomBytes(12).toString("hex").slice(0, 16)}`
const keyPrefix = `${apiKey.slice(0, 12)}...`
const keyHash = crypto.createHash("sha256").update(`${apiKey}:${process.env.API_KEY_PEPPER}`).digest("hex")
const conversationId = crypto.randomUUID()
const apiBase = "http://127.0.0.1:18081"

async function request(payload) {
  return fetch(`${apiBase}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "X-Client-Name": "Agent Mode Self Test",
    },
    body: JSON.stringify({ model: "horus-cyper-nano-1.0-beta", max_tokens: 64, ...payload }),
  })
}

let result = {}
try {
  await pool.query(
    `INSERT INTO access_requests (
       special_request_id, first_name, second_name, full_name, company_name,
       company_website, job, applicant_type, rpm_limit, business_mail,
       linkedin_profile_url, github_profile_url, portfolio_url, country_or_region,
       use_cases, responsible_use_accepted, responsible_use_accepted_at, status, source
     ) VALUES (
       $1, 'Agent', 'Self Test', 'Agent Self Test', 'TokenAI',
       'https://tokenai.llc', 'Integration test', 'developer_researcher', 10, $2,
       'https://linkedin.com/in/tokenai-test', 'https://github.com/tokenai-test', '', 'Internal',
       'Temporary automated Agent Mode verification.', true, now(), 'approved', 'agent-mode-self-test'
     )`,
    [specialRequestId, `agent-${suffix.toLowerCase()}@tokenai.llc`],
  )
  await pool.query(
    `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, rpm_limit)
     VALUES ($1, $2, $3, 10)`,
    [specialRequestId, keyPrefix, keyHash],
  )

  const firstResponse = await request({
    conversation_id: conversationId,
    messages: [{ role: "user", content: "Check the gateway status using the available tool." }],
    tools: [{
      type: "function",
      function: {
        name: "get_status",
        description: "Read a service status.",
        parameters: {
          type: "object",
          properties: { target: { type: "string" } },
          required: ["target"],
          additionalProperties: false,
        },
      },
    }],
    tool_choice: "required",
  })
  const firstBody = await firstResponse.json()
  const toolCall = firstBody.choices?.[0]?.message?.tool_calls?.[0]

  if (!toolCall) {
    throw new Error(`The model did not return a required tool call: ${JSON.stringify(firstBody).slice(0, 1000)}`)
  }

  const secondResponse = await request({
    conversation_id: conversationId,
    messages: [
      { role: "user", content: "Check the gateway status using the available tool." },
      { role: "assistant", content: null, tool_calls: [toolCall] },
      { role: "tool", tool_call_id: toolCall.id, content: '{"status":"healthy"}' },
    ],
  })
  const secondBody = await secondResponse.json()

  const streamResponse = await request({
    messages: [{ role: "user", content: "Return a short compatibility response." }],
    stream: true,
  })
  const streamBody = await streamResponse.text()

  const stored = await pool.query(
    `SELECT
       count(*)::int AS total_messages,
       count(*) FILTER (WHERE role = 'tool')::int AS tool_messages,
       count(*) FILTER (WHERE cm.role = 'assistant' AND cm.metadata ? 'toolCalls')::int AS tool_call_messages
     FROM conversation_messages cm
     JOIN api_conversations ac ON ac.id = cm.conversation_id
     WHERE ac.special_request_id = $1`,
    [specialRequestId],
  )

  result = {
    toolRequestStatus: firstResponse.status,
    toolCallName: toolCall?.function?.name,
    toolResultStatus: secondResponse.status,
    finalContent: secondBody.choices?.[0]?.message?.content,
    streamStatus: streamResponse.status,
    streamContentType: streamResponse.headers.get("content-type"),
    streamDone: streamBody.includes("data: [DONE]"),
    stored: stored.rows[0],
  }
} finally {
  await pool.query("DELETE FROM access_requests WHERE special_request_id = $1", [specialRequestId])
  await pool.query("DELETE FROM api_conversations WHERE special_request_id = $1", [specialRequestId])
  await pool.query("DELETE FROM api_usage_events WHERE special_request_id = $1", [specialRequestId])
  await pool.end()
}

console.log(JSON.stringify(result))
