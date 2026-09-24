import crypto from "node:crypto"
import { z } from "zod"

const publicModelId = "horus-cyper-nano-1.0-beta"
const acceptedModelIds = new Set([publicModelId, "Horus Cyper Nano 1.0 BETA"])

const toolCallSchema = z.object({
  id: z.string().trim().min(1).max(200),
  type: z.literal("function").default("function"),
  function: z.object({
    name: z.string().trim().min(1).max(128),
    arguments: z.string().max(100000),
  }),
})

const messageContentPartSchema = z
  .object({
    type: z.enum(["text", "input_text"]),
    text: z.string().max(100000),
  })
  .passthrough()

const messageContentSchema = z
  .union([z.string(), z.array(messageContentPartSchema).max(100)])
  .transform((content) =>
    Array.isArray(content) ? content.map((part) => part.text).filter(Boolean).join("\n") : content,
  )
  .refine((content) => content.length <= 100000, "Message content must not exceed 100000 characters.")

const chatMessageSchema = z
  .object({
    role: z.enum(["system", "user", "assistant", "tool"]),
    content: messageContentSchema.nullable().optional(),
    name: z.string().trim().min(1).max(128).optional(),
    tool_call_id: z.string().trim().min(1).max(200).optional(),
    tool_calls: z.array(toolCallSchema).max(64).optional(),
  })
  .superRefine((message, context) => {
    if (!message.content && !message.tool_calls?.length) {
      context.addIssue({ code: "custom", message: "A message must contain content or tool calls." })
    }
    if (message.role === "tool" && !message.tool_call_id) {
      context.addIssue({ code: "custom", message: "Tool messages require tool_call_id." })
    }
  })

const functionToolSchema = z.object({
  type: z.literal("function"),
  function: z.object({
    name: z.string().trim().min(1).max(128),
    description: z.string().max(4000).optional(),
    parameters: z.record(z.string(), z.unknown()),
  }),
})

const toolChoiceSchema = z.union([
  z.enum(["none", "auto", "required"]),
  z.object({
    type: z.literal("function"),
    function: z.object({ name: z.string().trim().min(1).max(128) }),
  }),
])

const chatCompletionSchema = z.object({
  model: z.string().trim().min(2).max(120).default(publicModelId),
  messages: z.array(chatMessageSchema).min(1).max(100),
  temperature: z.number().min(0).max(2).default(0.7),
  max_tokens: z.number().int().min(1).max(4096).default(4096),
  top_p: z.number().min(0).max(1).optional(),
  stop: z.array(z.string().min(1).max(100)).max(8).optional(),
  stream: z.boolean().optional().default(false),
  tools: z.array(functionToolSchema).max(64).optional(),
  tool_choice: toolChoiceSchema.optional(),
  parallel_tool_calls: z.boolean().optional(),
  conversation_id: z.string().uuid().optional(),
  user: z.string().trim().max(200).optional(),
  client_name: z.string().trim().max(120).optional(),
})

let upstreamCursor = 0
const globalQueueAdvisoryLock = 782398451

function positiveIntegerEnvironment(name, fallback, maximum = Number.MAX_SAFE_INTEGER) {
  const parsed = Number(process.env[name])
  return Number.isInteger(parsed) && parsed > 0 ? Math.min(parsed, maximum) : fallback
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds))
}

function gatewayError(message, statusCode, code, retryAfterSeconds) {
  const error = new Error(message)
  error.statusCode = statusCode
  error.code = code
  error.retryAfterSeconds = retryAfterSeconds
  error.expose = true
  return error
}

function analyzeProhibitedUse(messages) {
  const text = String([...messages].reverse().find((message) => message.role === "user")?.content || "")
    .slice(-30000)
    .normalize("NFKC")
    .toLowerCase()
  const normalizedRequest = text.replace(/^\s*(?:(?:can|could|would|will) you\s+|please\s+)/i, "")
  const requestIntent = /\b(?:build|collect|compile|create|design|develop|extract|generate|give me|help me|how do i|implement|label|make|need to|produce|provide|run|show me|synthesize|teach me|train|use your (?:answers|outputs|responses)|want to|write)\b/i
  const informationalOnly = /^\s*(?:compare|define|describe|discuss|explain|how does|summarize|what is|what are|why is)\b/i
  const distillationTerms = /\b(?:knowledge distill(?:ation|ing)?|distill(?:ation|ing)?|teacher[- ]student|student model|clone (?:the |this )?model|replicate (?:the |this )?model|extract (?:the |this )?model(?:'s)? (?:behavior|responses?))\b/i
  const trainingDataTerms = /\b(?:training (?:corpus|data|dataset|examples?|samples?)|synthetic (?:corpus|data|dataset|examples?|samples?)|fine[- ]tun(?:e|ing) (?:data|dataset)|instruction (?:data|dataset)|prompt[- ]response pairs?|labeled (?:data|dataset)|benchmark dataset)\b/i
  const rules = [
    {
      id: "knowledge_distillation",
      category: "knowledge_distillation",
      score: 6,
      matches: distillationTerms.test(text) && requestIntent.test(text) && !informationalOnly.test(normalizedRequest),
    },
    {
      id: "training_data_generation",
      category: "training_data_generation",
      score: 6,
      matches: requestIntent.test(text) && trainingDataTerms.test(text),
    },
  ]
  const matched = rules.filter((rule) => rule.matches)
  const score = matched.reduce((total, rule) => total + rule.score, 0)
  const severity = score >= 6 ? "critical" : score >= 4 ? "high" : score >= 2 ? "medium" : "low"
  return {
    severity,
    score,
    categories: [...new Set(matched.map((rule) => rule.category))],
    matchedRules: matched.map((rule) => rule.id),
    contentSha256: crypto.createHash("sha256").update(text).digest("hex"),
  }
}

async function acquireGlobalInferenceLease(pool, apiKey, requestId) {
  const capacity = positiveIntegerEnvironment("GLOBAL_INFERENCE_CAPACITY", 2, 64)
  const maxQueued = positiveIntegerEnvironment("GLOBAL_INFERENCE_MAX_QUEUE", 40, 1000)
  const timeoutMs = positiveIntegerEnvironment("GLOBAL_INFERENCE_QUEUE_TIMEOUT_MS", 120000, 600000)
  const leaseSeconds = positiveIntegerEnvironment("GLOBAL_INFERENCE_LEASE_SECONDS", 240, 900)
  const leaseId = crypto.randomUUID()
  const queuedAt = Date.now()
  const client = await pool.connect()
  try {
    await client.query("BEGIN")
    await client.query("SELECT pg_advisory_xact_lock($1::bigint)", [globalQueueAdvisoryLock])
    await client.query("DELETE FROM inference_queue_leases WHERE expires_at <= now()")
    const queued = await client.query("SELECT count(*)::int AS count FROM inference_queue_leases WHERE status = 'waiting'")
    if (Number(queued.rows[0]?.count || 0) >= maxQueued) {
      await client.query("COMMIT")
      throw gatewayError("The inference queue is full. Try again shortly.", 503, "inference_queue_full", 5)
    }
    await client.query(
      `INSERT INTO inference_queue_leases (
         id, request_id, api_key_id, special_request_id, status, expires_at
       ) VALUES ($1, $2, $3, $4, 'waiting', now() + make_interval(secs => $5))`,
      [leaseId, requestId, apiKey.id, apiKey.special_request_id, Math.ceil(timeoutMs / 1000) + leaseSeconds],
    )
    await client.query("COMMIT")
  } catch (error) {
    await client.query("ROLLBACK").catch(() => {})
    throw error
  } finally {
    client.release()
  }

  while (Date.now() - queuedAt < timeoutMs) {
    const attempt = await pool.connect()
    try {
      await attempt.query("BEGIN")
      await attempt.query("SELECT pg_advisory_xact_lock($1::bigint)", [globalQueueAdvisoryLock])
      await attempt.query("DELETE FROM inference_queue_leases WHERE expires_at <= now()")
      const running = await attempt.query("SELECT count(*)::int AS count FROM inference_queue_leases WHERE status = 'running'")
      const oldest = await attempt.query(
        "SELECT id FROM inference_queue_leases WHERE status = 'waiting' ORDER BY queued_at, id LIMIT 1 FOR UPDATE",
      )
      if (Number(running.rows[0]?.count || 0) < capacity && oldest.rows[0]?.id === leaseId) {
        await attempt.query(
          `UPDATE inference_queue_leases
           SET status = 'running', started_at = now(), expires_at = now() + make_interval(secs => $2)
           WHERE id = $1`,
          [leaseId, leaseSeconds],
        )
        await attempt.query("COMMIT")
        return { id: leaseId, capacity, waitMs: Date.now() - queuedAt }
      }
      await attempt.query("COMMIT")
    } catch (error) {
      await attempt.query("ROLLBACK").catch(() => {})
      await pool.query("DELETE FROM inference_queue_leases WHERE id = $1", [leaseId]).catch(() => {})
      throw error
    } finally {
      attempt.release()
    }
    await sleep(200 + crypto.randomInt(100))
  }

  await pool.query("DELETE FROM inference_queue_leases WHERE id = $1", [leaseId])
  throw gatewayError("Timed out while waiting for model capacity.", 503, "inference_queue_timeout", 3)
}

async function releaseGlobalInferenceLease(pool, lease) {
  if (!lease?.id) return
  await pool.query("DELETE FROM inference_queue_leases WHERE id = $1", [lease.id])
}

function upstreamChatTargets() {
  const upstreamMode = String(process.env.MODEL_UPSTREAM_MODE || "chat").trim().toLowerCase()
  const configured = String(process.env.MODEL_UPSTREAM_URLS || process.env.MODEL_UPSTREAM_URL || "")
    .split(",")
    .map((value) => value.trim().replace(/\/+$/, ""))
    .filter(Boolean)
    .map((value, index) => ({
      id: index + 1,
      url:
        upstreamMode === "completion"
          ? value.endsWith("/completion")
            ? value
            : `${value.replace(/\/v1$/, "")}/completion`
          : value.endsWith("/chat/completions")
            ? value
            : `${value}/chat/completions`,
    }))

  if (configured.length < 2) return configured

  const start = upstreamCursor % configured.length
  upstreamCursor = (upstreamCursor + 1) % configured.length
  return [...configured.slice(start), ...configured.slice(0, start)]
}

function renderQwenPrompt(messages, tools, toolChoice) {
  const firstMessage = messages[0]
  const hasSystemMessage = firstMessage?.role === "system"
  const systemMessage = hasSystemMessage ? firstMessage.content : "You are a helpful assistant."
  let prompt = `<|im_start|>system\n${systemMessage || ""}`

  if (tools?.length) {
    prompt += "\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
    prompt += "\n\nReturn each function call as JSON inside <tool_call></tool_call> XML tags."
    prompt += " Do not describe a tool call in prose and do not invent tool results."
    if (toolChoice === "required") {
      prompt += " You must call one of the provided tools before answering."
    } else if (typeof toolChoice === "object") {
      prompt += ` You must call the ${toolChoice.function.name} tool before answering.`
    } else if (toolChoice !== "none") {
      prompt += " Call a provided tool whenever the user asks you to perform an action or retrieve information it can supply."
    }
    prompt += "\n<tools>"
    for (const tool of tools) prompt += `\n${JSON.stringify(tool)}`
    prompt += "\n</tools>"
  }
  prompt += "<|im_end|>\n"

  for (const message of hasSystemMessage ? messages.slice(1) : messages) {
    if (message.role === "assistant" && message.tool_calls?.length) {
      prompt += `<|im_start|>assistant\n${message.content || ""}`
      for (const toolCall of message.tool_calls) {
        const args = toolCall.function.arguments || "{}"
        let parsedArgs
        try {
          parsedArgs = JSON.parse(args)
        } catch {
          parsedArgs = { raw: args }
        }
        prompt += `\n<tool_call>\n${JSON.stringify({ name: toolCall.function.name, arguments: parsedArgs })}\n</tool_call>`
      }
      prompt += "<|im_end|>\n"
      continue
    }

    if (message.role === "tool") {
      prompt += `<|im_start|>user\n<tool_response>\n${message.content || ""}\n</tool_response><|im_end|>\n`
      continue
    }

    prompt += `<|im_start|>${message.role}\n${message.content || ""}<|im_end|>\n`
  }

  return `${prompt}<|im_start|>assistant\n`
}

function parseXmlToolCalls(content) {
  if (typeof content !== "string") return { content, toolCalls: [] }

  if (!content.includes("<tool_call>")) {
    const jsonCandidate = content.trim().replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "")
    try {
      const parsed = JSON.parse(jsonCandidate)
      if (parsed?.name && typeof parsed.name === "string") {
        return {
          content: null,
          toolCalls: [{
            id: `call_${crypto.randomUUID().replaceAll("-", "")}`,
            type: "function",
            function: {
              name: parsed.name,
              arguments: typeof parsed.arguments === "string"
                ? parsed.arguments
                : JSON.stringify(parsed.arguments || {}),
            },
          }],
        }
      }
    } catch {}
    return { content, toolCalls: [] }
  }

  const toolCalls = []
  const remainingContent = content.replace(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g, (match, payload) => {
    try {
      const parsed = JSON.parse(payload)
      if (!parsed?.name || typeof parsed.name !== "string") return match
      const args = typeof parsed.arguments === "string" ? parsed.arguments : JSON.stringify(parsed.arguments || {})
      toolCalls.push({
        id: `call_${crypto.randomUUID().replaceAll("-", "")}`,
        type: "function",
        function: { name: parsed.name, arguments: args },
      })
      return ""
    } catch {
      return match
    }
  }).trim()

  return { content: remainingContent || null, toolCalls }
}

function sanitizeModelContent(content) {
  if (typeof content !== "string") return content

  let source = content
    .replaceAll("\0", "")
    .replaceAll("\uFFFD", "")
    .replaceAll("<|im_end|>", "\n")
    .replace(/<\|im_start\|>assistant\s*/gi, "\n")
    .replace(/<\|im_start\|>(?:user|system|tool)[\s\S]*$/i, "")
    .replace(/\/+renderer\/+renderer/gi, "\n")
    .replace(/\s+w\u0142a\u015bc(?:\.PLLCHEMY)?[\s\S]*$/iu, "")
  const repeatedLoadArtifact = /:load(?=(?:[\s\S]*:load){2})/i.exec(source)
  if (repeatedLoadArtifact) {
    const prefix = source.slice(0, repeatedLoadArtifact.index)
    const sentenceEnds = [...prefix.matchAll(/[.!?](?=\s|$)/g)]
    const lastSentenceEnd = sentenceEnds.at(-1)?.index
    source = lastSentenceEnd >= 20 ? prefix.slice(0, lastSentenceEnd + 1) : prefix
  }

  const sourceLines = source.split(/\r?\n/)
  const cleanLines = []

  for (const line of sourceLines) {
    const trimmed = line.trim()
    const isLeakedRoleMarker = trimmed.length <= 64 && /^[^\s]*(?:assistant|user|system|tool)$/iu.test(trimmed)
    if (!isLeakedRoleMarker) {
      if (trimmed && cleanLines.at(-1)?.trim() === trimmed) continue
      cleanLines.push(line)
      continue
    }

    const previous = cleanLines.at(-1)
    if (previous) {
      const asciiCharacters = [...previous].filter((character) => character.charCodeAt(0) <= 127).length
      if (asciiCharacters / Math.max(previous.length, 1) > 0.8) {
        cleanLines[cleanLines.length - 1] = previous
          .replace(/[^\x00-\x7F]{1,16}\s*$/u, "")
          .replace(/(?:[.$]?[A-Za-z]*ernetes)\s*$/i, "")
          .trimEnd()
      }
    }
    break
  }

  const cleaned = cleanLines
    .join("\n")
    .replace(/[^\x00-\x7F]{1,16}assistant\s*$/iu, "")
    .replace(/([.!?])assistant\s*$/i, "$1")
    .replace(/(?:^|\n)\s*(?:assistant|user|system|tool)\s*:?\s*$/gimu, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()

  const paragraphs = cleaned.split(/\n{2,}/u)
  const deduplicated = []
  for (const paragraph of paragraphs) {
    const normalized = paragraph.trim().toLowerCase()
    if (normalized.length >= 24 && deduplicated.at(-1)?.trim().toLowerCase() === normalized) continue
    deduplicated.push(paragraph)
  }
  return deduplicated.join("\n\n").trim()
}

function requiredToolJsonSchema(tools, toolChoice) {
  if (!tools?.length || toolChoice === "none" || toolChoice === "auto" || toolChoice === undefined) return null

  const requiredName = typeof toolChoice === "object" ? toolChoice.function.name : null
  const candidates = requiredName
    ? tools.filter((tool) => tool.function.name === requiredName)
    : tools
  if (candidates.length === 0) return null

  const schemas = candidates.map((tool) => ({
    type: "object",
    properties: {
      name: { const: tool.function.name },
      arguments: tool.function.parameters,
    },
    required: ["name", "arguments"],
    additionalProperties: false,
  }))
  return schemas.length === 1 ? schemas[0] : { oneOf: schemas }
}

function upstreamHeaders(requestId) {
  const headers = {
    "Content-Type": "application/json",
    "X-Request-ID": requestId,
  }
  const apiKey = String(process.env.MODEL_UPSTREAM_API_KEY || "").trim()
  if (!apiKey) return headers

  const headerName = String(process.env.MODEL_UPSTREAM_AUTH_HEADER || "Authorization").trim()
  const scheme = String(process.env.MODEL_UPSTREAM_AUTH_SCHEME || "Bearer").trim()
  headers[headerName] = scheme ? `${scheme} ${apiKey}` : apiKey
  return headers
}

function normalizeUpstreamResponse(data, requestId, conversationId) {
  const firstChoice = data?.choices?.[0]
  const upstreamMessage = firstChoice?.message || {}
  const rawContent =
    upstreamMessage.content ??
    firstChoice?.text ??
    data?.message?.content ??
    data?.content ??
    data?.response
  const parsedContent = parseXmlToolCalls(sanitizeModelContent(rawContent))
  const toolCalls = Array.isArray(upstreamMessage.tool_calls) && upstreamMessage.tool_calls.length
    ? upstreamMessage.tool_calls
    : parsedContent.toolCalls
  const content = parsedContent.content

  if (typeof content !== "string" && toolCalls.length === 0) {
    throw new Error("The model server returned a response without assistant content or tool calls.")
  }

  const message = {
    role: "assistant",
    content: typeof content === "string" ? content : null,
    ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
  }

  return {
    id: data?.id || `chatcmpl_${requestId}`,
    object: "chat.completion",
    created: data?.created || Math.floor(Date.now() / 1000),
    model: publicModelId,
    choices: [
      {
        index: 0,
        message,
        finish_reason: firstChoice?.finish_reason || data?.finish_reason || (toolCalls.length ? "tool_calls" : "stop"),
      },
    ],
    usage: {
      prompt_tokens: Number(data?.usage?.prompt_tokens || data?.usage?.input_tokens || data?.tokens_evaluated || 0),
      completion_tokens: Number(data?.usage?.completion_tokens || data?.usage?.output_tokens || data?.tokens_predicted || 0),
      total_tokens: Number(
        data?.usage?.total_tokens ||
          Number(data?.usage?.prompt_tokens || data?.usage?.input_tokens || data?.tokens_evaluated || 0) +
            Number(data?.usage?.completion_tokens || data?.usage?.output_tokens || data?.tokens_predicted || 0),
      ),
    },
    request_id: requestId,
    conversation_id: conversationId,
  }
}

function assessRawModelOutput(data) {
  const firstChoice = data?.choices?.[0]
  const upstreamMessage = firstChoice?.message || {}
  if (Array.isArray(upstreamMessage.tool_calls) && upstreamMessage.tool_calls.length > 0) return { ok: true, issues: [] }
  const raw = String(
    upstreamMessage.content ?? firstChoice?.text ?? data?.message?.content ?? data?.content ?? data?.response ?? "",
  )
  const warnings = []
  const repeatedLoadArtifact = /:load(?=(?:[\s\S]*:load){2})/i.test(raw)
  if (repeatedLoadArtifact) warnings.push("raw_repeated_token_artifact")
  if (/\/+renderer\/+renderer|w\u0142a\u015bc|PLLCHEMY/iu.test(raw)) warnings.push("raw_training_artifact")
  if (/<\|im_start\|>(?:user|system|tool)|<\|im_end\|>\s*<\|im_start\|>/iu.test(raw)) warnings.push("raw_chat_marker_leak")
  if (/(?:^|\n)\s*(?:assistant|user|system|tool)\s*:?\s*$/imu.test(raw)) warnings.push("raw_role_marker_leak")
  if (/(?:^|\n)[^\x00-\x7F]{1,16}(?:assistant|user|system|tool)\s*$/imu.test(raw)) warnings.push("raw_role_marker_leak")

  const paragraphs = raw
    .split(/\n{2,}/u)
    .map((paragraph) => paragraph.trim().toLowerCase())
    .filter((paragraph) => paragraph.length >= 40)
  for (let index = 2; index < paragraphs.length; index += 1) {
    if (paragraphs[index] === paragraphs[index - 1] && paragraphs[index] === paragraphs[index - 2]) {
      warnings.push("raw_repeated_output")
      break
    }
  }

  const cleaned = sanitizeModelContent(raw)
  const issues = []
  if (!String(cleaned || "").trim()) issues.push(raw.trim() ? "empty_after_sanitize" : "empty_output")
  if (repeatedLoadArtifact && String(cleaned || "").trim().length < 40) issues.push("insufficient_output_after_sanitize")
  if (/<\|im_(?:start|end)\|>|(?:^|\n)\s*(?:assistant|user|system|tool)\s*:?\s*$/imu.test(String(cleaned || ""))) {
    issues.push("role_marker_after_sanitize")
  }
  const cleanedParagraphs = String(cleaned || "")
    .split(/\n{2,}/u)
    .map((paragraph) => paragraph.trim().toLowerCase())
    .filter((paragraph) => paragraph.length >= 40)
  for (let index = 2; index < cleanedParagraphs.length; index += 1) {
    if (cleanedParagraphs[index] === cleanedParagraphs[index - 1] && cleanedParagraphs[index] === cleanedParagraphs[index - 2]) {
      issues.push("repeated_output_after_sanitize")
      break
    }
  }
  return { ok: issues.length === 0, issues, warnings: [...new Set(warnings)] }
}

function sendBufferedStream(res, responseBody) {
  const choice = responseBody.choices[0]
  const message = choice.message
  const chunkBase = {
    id: responseBody.id,
    object: "chat.completion.chunk",
    created: responseBody.created,
    model: responseBody.model,
  }
  const delta = { role: "assistant" }
  if (typeof message.content === "string") delta.content = message.content
  if (message.tool_calls?.length) {
    delta.tool_calls = message.tool_calls.map((toolCall, index) => ({ index, ...toolCall }))
  }

  res.status(200)
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8")
  res.setHeader("Cache-Control", "no-cache, no-transform")
  res.setHeader("Connection", "keep-alive")
  res.write(`data: ${JSON.stringify({ ...chunkBase, choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`)
  res.write(
    `data: ${JSON.stringify({
      ...chunkBase,
      choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason }],
      usage: responseBody.usage,
      request_id: responseBody.request_id,
      conversation_id: responseBody.conversation_id,
    })}\n\n`,
  )
  res.end("data: [DONE]\n\n")
}

async function storeInteraction(pool, {
  apiKey,
  conversationId,
  requestId,
  requestPath,
  clientName,
  userReference,
  messages,
  assistantResponse,
  latencyMs,
  usage,
  statusCode,
  eventType,
  metadata,
  distillationReview,
}) {
  const client = await pool.connect()
  try {
    await client.query("BEGIN")
    const conversation = await client.query(
      `INSERT INTO api_conversations (
         id, special_request_id, api_key_prefix, user_reference, client_name, model, metadata
       ) VALUES ($1, $2, $3, $4, $5, $6, $7)
       ON CONFLICT (id) DO UPDATE SET
         updated_at = now(),
         user_reference = COALESCE(EXCLUDED.user_reference, api_conversations.user_reference),
         client_name = COALESCE(EXCLUDED.client_name, api_conversations.client_name),
         metadata = api_conversations.metadata || EXCLUDED.metadata
       WHERE api_conversations.special_request_id = EXCLUDED.special_request_id
       RETURNING id`,
      [
        conversationId,
        apiKey.special_request_id,
        apiKey.key_prefix,
        userReference || null,
        clientName || null,
        publicModelId,
        { lastRequestId: requestId, ...(metadata || {}) },
      ],
    )
    if (conversation.rowCount === 0) {
      const error = new Error("Conversation ID belongs to a different Access Token.")
      error.statusCode = 409
      throw error
    }

    for (const [index, message] of messages.entries()) {
      await client.query(
        `INSERT INTO conversation_messages (conversation_id, role, content, metadata)
         VALUES ($1, $2, $3, $4)`,
        [
          conversationId,
          message.role,
          message.content || "",
          {
            requestId,
            sequenceIndex: index,
            direction: "input",
            ...(message.name ? { name: message.name } : {}),
            ...(message.tool_call_id ? { toolCallId: message.tool_call_id } : {}),
            ...(message.tool_calls?.length ? { toolCalls: message.tool_calls } : {}),
          },
        ],
      )
    }

    if (assistantResponse) {
      await client.query(
        `INSERT INTO conversation_messages (conversation_id, role, content, token_count, latency_ms, metadata)
         VALUES ($1, 'assistant', $2, $3, $4, $5)`,
        [
          conversationId,
          assistantResponse.content || "",
          usage?.completion_tokens || null,
          latencyMs,
          {
            requestId,
            sequenceIndex: messages.length,
            direction: "output",
            ...(assistantResponse.tool_calls?.length ? { toolCalls: assistantResponse.tool_calls } : {}),
          },
        ],
      )
    }

    await client.query(
      `INSERT INTO api_usage_events (
         special_request_id, api_key_prefix, event_type, request_path, request_id,
         input_tokens, output_tokens, status_code, metadata
       ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
      [
        apiKey.special_request_id,
        apiKey.key_prefix,
        eventType,
        requestPath,
        requestId,
        usage?.prompt_tokens || null,
        usage?.completion_tokens || null,
        statusCode,
        { conversationId, latencyMs, ...(metadata || {}) },
      ],
    )

    if (distillationReview?.matchedRules?.length) {
      const lastUserMessage = [...messages]
        .reverse()
        .find((message) => message.role === "user")?.content || ""
      const modelResponse = assistantResponse
        ? assistantResponse.content || JSON.stringify(assistantResponse.tool_calls || [])
        : null
      await client.query(
        `INSERT INTO api_safety_events (
           api_key_id, special_request_id, request_id, conversation_id, severity, action,
           categories, matched_rules, content_sha256, user_message, model_response,
           review_status, metadata
         ) VALUES ($1, $2, $3, $4, $5, 'observed', $6, $7, $8, $9, $10, 'pending', $11)
         ON CONFLICT (request_id) DO UPDATE SET
           conversation_id = EXCLUDED.conversation_id,
           user_message = EXCLUDED.user_message,
           model_response = COALESCE(EXCLUDED.model_response, api_safety_events.model_response),
           metadata = api_safety_events.metadata || EXCLUDED.metadata`,
        [
          apiKey.id,
          apiKey.special_request_id,
          requestId,
          conversationId,
          distillationReview.severity,
          distillationReview.categories,
          distillationReview.matchedRules,
          distillationReview.contentSha256,
          String(lastUserMessage).slice(0, 100000),
          modelResponse === null ? null : String(modelResponse).slice(0, 100000),
          {
            score: distillationReview.score,
            detectionMode: "manual_review_only",
            accessAutomaticallyChanged: false,
          },
        ],
      )
    }
    await client.query("COMMIT")
  } catch (error) {
    await client.query("ROLLBACK")
    throw error
  } finally {
    client.release()
  }
}

async function conversationBelongsToApiKey(pool, conversationId, specialRequestId) {
  const result = await pool.query(
    "SELECT special_request_id FROM api_conversations WHERE id = $1",
    [conversationId],
  )
  return result.rowCount === 0 || result.rows[0].special_request_id === specialRequestId
}

function fixedCompletionResponse(message, requestId, conversationId) {
  return {
    id: `chatcmpl_${requestId}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: publicModelId,
    choices: [{ index: 0, message: { role: "assistant", content: message }, finish_reason: "stop" }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    request_id: requestId,
    conversation_id: conversationId,
  }
}

function fixedClineCompletionResponse(message, requestId, conversationId, tools) {
  const completionTool = tools?.find((tool) =>
    ["attempt_completion", "submit_and_exit"].includes(tool.function.name),
  )
  if (!completionTool) {
    const escapedMessage = message
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
    return fixedCompletionResponse(
      `<attempt_completion>\n<result>${escapedMessage}</result>\n</attempt_completion>`,
      requestId,
      conversationId,
    )
  }

  const response = fixedCompletionResponse(message, requestId, conversationId)
  response.choices[0].message = {
    role: "assistant",
    content: null,
    tool_calls: [
      {
        id: `call_${crypto.randomUUID().replaceAll("-", "")}`,
        type: "function",
        function: {
          name: completionTool.function.name,
          arguments: JSON.stringify({ result: message }),
        },
      },
    ],
  }
  response.choices[0].finish_reason = "tool_calls"
  return response
}

async function storeFailedRequest(pool, context, error) {
  try {
    await storeInteraction(pool, {
      ...context,
      assistantResponse: null,
      usage: null,
      eventType: "chat_completion_error",
      metadata: { upstreamError: String(error).slice(0, 500) },
    })
  } catch (storageError) {
    console.error("Failed to store model API error event", storageError)
  }
}

export function registerModelGateway({
  app,
  pool,
  authenticateApiKey,
  enforceApiKeyRateLimit,
  enforceApiKeyConcurrency,
}) {
  app.get("/v1/access", authenticateApiKey, (req, res) => {
    res.json({
      status: "active",
      model: publicModelId,
      model_type: "text-to-text",
      key_prefix: req.apiKey.key_prefix,
      special_request_id: req.apiKey.special_request_id,
      rpm_limit: Number(req.apiKey.rpm_limit),
    })
  })

  app.get("/v1/models", authenticateApiKey, (_req, res) => {
    res.json({
      object: "list",
      data: [
        {
          id: publicModelId,
          object: "model",
          created: 0,
          owned_by: "tokenai",
          model_type: "text-to-text",
          input_modalities: ["text"],
          output_modalities: ["text"],
          context_window: 32768,
          max_output_tokens: 4096,
        },
      ],
    })
  })

  app.post(
    "/v1/chat/completions",
    authenticateApiKey,
    enforceApiKeyRateLimit,
    enforceApiKeyConcurrency,
    async (req, res) => {
      const requestId = `hcnreq_${crypto.randomUUID()}`
      const startedAt = Date.now()
      res.setHeader("X-Request-ID", requestId)

      const parsed = chatCompletionSchema.safeParse(req.body)
      if (!parsed.success) {
        res.status(400).json({
          error: { code: "invalid_request", message: "Invalid chat completion request.", issues: parsed.error.issues },
          request_id: requestId,
        })
        return
      }

      const data = parsed.data
      if (!acceptedModelIds.has(data.model)) {
        res.status(404).json({
          error: { code: "model_not_found", message: `Model '${data.model}' is not available.` },
          request_id: requestId,
        })
        return
      }

      const conversationId = data.conversation_id || crypto.randomUUID()
      if (!(await conversationBelongsToApiKey(pool, conversationId, req.apiKey.special_request_id))) {
        res.status(409).json({
          error: { code: "conversation_owner_mismatch", message: "Conversation ID belongs to another Access Token." },
          request_id: requestId,
        })
        return
      }
      const requestPath = req.originalUrl || req.path
      const distillationReview = analyzeProhibitedUse(data.messages)
      const baseContext = {
        apiKey: req.apiKey,
        conversationId,
        requestId,
        requestPath,
        clientName: data.client_name || req.get("x-client-name") || null,
        userReference: data.user || null,
        messages: data.messages,
        latencyMs: 0,
        statusCode: 500,
        distillationReview,
      }
      const fixedResponse = req.apiKey.metadata?.fixedResponse
      if (typeof fixedResponse === "string" && fixedResponse.length > 0) {
        const fixedMessage = fixedResponse.slice(0, 4000)
        const responseBody = req.apiKey.metadata?.clineCompletion
          ? fixedClineCompletionResponse(fixedMessage, requestId, conversationId, data.tools)
          : fixedCompletionResponse(fixedMessage, requestId, conversationId)
        const latencyMs = Date.now() - startedAt
        await storeInteraction(pool, {
          ...baseContext,
          assistantResponse: responseBody.choices[0].message,
          latencyMs,
          usage: responseBody.usage,
          statusCode: 200,
          eventType: "fixed_test_completion",
          metadata: {
            fixedTestResponse: true,
            clineCompletion: Boolean(req.apiKey.metadata?.clineCompletion),
          },
        })
        if (data.stream) {
          sendBufferedStream(res, responseBody)
          return
        }
        res.json(responseBody)
        return
      }
      const upstreamTargets = upstreamChatTargets()

      if (upstreamTargets.length === 0) {
        const latencyMs = Date.now() - startedAt
        await storeFailedRequest(
          pool,
          { ...baseContext, latencyMs, statusCode: 503 },
          "MODEL_UPSTREAM_URLS is not configured",
        )
        res.status(503).json({
          error: {
            code: "model_not_configured",
            message: "The model inference service is not connected yet.",
          },
          request_id: requestId,
          conversation_id: conversationId,
        })
        return
      }

      const upstreamModel = String(process.env.MODEL_UPSTREAM_MODEL || publicModelId).trim()
      const upstreamMode = String(process.env.MODEL_UPSTREAM_MODE || "chat").trim().toLowerCase()
      const timeoutMs = Number(process.env.MODEL_UPSTREAM_TIMEOUT_MS || 120000)
      const stop = [...new Set([...(data.stop || []), "<|im_end|>", "<|endoftext|>"])].slice(0, 8)
      const toolJsonSchema = requiredToolJsonSchema(data.tools, data.tool_choice)
      const upstreamPayload = upstreamMode === "completion"
        ? {
            prompt: renderQwenPrompt(data.messages, data.tools, data.tool_choice),
            temperature: data.tools?.length ? Math.min(data.temperature, 0.2) : data.temperature,
            n_predict: data.max_tokens,
            stream: false,
            stop,
            ...(data.top_p === undefined ? {} : { top_p: data.top_p }),
            ...(toolJsonSchema ? { json_schema: toolJsonSchema } : {}),
          }
        : {
            model: upstreamModel,
            messages: data.messages,
            temperature: data.temperature,
            max_tokens: data.max_tokens,
            stream: false,
            ...(data.top_p === undefined ? {} : { top_p: data.top_p }),
            stop,
            ...(data.tools === undefined ? {} : { tools: data.tools }),
            ...(data.tool_choice === undefined ? {} : { tool_choice: data.tool_choice }),
            ...(data.parallel_tool_calls === undefined ? {} : { parallel_tool_calls: data.parallel_tool_calls }),
          }

      let inferenceLease
      try {
        inferenceLease = await acquireGlobalInferenceLease(pool, req.apiKey, requestId)
        res.setHeader("X-Inference-Queue-Wait-Ms", String(inferenceLease.waitMs))
        res.setHeader("X-Inference-Capacity", String(inferenceLease.capacity))
        let upstreamResponse
        let upstreamData
        let selectedTarget
        let lastUpstreamError
        let responseQuality = { ok: true, issues: [], warnings: [] }
        let qualityRetries = 0

        for (const [targetIndex, target] of upstreamTargets.entries()) {
          try {
            const response = await fetch(target.url, {
              method: "POST",
              headers: upstreamHeaders(requestId),
              body: JSON.stringify(upstreamPayload),
              signal: AbortSignal.timeout(timeoutMs),
            })
            const responseText = await response.text()
            let responseData
            try {
              responseData = responseText ? JSON.parse(responseText) : {}
            } catch {
              responseData = { error: responseText.slice(0, 500) }
            }

            if (response.status >= 500 && targetIndex < upstreamTargets.length - 1) {
              lastUpstreamError = responseData?.error?.message || responseData?.error || `Upstream HTTP ${response.status}`
              continue
            }

            if (response.ok) {
              const quality = assessRawModelOutput(responseData)
              if (!quality.ok && targetIndex < upstreamTargets.length - 1) {
                responseQuality = quality
                qualityRetries += 1
                lastUpstreamError = `Model output quality check failed: ${quality.issues.join(", ")}`
                continue
              }
              responseQuality = quality
              if (!quality.ok && String(process.env.MODEL_QUALITY_FAIL_CLOSED || "true").toLowerCase() !== "false") {
                throw gatewayError(
                  "The model produced an invalid response. Try again.",
                  502,
                  "model_output_quality_failure",
                  1,
                )
              }
            }

            upstreamResponse = response
            upstreamData = responseData
            selectedTarget = target
            break
          } catch (error) {
            lastUpstreamError = error
            if (targetIndex === upstreamTargets.length - 1) throw error
          }
        }

        if (!upstreamResponse || !selectedTarget) {
          throw lastUpstreamError || new Error("No model worker was available.")
        }

        if (!upstreamResponse.ok) {
          const latencyMs = Date.now() - startedAt
          await storeFailedRequest(
            pool,
            { ...baseContext, latencyMs, statusCode: upstreamResponse.status },
            upstreamData?.error?.message || upstreamData?.error || `Upstream HTTP ${upstreamResponse.status}`,
          )
          res.status(502).json({
            error: { code: "model_upstream_error", message: "The model service could not complete the request." },
            request_id: requestId,
            conversation_id: conversationId,
          })
          return
        }

        const responseBody = normalizeUpstreamResponse(upstreamData, requestId, conversationId)
        const latencyMs = Date.now() - startedAt
        await storeInteraction(pool, {
          ...baseContext,
          assistantResponse: responseBody.choices[0].message,
          latencyMs,
          usage: responseBody.usage,
          statusCode: 200,
          eventType: "chat_completion",
          metadata: {
            upstreamModel,
            upstreamMode,
            upstreamWorker: selectedTarget.id,
            inferenceQueueWaitMs: inferenceLease.waitMs,
            modelQualityRetries: qualityRetries,
            modelQualityIssues: responseQuality.issues,
            modelQualityWarnings: responseQuality.warnings || [],
            ...(distillationReview.matchedRules.length
              ? {
                  distillationReviewSeverity: distillationReview.severity,
                  distillationReviewRules: distillationReview.matchedRules,
                }
              : {}),
          },
        })
        if (data.stream) {
          sendBufferedStream(res, responseBody)
          return
        }
        res.json(responseBody)
      } catch (error) {
        const latencyMs = Date.now() - startedAt
        console.error("Model gateway request failed", { requestId, error })
        const statusCode = Number(error.statusCode || 502)
        await storeFailedRequest(pool, { ...baseContext, latencyMs, statusCode }, error)
        if (error.retryAfterSeconds) res.setHeader("Retry-After", String(error.retryAfterSeconds))
        res.status(statusCode).json({
          error: {
            code: error.code || "model_gateway_error",
            message: error.expose ? error.message : "The model service is temporarily unavailable.",
          },
          request_id: requestId,
          conversation_id: conversationId,
        })
      } finally {
        await releaseGlobalInferenceLease(pool, inferenceLease).catch((error) => {
          console.error("Failed to release global inference lease", { requestId, error })
        })
      }
    },
  )
}
