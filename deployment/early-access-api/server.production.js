import crypto from "node:crypto"
import dns from "node:dns/promises"
import fs from "node:fs"
import path from "node:path"
import cors from "cors"
import express from "express"
import helmet from "helmet"
import nodemailer from "nodemailer"
import pg from "pg"
import { z } from "zod"
import { registerModelGateway } from "./model-gateway.js"

const { Pool } = pg
const app = express()
const port = Number(process.env.PORT || 8080)
const adminToken = process.env.ADMIN_TOKEN
const apiKeyPepper = process.env.API_KEY_PEPPER
const apiKeyEncryptionKeyValue = process.env.API_KEY_ENCRYPTION_KEY
const defaultKeyRateLimitPerMinute = 20
const maxConcurrentRequestsPerKey = 2
const globalInferenceCapacity = Number(process.env.GLOBAL_INFERENCE_CAPACITY || 2)
const logoPath = path.join(process.cwd(), "assets", "tokenai-logo.png")

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  max: 10,
})

const allowedOrigins = (process.env.TOKENAI_ALLOWED_ORIGINS || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean)

if (!adminToken || adminToken.length < 32) {
  throw new Error("ADMIN_TOKEN must be configured with at least 32 characters.")
}
if (!apiKeyPepper || apiKeyPepper.length < 32) {
  throw new Error("API_KEY_PEPPER must be configured with at least 32 characters.")
}
if (!apiKeyEncryptionKeyValue || apiKeyEncryptionKeyValue.length < 32) {
  throw new Error("API_KEY_ENCRYPTION_KEY must be configured with at least 32 characters.")
}

const apiKeyEncryptionKey = crypto
  .createHash("sha256")
  .update(`tokenai-api-key-encryption:${apiKeyEncryptionKeyValue}`)
  .digest()

const freeEmailDomains = new Set([
  "126.com", "163.com", "aol.com", "att.net", "bk.ru", "comcast.net", "daum.net", "duck.com",
  "email.com", "fastmail.com", "free.fr", "freenet.de", "gmail.com", "gmx.at", "gmx.com", "gmx.de", "gmx.net",
  "googlemail.com", "hotmail.co.uk", "hotmail.com", "icloud.com", "inbox.com", "laposte.net",
  "list.ru", "live.com", "live.co.uk", "mail.com", "mail.ru", "me.com", "msn.com", "naver.com",
  "outlook.com", "pm.me", "proton.me", "protonmail.com", "qq.com", "rambler.ru", "rediffmail.com",
  "rocketmail.com", "tutanota.com", "tuta.io", "verizon.net", "web.de", "yahoo.co.in",
  "yahoo.co.uk", "yahoo.com", "yandex.com", "yandex.ru", "ymail.com", "zoho.com",
])

app.disable("x-powered-by")
app.use(helmet())
app.use(express.json({ limit: "1mb" }))
app.use((_req, res, next) => {
  res.setHeader("Cache-Control", "no-store")
  next()
})
app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.length === 0 || allowedOrigins.includes(origin)) {
        callback(null, true)
        return
      }
      callback(new Error("Origin is not allowed"))
    },
  }),
)

function catchRouteErrors(handler) {
  return (req, res, next) => {
    try {
      const result = handler(req, res, next)
      if (result && typeof result.catch === "function") result.catch(next)
    } catch (error) {
      next(error)
    }
  }
}

for (const method of ["get", "post", "patch", "put", "delete"]) {
  const register = app[method].bind(app)
  app[method] = (route, ...handlers) => register(route, ...handlers.map(catchRouteErrors))
}

const urlField = z
  .string()
  .trim()
  .min(5)
  .transform((value) => (/^https?:\/\//i.test(value) ? value : `https://${value}`))
  .refine((value) => {
    try {
      const parsed = new URL(value)
      return ["http:", "https:"].includes(parsed.protocol) && parsed.hostname.includes(".")
    } catch {
      return false
    }
  }, "A valid URL is required.")

const optionalUrlField = z
  .string()
  .trim()
  .max(300)
  .transform((value) => {
    if (!value) return ""
    return /^https?:\/\//i.test(value) ? value : `https://${value}`
  })
  .refine((value) => {
    if (!value) return true
    try {
      const parsed = new URL(value)
      return ["http:", "https:"].includes(parsed.protocol) && parsed.hostname.includes(".")
    } catch {
      return false
    }
  }, "A valid URL is required.")

function isUrlOnDomain(value, domain) {
  try {
    const hostname = new URL(value).hostname.toLowerCase()
    return hostname === domain || hostname.endsWith(`.${domain}`)
  } catch {
    return false
  }
}

function isOptionalUrlOnDomain(value, domain) {
  return !value || isUrlOnDomain(value, domain)
}

function isEducationalEmail(value) {
  const email = String(value || "").trim().toLowerCase()
  const domain = email.split("@")[1]
  return (
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email) &&
    Boolean(domain) &&
    !freeEmailDomains.has(domain) &&
    (domain.endsWith(".edu") || domain.includes(".edu.") || /\.ac\.[a-z]{2,}$/i.test(domain))
  )
}

async function hasDeliverableMailDomain(value) {
  const domain = String(value || "").trim().toLowerCase().split("@")[1]
  if (!domain) return false

  try {
    const records = await Promise.race([
      dns.resolveMx(domain),
      new Promise((_, reject) => setTimeout(() => reject(new Error("MX lookup timed out")), 5000)),
    ])
    if (records.some((record) => record.exchange)) return true
  } catch {}

  try {
    const addresses = await Promise.race([
      dns.resolve(domain),
      new Promise((_, reject) => setTimeout(() => reject(new Error("Mail-domain lookup timed out")), 5000)),
    ])
    return addresses.length > 0
  } catch {
    return false
  }
}

function getGitHubUsername(value) {
  try {
    const parsed = new URL(value)
    const hostname = parsed.hostname.toLowerCase().replace(/^www\./, "")
    if (hostname !== "github.com") return ""
    const username = parsed.pathname.split("/").filter(Boolean)[0] || ""
    if (!/^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/.test(username)) return ""
    return username
  } catch {
    return ""
  }
}

function getLinkedInHandle(value) {
  try {
    const parsed = new URL(value)
    const hostname = parsed.hostname.toLowerCase().replace(/^www\./, "")
    if (hostname !== "linkedin.com") return ""
    const parts = parsed.pathname.split("/").filter(Boolean)
    if (parts[0] !== "in" || !parts[1]) return ""
    const handle = parts[1]
    if (!/^[A-Za-z0-9-_%]{3,120}$/.test(handle)) return ""
    return handle
  } catch {
    return ""
  }
}

async function validateProfileLinks(data) {
  if (data.linkedinProfileUrl && !getLinkedInHandle(data.linkedinProfileUrl)) {
    return "A valid LinkedIn profile URL is required."
  }

  if (data.githubProfileUrl) {
    const username = getGitHubUsername(data.githubProfileUrl)
    if (!username) return "A valid GitHub profile URL is required."

    const response = await fetch(`https://api.github.com/users/${encodeURIComponent(username)}`, {
      headers: {
        Accept: "application/vnd.github+json",
        "User-Agent": "TokenAI-EarlyAccess-Validator",
      },
      signal: AbortSignal.timeout(5000),
    })

    if (response.status === 404) return "GitHub account was not found."
    if (!response.ok) return "GitHub account could not be verified right now."
  }

  return ""
}

const accessRequestSchema = z
  .object({
    firstName: z.string().trim().min(2).max(80),
    secondName: z.string().trim().min(2).max(80),
    applicantType: z.enum(["developer_researcher", "student"]).default("developer_researcher"),
    companyName: z.string().trim().min(2).max(160),
    companyWebsite: urlField,
    job: z.string().trim().min(2).max(160),
    businessMail: z
      .string()
      .trim()
      .email()
      .max(180)
      .transform((value) => value.toLowerCase())
      .refine((value) => {
        const domain = value.split("@")[1]
        return Boolean(domain) && !freeEmailDomains.has(domain)
      }, "A business or educational email is required."),
    linkedinProfileUrl: optionalUrlField,
    githubProfileUrl: optionalUrlField,
    portfolioUrl: optionalUrlField.default(""),
    countryOrRegion: z.string().trim().min(2).max(120),
    useCases: z.string().trim().min(80).max(6000),
    responsibleUseAccepted: z.literal(true),
  })

const socialPostReviewSchema = z.object({
  socialPostUrl: urlField.refine(
    (value) => isUrlOnDomain(value, "linkedin.com") || isUrlOnDomain(value, "facebook.com"),
    "The social post must be on LinkedIn or Facebook.",
  ),
  status: z.enum(["verified", "rejected"]),
  reviewedBy: z.string().trim().min(2).max(120).default("dashboard"),
})
  .superRefine((data, context) => {
    if (data.applicantType === "student" && !isEducationalEmail(data.businessMail)) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["businessMail"],
        message: "A university or educational email address is required for student applicants.",
      })
    }

    if (data.applicantType !== "student") {
      if (!isUrlOnDomain(data.linkedinProfileUrl, "linkedin.com")) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["linkedinProfileUrl"],
          message: "A LinkedIn profile URL is required.",
        })
      }
      if (!isUrlOnDomain(data.githubProfileUrl, "github.com")) {
        context.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["githubProfileUrl"],
          message: "A GitHub profile URL is required.",
        })
      }
    }

    if (!isOptionalUrlOnDomain(data.linkedinProfileUrl, "linkedin.com")) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["linkedinProfileUrl"],
        message: "LinkedIn profile URL must be on linkedin.com.",
      })
    }

    if (!isOptionalUrlOnDomain(data.githubProfileUrl, "github.com")) {
      context.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["githubProfileUrl"],
        message: "GitHub profile URL must be on github.com.",
      })
    }
  })

function getDefaultRpmLimit(applicantType) {
  return applicantType === "student" ? 10 : 20
}

function createSpecialRequestId() {
  const datePart = new Date().toISOString().slice(0, 10).replace(/-/g, "")
  const randomPart = crypto.randomBytes(6).toString("base64url").replace(/[^A-Z0-9]/gi, "").slice(0, 10).toUpperCase()
  return `HCN10-${datePart}-${randomPart}`
}

function createApiKey() {
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
  let suffix = ""
  for (let index = 0; index < 16; index += 1) {
    suffix += alphabet[crypto.randomInt(alphabet.length)]
  }
  return `horusNN-${suffix}`
}

function hashSecret(secret) {
  return crypto.createHash("sha256").update(`${secret}:${apiKeyPepper}`).digest("hex")
}

function encryptSecret(secret) {
  const iv = crypto.randomBytes(12)
  const cipher = crypto.createCipheriv("aes-256-gcm", apiKeyEncryptionKey, iv)
  const encrypted = Buffer.concat([cipher.update(secret, "utf8"), cipher.final()])
  const tag = cipher.getAuthTag()
  return `v1.${iv.toString("base64url")}.${tag.toString("base64url")}.${encrypted.toString("base64url")}`
}

function redactApiKeys(value) {
  return String(value || "").replace(/horusNN-[A-Za-z0-9]{16}/g, "horusNN-[REDACTED]")
}

function safeSecretEquals(left, right) {
  const leftBuffer = Buffer.from(String(left || ""))
  const rightBuffer = Buffer.from(String(right || ""))
  return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer)
}

function requireAdmin(req, res, next) {
  const token = req.get("authorization")?.replace(/^Bearer\s+/i, "")
  if (!safeSecretEquals(token, adminToken)) {
    res.status(401).json({ error: "Unauthorized" })
    return
  }
  next()
}

async function authenticateApiKey(req, res, next) {
  const raw = req.get("x-api-key") || req.get("authorization")?.replace(/^Bearer\s+/i, "")
  if (!raw || !/^horusNN-[A-Za-z0-9]{16}$/.test(raw)) {
    res.status(401).json({ error: "Access Token is required" })
    return
  }

  const keyHash = hashSecret(raw)
  const result = await pool.query(
    `SELECT ak.id, ak.key_prefix, ak.special_request_id, ak.rpm_limit, ak.metadata, ak.expires_at,
       ar.status AS request_status
     FROM api_keys ak
     JOIN access_requests ar ON ar.special_request_id = ak.special_request_id
     WHERE ak.key_hash = $1
       AND ak.status = 'active'
       AND (ak.expires_at IS NULL OR ak.expires_at > now())`,
    [keyHash],
  )

  if (result.rowCount === 0) {
    res.status(401).json({ error: "Invalid Access Token" })
    return
  }

  if (result.rows[0].request_status !== "approved") {
    res.status(403).json({ error: "API access is disabled." })
    return
  }

  req.apiKey = { ...result.rows[0], keyHash }
  await pool.query("UPDATE api_keys SET last_used_at = now() WHERE id = $1", [req.apiKey.id])
  next()
}

async function enforceApiKeyRateLimit(req, res, next) {
  const limit = Number(req.apiKey.rpm_limit || defaultKeyRateLimitPerMinute)
  const client = await pool.connect()
  try {
    await client.query("BEGIN")
    await client.query("SELECT pg_advisory_xact_lock($1::bigint)", [req.apiKey.id])
    await client.query(
      "DELETE FROM api_request_rate_events WHERE api_key_id = $1 AND created_at < now() - interval '10 minutes'",
      [req.apiKey.id],
    )
    const recent = await client.query(
      `SELECT count(*)::int AS count, min(created_at) AS oldest
       FROM api_request_rate_events
       WHERE api_key_id = $1 AND created_at > now() - interval '60 seconds'`,
      [req.apiKey.id],
    )
    const count = Number(recent.rows[0]?.count || 0)
    const oldest = recent.rows[0]?.oldest ? new Date(recent.rows[0].oldest).getTime() : Date.now()
    const resetAt = oldest + 60_000

    res.setHeader("X-RateLimit-Limit", String(limit))
    res.setHeader("X-RateLimit-Remaining", String(Math.max(0, limit - count - 1)))
    res.setHeader("X-RateLimit-Reset", String(Math.ceil(resetAt / 1000)))

    if (count >= limit) {
      const retryAfter = Math.max(1, Math.ceil((resetAt - Date.now()) / 1000))
      await client.query(
        `INSERT INTO api_usage_events (
           special_request_id, api_key_prefix, event_type, request_path, status_code, metadata
         ) VALUES ($1, $2, 'rate_limit_exceeded', $3, 429, $4)`,
        [req.apiKey.special_request_id, req.apiKey.key_prefix, req.originalUrl || req.path, { limit, retryAfter }],
      )
      await client.query("COMMIT")
      res.setHeader("X-RateLimit-Remaining", "0")
      res.setHeader("Retry-After", String(retryAfter))
      res.status(429).json({
        error: `Rate limit exceeded. This Access Token is limited to ${limit} requests per minute.`,
        retryAfterSeconds: retryAfter,
      })
      return
    }

    await client.query(
      "INSERT INTO api_request_rate_events (api_key_id, special_request_id) VALUES ($1, $2)",
      [req.apiKey.id, req.apiKey.special_request_id],
    )
    await client.query("COMMIT")
    next()
  } catch (error) {
    await client.query("ROLLBACK")
    next(error)
  } finally {
    client.release()
  }
}

function publicRequestFingerprint(req) {
  const forwarded = String(req.get("cf-connecting-ip") || req.get("x-forwarded-for") || "")
    .split(",")[0]
    .trim()
  const address = forwarded || req.socket.remoteAddress || "unknown"
  return hashSecret(`public-request:${address}`)
}

function enforcePublicRateLimit(eventType, limit, windowSeconds) {
  return async (req, res, next) => {
    const fingerprint = publicRequestFingerprint(req)
    const advisoryKey = BigInt(`0x${fingerprint.slice(0, 15)}`).toString()
    const client = await pool.connect()
    try {
      await client.query("BEGIN")
      await client.query("SELECT pg_advisory_xact_lock($1::bigint)", [advisoryKey])
      await client.query(
        "DELETE FROM public_request_rate_events WHERE fingerprint = $1 AND created_at < now() - interval '24 hours'",
        [fingerprint],
      )
      const recent = await client.query(
        `SELECT count(*)::int AS count, min(created_at) AS oldest
         FROM public_request_rate_events
         WHERE fingerprint = $1 AND event_type = $2
           AND created_at > now() - make_interval(secs => $3)`,
        [fingerprint, eventType, windowSeconds],
      )
      const count = Number(recent.rows[0]?.count || 0)
      if (count >= limit) {
        const oldest = recent.rows[0]?.oldest ? new Date(recent.rows[0].oldest).getTime() : Date.now()
        const retryAfter = Math.max(1, Math.ceil((oldest + windowSeconds * 1000 - Date.now()) / 1000))
        await client.query("COMMIT")
        res.setHeader("Retry-After", String(retryAfter))
        res.status(429).json({ error: "Too many requests. Try again later." })
        return
      }
      await client.query(
        "INSERT INTO public_request_rate_events (fingerprint, event_type) VALUES ($1, $2)",
        [fingerprint, eventType],
      )
      await client.query("COMMIT")
      next()
    } catch (error) {
      await client.query("ROLLBACK")
      next(error)
    } finally {
      client.release()
    }
  }
}

async function enforceApiKeyConcurrency(req, res, next) {
  const leaseId = crypto.randomUUID()
  const client = await pool.connect()
  try {
    await client.query("BEGIN")
    await client.query("SELECT pg_advisory_xact_lock($1::bigint)", [req.apiKey.id])
    await client.query("DELETE FROM api_inflight_requests WHERE api_key_id = $1 AND expires_at <= now()", [req.apiKey.id])
    const active = await client.query(
      "SELECT count(*)::int AS count FROM api_inflight_requests WHERE api_key_id = $1",
      [req.apiKey.id],
    )
    if (Number(active.rows[0]?.count || 0) >= maxConcurrentRequestsPerKey) {
      await client.query("COMMIT")
      res.setHeader("Retry-After", "1")
      res.status(429).json({
        error: `Too many concurrent requests. This Access Token allows ${maxConcurrentRequestsPerKey} at a time.`,
      })
      return
    }
    await client.query(
      "INSERT INTO api_inflight_requests (id, api_key_id, expires_at) VALUES ($1, $2, now() + interval '3 minutes')",
      [leaseId, req.apiKey.id],
    )
    await client.query("COMMIT")

    let released = false
    const release = () => {
      if (released) return
      released = true
      pool.query("DELETE FROM api_inflight_requests WHERE id = $1", [leaseId]).catch((error) => {
        console.error("Failed to release API concurrency lease", error)
      })
    }
    res.once("finish", release)
    res.once("close", release)
    next()
  } catch (error) {
    await client.query("ROLLBACK")
    next(error)
  } finally {
    client.release()
  }
}

function canSendMail() {
  return Boolean(process.env.SMTP_HOST && process.env.SMTP_USER && process.env.SMTP_PASS && process.env.SMTP_FROM_EMAIL)
}

function baseEmailHtml(title, bodyHtml) {
  const logo = fs.existsSync(logoPath)
    ? `<img src="cid:tokenai-logo" alt="TokenAI" style="display:block;width:180px;height:auto;margin:0 auto 28px;" />`
    : `<div style="font-size:28px;font-weight:700;color:#101828;margin-bottom:28px;">TokenAI</div>`

  return `<!doctype html>
<html>
  <body style="margin:0;background:#f8fafc;font-family:Arial,Helvetica,sans-serif;color:#0f172a;">
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f8fafc;padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:640px;background:#ffffff;border:1px solid #e2e8f0;">
            <tr>
              <td style="padding:36px 36px 18px;text-align:center;">${logo}</td>
            </tr>
            <tr>
              <td style="padding:0 36px 36px;">
                <h1 style="margin:0 0 20px;font-size:28px;line-height:1.2;font-weight:500;color:#020617;">${title}</h1>
                <div style="font-size:15px;line-height:1.75;color:#334155;">${bodyHtml}</div>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>`
}

function paragraph(text) {
  return `<p style="margin:0 0 16px;">${String(text).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]))}</p>`
}

const mailTemplates = {
  requestReceived({ firstName, specialRequestId }) {
    const subject = "Horus Cyper Nano 1.0 BETA Early Access Request Received"
    const bodyText = [
      `Hi ${firstName},`,
      "",
      "Your Horus Cyper Nano 1.0 BETA Early Access request has been received.",
      "",
      `Special Request ID: ${specialRequestId}`,
      "Keep this ID private and easy to find. You can use it to follow up on your application status, reply with your social post URL, send corrections, or contact TokenAI support about complaints or review questions.",
      "",
      "To qualify for an Access Token, publish a public LinkedIn or Facebook post that clearly mentions TokenAI and Horus Cyper Nano 1.0 BETA, includes #HorusCyper, and states that you applied for Early Access as a developer, researcher, or student.",
      "",
      `Reply to this email with your public post URL and your Special Request ID: ${specialRequestId}`,
      "",
      "After TokenAI receives your qualifying public post reply, the approval review takes from 2 hours up to 48 hours.",
      "If you need support, include your Special Request ID when contacting support@tokenai.llc.",
      "",
      "TokenAI",
    ].join("\n")
    const bodyHtml = [
      paragraph(`Hi ${firstName},`),
      paragraph("Your Horus Cyper Nano 1.0 BETA Early Access request has been received."),
      paragraph(`Special Request ID: ${specialRequestId}`),
      paragraph("Keep this ID private and easy to find. You can use it to follow up on your application status, reply with your social post URL, send corrections, or contact TokenAI support about complaints or review questions."),
      paragraph("To qualify for an Access Token, publish a public LinkedIn or Facebook post that clearly mentions TokenAI and Horus Cyper Nano 1.0 BETA, includes #HorusCyper, and states that you applied for Early Access as a developer, researcher, or student."),
      paragraph(`Reply to this email with your public post URL and your Special Request ID: ${specialRequestId}`),
      paragraph("After TokenAI receives your qualifying public post reply, the approval review takes from 2 hours up to 48 hours."),
      paragraph("If you need support, include your Special Request ID when contacting support@tokenai.llc."),
    ].join("")
    return { subject, bodyText, bodyHtml: baseEmailHtml(subject, bodyHtml) }
  },
  approved({ firstName, specialRequestId, apiKey, rpmLimit, neuralNodeExample }) {
    const subject = "Horus Cyper Nano 1.0 BETA Access Approved"
    const bodyText = [
      `Hi ${firstName},`,
      "",
      "Your Horus Cyper Nano 1.0 BETA Early Access request has been approved.",
      "",
      `Special Request ID: ${specialRequestId}`,
      `Access Token: ${apiKey}`,
      `Rate limit: ${rpmLimit} requests per minute`,
      "",
      "NeuralNode Framework example:",
      neuralNodeExample,
      "",
      "Keep this Access Token private. TokenAI may revoke access if the token is shared or used outside the responsible use terms.",
      "",
      "TokenAI",
    ].join("\n")
    const bodyHtml = [
      paragraph(`Hi ${firstName},`),
      paragraph("Your Horus Cyper Nano 1.0 BETA Early Access request has been approved."),
      paragraph(`Special Request ID: ${specialRequestId}`),
      paragraph(`Access Token: ${apiKey}`),
      paragraph(`Rate limit: ${rpmLimit} requests per minute`),
      `<pre style="white-space:pre-wrap;background:#f8fafc;border:1px solid #e2e8f0;padding:14px;color:#0f172a;">${String(neuralNodeExample).replace(/[&<>]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[char]))}</pre>`,
      paragraph("Keep this Access Token private. TokenAI may revoke access if the token is shared or used outside the responsible use terms."),
    ].join("")
    return { subject, bodyText, bodyHtml: baseEmailHtml(subject, bodyHtml) }
  },
  accessDisabled({ firstName, reason }) {
    const subject = "Horus Cyper Nano 1.0 BETA API Access Disabled"
    const bodyText = [
      `Hi ${firstName},`,
      "",
      "Your Horus Cyper Nano 1.0 BETA Early Access API has been permanently disabled for violating the responsible use terms.",
      "",
      `Reason: ${reason || "Prohibited Knowledge Distillation or model-output training-data generation was confirmed during manual review."}`,
      "",
      "TokenAI",
    ].join("\n")
    const bodyHtml = [
      paragraph(`Hi ${firstName},`),
      paragraph("Your Horus Cyper Nano 1.0 BETA Early Access API has been permanently disabled for violating the responsible use terms."),
      paragraph(`Reason: ${reason || "Prohibited Knowledge Distillation or model-output training-data generation was confirmed during manual review."}`),
    ].join("")
    return { subject, bodyText, bodyHtml: baseEmailHtml(subject, bodyHtml) }
  },
  distillationWarning({ firstName, specialRequestId, detectedAt }) {
    const subject = "Warning: Prohibited Horus Cyper Nano API Activity Detected"
    const bodyText = [
      `Hi ${firstName},`,
      "",
      "TokenAI detected an API request that may involve Knowledge Distillation or using Horus Cyper Nano model outputs to create datasets or training data.",
      "",
      `Special Request ID: ${specialRequestId}`,
      `Detected at: ${detectedAt}`,
      "",
      "Your Access Token has not been disabled. Stop this activity immediately. TokenAI will review the request manually, and confirmed prohibited use may result in permanent Access Token disablement.",
      "",
      "Cybersecurity research and security testing are not prohibited by this warning. The restriction applies specifically to Knowledge Distillation and model-output data generation for datasets or model training.",
      "",
      "If you believe this detection is incorrect, reply to this email or contact support@tokenai.llc and include your Special Request ID.",
      "",
      "TokenAI",
    ].join("\n")
    const bodyHtml = [
      paragraph(`Hi ${firstName},`),
      paragraph("TokenAI detected an API request that may involve Knowledge Distillation or using Horus Cyper Nano model outputs to create datasets or training data."),
      paragraph(`Special Request ID: ${specialRequestId}`),
      paragraph(`Detected at: ${detectedAt}`),
      paragraph("Your Access Token has not been disabled. Stop this activity immediately. TokenAI will review the request manually, and confirmed prohibited use may result in permanent Access Token disablement."),
      paragraph("Cybersecurity research and security testing are not prohibited by this warning. The restriction applies specifically to Knowledge Distillation and model-output data generation for datasets or model training."),
      paragraph("If you believe this detection is incorrect, reply to this email or contact support@tokenai.llc and include your Special Request ID."),
    ].join("")
    return { subject, bodyText, bodyHtml: baseEmailHtml(subject, bodyHtml) }
  },
  socialPostReminder({ firstName, specialRequestId }) {
    const subject = "Action Required: Horus Cyper Nano 1.0 BETA Social Post"
    const bodyText = [
      `Hi ${firstName},`,
      "",
      `Your Special Request ID is ${specialRequestId}.`,
      "Please reply with a public LinkedIn or Facebook post URL that mentions TokenAI, Horus Cyper Nano 1.0 BETA, and #HorusCyper.",
      "",
      "TokenAI",
    ].join("\n")
    const bodyHtml = [
      paragraph(`Hi ${firstName},`),
      paragraph(`Your Special Request ID is ${specialRequestId}.`),
      paragraph("Please reply with a public LinkedIn or Facebook post URL that mentions TokenAI, Horus Cyper Nano 1.0 BETA, and #HorusCyper."),
    ].join("")
    return { subject, bodyText, bodyHtml: baseEmailHtml(subject, bodyHtml) }
  },
}

async function sendOrQueueEmail(client, { specialRequestId, recipient, subject, bodyText, bodyHtml }) {
  const insert = await client.query(
    `INSERT INTO email_outbox (special_request_id, recipient, subject, body_text, status)
     VALUES ($1, $2, $3, $4, $5)
     RETURNING id`,
    [specialRequestId, recipient, subject, redactApiKeys(bodyText), canSendMail() ? "pending" : "pending_smtp_configuration"],
  )

  if (!canSendMail()) return { id: insert.rows[0].id, status: "pending_smtp_configuration" }

  try {
    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: Number(process.env.SMTP_PORT || 465),
      secure: String(process.env.SMTP_SECURE || "true") === "true",
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
      connectionTimeout: 10000,
      greetingTimeout: 10000,
      socketTimeout: 20000,
    })

    const sent = await transporter.sendMail({
      from: `"${process.env.SMTP_FROM_NAME || "TokenAI"}" <${process.env.SMTP_FROM_EMAIL}>`,
      to: recipient,
      subject,
      text: bodyText,
      html: bodyHtml,
      replyTo: process.env.REPLY_TO_EMAIL || process.env.SMTP_FROM_EMAIL,
      attachments: fs.existsSync(logoPath)
        ? [{ filename: "tokenai-logo.png", path: logoPath, cid: "tokenai-logo" }]
        : [],
    })

    await client.query(
      `UPDATE email_outbox SET status = 'sent', provider_message_id = $1, sent_at = now() WHERE id = $2`,
      [sent.messageId, insert.rows[0].id],
    )
    return { id: insert.rows[0].id, status: "sent" }
  } catch (error) {
    await client.query("UPDATE email_outbox SET status = 'failed', error_message = $1 WHERE id = $2", [
      error.message,
      insert.rows[0].id,
    ])
    return { id: insert.rows[0].id, status: "failed", error: error.message }
  }
}

app.get("/health", async (_req, res) => {
  const [db, queue] = await Promise.all([
    pool.query("SELECT now() AS now"),
    pool.query(
      `SELECT
         count(*) FILTER (WHERE status = 'running')::int AS running,
         count(*) FILTER (WHERE status = 'waiting')::int AS waiting
       FROM inference_queue_leases
       WHERE expires_at > now()`,
    ),
  ])
  res.json({
    ok: true,
    service: "tokenai-early-access-api",
    dbTime: db.rows[0].now,
    inference: {
      capacity: globalInferenceCapacity,
      running: queue.rows[0]?.running || 0,
      waiting: queue.rows[0]?.waiting || 0,
    },
  })
})

app.get("/v1/admin/operations", requireAdmin, async (_req, res) => {
  const [queue, usage, safety, backups] = await Promise.all([
    pool.query(
      `SELECT status, count(*)::int AS count,
              max(EXTRACT(EPOCH FROM (now() - queued_at)) * 1000)::bigint AS oldest_wait_ms
       FROM inference_queue_leases WHERE expires_at > now() GROUP BY status ORDER BY status`,
    ),
    pool.query(
      `SELECT count(*)::int AS requests,
              round(avg((metadata->>'latencyMs')::numeric))::int AS average_latency_ms,
              percentile_cont(0.95) WITHIN GROUP (ORDER BY (metadata->>'latencyMs')::numeric)::int AS p95_latency_ms,
              count(*) FILTER (WHERE status_code >= 500)::int AS server_errors
       FROM api_usage_events
       WHERE created_at > now() - interval '1 hour'
         AND metadata ? 'latencyMs'`,
    ),
    pool.query(
      `SELECT severity, action, count(*)::int AS count
       FROM api_safety_events
       WHERE created_at > now() - interval '24 hours'
       GROUP BY severity, action ORDER BY severity, action`,
    ),
    pool.query("SELECT now() AS checked_at"),
  ])
  res.json({
    inferenceCapacity: globalInferenceCapacity,
    queue: queue.rows,
    usageLastHour: usage.rows[0],
    safetyLast24Hours: safety.rows,
    checkedAt: backups.rows[0].checked_at,
  })
})

app.get("/v1/admin/mail-status", requireAdmin, (_req, res) => {
  res.json({
    ready: canSendMail(),
    smtpHost: Boolean(process.env.SMTP_HOST),
    smtpUser: Boolean(process.env.SMTP_USER),
    smtpPass: Boolean(process.env.SMTP_PASS),
    smtpFromEmail: Boolean(process.env.SMTP_FROM_EMAIL),
  })
})

app.get("/v1/admin/overview", requireAdmin, async (_req, res) => {
  const [requests, applicantTypes, users, conversations, messages, emails, distillationAlerts] = await Promise.all([
    pool.query("SELECT status, count(*)::int AS count FROM access_requests GROUP BY status ORDER BY status"),
    pool.query("SELECT applicant_type, count(*)::int AS count FROM access_requests GROUP BY applicant_type ORDER BY applicant_type"),
    pool.query("SELECT count(DISTINCT special_request_id)::int AS count FROM api_keys"),
    pool.query("SELECT count(*)::int AS count FROM api_conversations"),
    pool.query("SELECT count(*)::int AS count FROM conversation_messages"),
    pool.query("SELECT status, count(*)::int AS count FROM email_outbox GROUP BY status ORDER BY status"),
    pool.query("SELECT count(*)::int AS count FROM api_safety_events WHERE review_status IN ('pending', 'warning_sent')"),
  ])

  res.json({
    requests: requests.rows,
    applicantTypes: applicantTypes.rows,
    users: users.rows[0]?.count || 0,
    conversations: conversations.rows[0]?.count || 0,
    messages: messages.rows[0]?.count || 0,
    emails: emails.rows,
    distillationAlerts: distillationAlerts.rows[0]?.count || 0,
  })
})

app.get("/v1/admin/distillation-alerts", requireAdmin, async (req, res) => {
  const limit = Math.min(Math.max(Number(req.query.limit || 100), 1), 500)
  const status = String(req.query.status || "").trim()
  const allowedStatuses = new Set(["pending", "warning_sent", "token_disabled", "dismissed"])
  const values = []
  const where = []
  if (status) {
    if (!allowedStatuses.has(status)) {
      res.status(400).json({ error: "Invalid review status." })
      return
    }
    values.push(status)
    where.push(`se.review_status = $${values.length}`)
  }
  values.push(limit)
  const result = await pool.query(
    `SELECT
       se.id, se.special_request_id, se.request_id, se.conversation_id, se.severity,
       se.action, se.categories, se.matched_rules, se.user_message, se.model_response,
       se.review_status, se.warning_sent_at, se.reviewed_at, se.reviewed_by, se.created_at,
       ar.full_name, ar.first_name, ar.second_name, ar.business_mail, ar.applicant_type,
       ar.company_name, ar.company_website, ar.job, ar.linkedin_profile_url, ar.github_profile_url,
       ar.portfolio_url, ar.country_or_region, ar.use_cases, ar.model, ar.status AS request_status,
       ar.social_post_status, ar.social_post_url, ar.source, ar.created_at AS applicant_created_at,
       ak.id AS api_key_id, ak.key_prefix, ak.status AS token_status, ak.rpm_limit
     FROM api_safety_events se
     LEFT JOIN access_requests ar ON ar.special_request_id = se.special_request_id
     LEFT JOIN api_keys ak ON ak.id = se.api_key_id
     ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
     ORDER BY se.created_at DESC
     LIMIT $${values.length}`,
    values,
  )
  res.json({ alerts: result.rows })
})

app.post("/v1/admin/distillation-alerts/:id/warn", requireAdmin, async (req, res) => {
  const alertId = Number(req.params.id)
  if (!Number.isSafeInteger(alertId) || alertId < 1) {
    res.status(400).json({ error: "Invalid alert ID." })
    return
  }
  const alert = await pool.query(
    `SELECT se.*, ar.first_name, ar.business_mail, ar.source
     FROM api_safety_events se
     LEFT JOIN access_requests ar ON ar.special_request_id = se.special_request_id
     WHERE se.id = $1`,
    [alertId],
  )
  if (alert.rowCount === 0) {
    res.status(404).json({ error: "Distillation alert not found." })
    return
  }
  const row = alert.rows[0]
  if (["token_disabled", "dismissed"].includes(row.review_status)) {
    res.status(409).json({ error: `This alert is already ${row.review_status}.` })
    return
  }
  if (!row.business_mail) {
    res.status(409).json({ error: "The applicant record is no longer available, so a warning email cannot be sent." })
    return
  }
  const emailTemplate = mailTemplates.distillationWarning({
    firstName: row.first_name || "there",
    specialRequestId: row.special_request_id,
    detectedAt: new Date(row.created_at).toISOString(),
  })
  const email = row.source === "production-readiness-self-test"
    ? { id: null, status: "skipped_test" }
    : await sendOrQueueEmail(pool, {
        specialRequestId: row.special_request_id,
        recipient: row.business_mail,
        subject: emailTemplate.subject,
        bodyText: emailTemplate.bodyText,
        bodyHtml: emailTemplate.bodyHtml,
      })
  if (!["sent", "skipped_test"].includes(email.status)) {
    res.status(502).json({ error: "The warning email was not delivered.", emailStatus: email.status })
    return
  }
  await pool.query(
    `UPDATE api_safety_events
     SET action = 'warning_sent', review_status = 'warning_sent', warning_sent_at = now(),
         warning_email_id = $2, reviewed_by = $3
     WHERE id = $1`,
    [alertId, email.id, String(req.body?.reviewedBy || "dashboard").trim().slice(0, 120)],
  )
  res.json({ alertId, reviewStatus: "warning_sent", emailStatus: email.status })
})

app.post("/v1/admin/distillation-alerts/:id/disable", requireAdmin, async (req, res) => {
  const alertId = Number(req.params.id)
  if (!Number.isSafeInteger(alertId) || alertId < 1) {
    res.status(400).json({ error: "Invalid alert ID." })
    return
  }
  const reviewedBy = String(req.body?.reviewedBy || "dashboard").trim().slice(0, 120)
  const reason = "Manual review confirmed prohibited Knowledge Distillation or model-output training-data generation."
  const client = await pool.connect()
  let user
  try {
    await client.query("BEGIN")
    const alert = await client.query(
      `SELECT se.*, ar.first_name, ar.business_mail, ar.source
       FROM api_safety_events se
       LEFT JOIN access_requests ar ON ar.special_request_id = se.special_request_id
       WHERE se.id = $1 FOR UPDATE OF se`,
      [alertId],
    )
    if (alert.rowCount === 0) {
      await client.query("ROLLBACK")
      res.status(404).json({ error: "Distillation alert not found." })
      return
    }
    user = alert.rows[0]
    await client.query(
      `UPDATE api_keys
       SET status = 'disabled', revoked_at = COALESCE(revoked_at, now()),
           safety_strikes = safety_strikes + 1, disabled_reason = $2,
           metadata = metadata || $3::jsonb
       WHERE id = $1`,
      [user.api_key_id, reason, JSON.stringify({ manuallyDisabledAt: new Date().toISOString(), distillationAlertId: alertId, reviewedBy })],
    )
    await client.query(
      "UPDATE access_requests SET status = 'disabled', updated_at = now() WHERE special_request_id = $1",
      [user.special_request_id],
    )
    await client.query("DELETE FROM api_request_rate_events WHERE api_key_id = $1", [user.api_key_id])
    await client.query("DELETE FROM api_inflight_requests WHERE api_key_id = $1", [user.api_key_id])
    await client.query("DELETE FROM inference_queue_leases WHERE api_key_id = $1", [user.api_key_id])
    await client.query(
      `UPDATE api_safety_events
       SET action = 'token_disabled', review_status = 'token_disabled', reviewed_at = now(), reviewed_by = $2
       WHERE id = $1`,
      [alertId, reviewedBy],
    )
    await client.query("COMMIT")
  } catch (error) {
    await client.query("ROLLBACK")
    throw error
  } finally {
    client.release()
  }
  let emailStatus = "not_sent"
  if (user?.business_mail && user.source !== "production-readiness-self-test") {
    const emailTemplate = mailTemplates.accessDisabled({ firstName: user.first_name || "there", reason })
    const email = await sendOrQueueEmail(pool, {
      specialRequestId: user.special_request_id,
      recipient: user.business_mail,
      subject: emailTemplate.subject,
      bodyText: emailTemplate.bodyText,
      bodyHtml: emailTemplate.bodyHtml,
    })
    emailStatus = email.status
  }
  res.json({ alertId, specialRequestId: user.special_request_id, reviewStatus: "token_disabled", emailStatus })
})

app.post("/v1/admin/distillation-alerts/:id/dismiss", requireAdmin, async (req, res) => {
  const alertId = Number(req.params.id)
  if (!Number.isSafeInteger(alertId) || alertId < 1) {
    res.status(400).json({ error: "Invalid alert ID." })
    return
  }
  const result = await pool.query(
    `UPDATE api_safety_events
     SET action = 'dismissed', review_status = 'dismissed', reviewed_at = now(), reviewed_by = $2
     WHERE id = $1 AND review_status <> 'token_disabled'
     RETURNING id`,
    [alertId, String(req.body?.reviewedBy || "dashboard").trim().slice(0, 120)],
  )
  if (result.rowCount === 0) {
    res.status(404).json({ error: "The alert was not found or its token is already disabled." })
    return
  }
  res.json({ alertId, reviewStatus: "dismissed" })
})

app.get("/v1/admin/access-requests", requireAdmin, async (req, res) => {
  const limit = Math.min(Math.max(Number(req.query.limit || 100), 1), 500)
  const status = String(req.query.status || "").trim()
  const q = String(req.query.q || "").trim()
  const values = []
  const where = []

  if (status) {
    values.push(status)
    where.push(`ar.status = $${values.length}`)
  }
  if (q) {
    values.push(`%${q.toLowerCase()}%`)
    where.push(`(
      lower(ar.special_request_id) LIKE $${values.length}
      OR lower(ar.full_name) LIKE $${values.length}
      OR lower(ar.business_mail) LIKE $${values.length}
      OR lower(ar.company_name) LIKE $${values.length}
    )`)
  }
  values.push(limit)

  const result = await pool.query(
    `SELECT
       ar.special_request_id, ar.first_name, ar.second_name, ar.full_name, ar.applicant_type, ar.rpm_limit,
       ar.company_name, ar.company_website, ar.job, ar.business_mail, ar.linkedin_profile_url, ar.github_profile_url,
       ar.portfolio_url,
       ar.country_or_region, ar.use_cases, ar.model, ar.status, ar.social_post_status, ar.social_post_url,
       ar.api_key_prefix, ar.reviewed_at, ar.reviewed_by, ar.created_at, ar.updated_at,
       COALESCE(json_agg(json_build_object(
         'id', ak.id, 'keyPrefix', ak.key_prefix, 'status', ak.status, 'rpmLimit', ak.rpm_limit,
         'createdAt', ak.created_at, 'lastUsedAt', ak.last_used_at, 'revokedAt', ak.revoked_at
       ) ORDER BY ak.created_at DESC) FILTER (WHERE ak.id IS NOT NULL), '[]'::json) AS api_keys,
       count(DISTINCT ac.id)::int AS conversation_count,
       count(DISTINCT eo.id)::int AS email_count
     FROM access_requests ar
     LEFT JOIN api_keys ak ON ak.special_request_id = ar.special_request_id
     LEFT JOIN api_conversations ac ON ac.special_request_id = ar.special_request_id
     LEFT JOIN email_outbox eo ON eo.special_request_id = ar.special_request_id
     ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
     GROUP BY ar.special_request_id
     ORDER BY ar.created_at DESC
     LIMIT $${values.length}`,
    values,
  )

  res.json({ requests: result.rows })
})

app.get("/v1/admin/users", requireAdmin, async (req, res) => {
  const limit = Math.min(Math.max(Number(req.query.limit || 100), 1), 500)
  const result = await pool.query(
    `SELECT
       ar.special_request_id, ar.full_name, ar.first_name, ar.applicant_type, ar.rpm_limit,
       ar.company_name, ar.business_mail, ar.country_or_region, ar.status AS request_status,
       ak.id AS api_key_id, ak.key_prefix, ak.status AS token_status, ak.rpm_limit AS token_rpm_limit,
       ak.created_at AS token_created_at, ak.last_used_at, ak.revoked_at,
       count(DISTINCT ac.id)::int AS conversation_count,
       count(cm.id)::int AS message_count,
       max(cm.created_at) AS last_message_at
     FROM access_requests ar
     LEFT JOIN api_keys ak ON ak.special_request_id = ar.special_request_id
     LEFT JOIN api_conversations ac ON ac.special_request_id = ar.special_request_id
     LEFT JOIN conversation_messages cm ON cm.conversation_id = ac.id
     GROUP BY ar.special_request_id, ak.id
     ORDER BY ar.created_at DESC
     LIMIT $1`,
    [limit],
  )

  res.json({ users: result.rows })
})

app.delete("/v1/admin/users/:specialRequestId", requireAdmin, async (req, res) => {
  const specialRequestId = String(req.params.specialRequestId || "").trim()
  const client = await pool.connect()

  try {
    await client.query("BEGIN")
    const request = await client.query(
      "SELECT special_request_id FROM access_requests WHERE special_request_id = $1 FOR UPDATE",
      [specialRequestId],
    )
    if (request.rowCount === 0) {
      await client.query("ROLLBACK")
      res.status(404).json({ error: "User was not found." })
      return
    }

    const retained = await client.query(
      `SELECT count(DISTINCT ac.id)::int AS conversations, count(cm.id)::int AS messages
       FROM api_conversations ac
       LEFT JOIN conversation_messages cm ON cm.conversation_id = ac.id
       WHERE ac.special_request_id = $1`,
      [specialRequestId],
    )
    await client.query(
      `UPDATE api_conversations
       SET metadata = metadata || $2::jsonb, updated_at = now()
       WHERE special_request_id = $1`,
      [specialRequestId, JSON.stringify({ ownerDeleted: true, ownerDeletedAt: new Date().toISOString() })],
    )
    await client.query("DELETE FROM access_requests WHERE special_request_id = $1", [specialRequestId])
    await client.query("COMMIT")

    res.json({
      deleted: true,
      specialRequestId,
      retainedConversations: retained.rows[0]?.conversations || 0,
      retainedMessages: retained.rows[0]?.messages || 0,
      emailCanRegisterAgain: true,
    })
  } catch (error) {
    await client.query("ROLLBACK")
    console.error(error)
    res.status(500).json({ error: "User could not be deleted." })
  } finally {
    client.release()
  }
})

app.get("/v1/admin/conversations", requireAdmin, async (req, res) => {
  const limit = Math.min(Math.max(Number(req.query.limit || 100), 1), 500)
  const specialRequestId = String(req.query.specialRequestId || "").trim()
  const values = []
  const where = []
  if (specialRequestId) {
    values.push(specialRequestId)
    where.push(`ac.special_request_id = $${values.length}`)
  }
  values.push(limit)

  const result = await pool.query(
    `SELECT
       ac.id, ac.special_request_id,
       COALESCE(ar.full_name, 'Deleted user') AS full_name,
       ar.business_mail, ar.applicant_type,
       (ar.special_request_id IS NULL) AS owner_deleted,
       ac.api_key_prefix, ac.user_reference, ac.client_name, ac.model, ac.metadata,
       ac.created_at, ac.updated_at,
       count(cm.id)::int AS message_count,
       max(cm.created_at) AS last_message_at,
       (array_agg(cm.content ORDER BY cm.created_at DESC))[1] AS latest_message
     FROM api_conversations ac
     LEFT JOIN access_requests ar ON ar.special_request_id = ac.special_request_id
     LEFT JOIN conversation_messages cm ON cm.conversation_id = ac.id
     ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
     GROUP BY ac.id, ar.special_request_id
     ORDER BY ac.updated_at DESC
     LIMIT $${values.length}`,
    values,
  )

  res.json({ conversations: result.rows })
})

app.get("/v1/admin/conversations/:conversationId/messages", requireAdmin, async (req, res) => {
  const result = await pool.query(
    `SELECT cm.id, cm.conversation_id, cm.role, cm.content, cm.token_count, cm.latency_ms, cm.created_at, cm.metadata
     FROM conversation_messages cm
     WHERE cm.conversation_id = $1
     ORDER BY cm.created_at ASC, cm.id ASC`,
    [req.params.conversationId],
  )

  res.json({ messages: result.rows })
})

app.get("/v1/admin/email-outbox", requireAdmin, async (req, res) => {
  const limit = Math.min(Math.max(Number(req.query.limit || 100), 1), 500)
  const result = await pool.query(
    `SELECT eo.id, eo.special_request_id, ar.full_name, eo.recipient, eo.subject, eo.status,
       eo.provider_message_id, eo.error_message, eo.created_at, eo.sent_at
     FROM email_outbox eo
     LEFT JOIN access_requests ar ON ar.special_request_id = eo.special_request_id
     ORDER BY eo.created_at DESC
     LIMIT $1`,
    [limit],
  )

  res.json({ emails: result.rows })
})

app.post("/v1/admin/email/send", requireAdmin, async (req, res) => {
  const recipient = String(req.body?.recipient || "").trim()
  const subject = String(req.body?.subject || "").trim()
  const bodyText = String(req.body?.bodyText || "").trim()
  const specialRequestId = req.body?.specialRequestId ? String(req.body.specialRequestId).trim() : null

  if (!recipient || !subject || !bodyText) {
    res.status(400).json({ error: "Recipient, subject, and bodyText are required." })
    return
  }

  const client = await pool.connect()
  try {
    const email = await sendOrQueueEmail(client, {
      specialRequestId,
      recipient,
      subject,
      bodyText,
      bodyHtml: baseEmailHtml(subject, bodyText.split(/\n{2,}/).map((block) => paragraph(block)).join("")),
    })
    res.json({ recipient, emailStatus: email.status })
  } finally {
    client.release()
  }
})

app.patch("/v1/admin/api-keys/:id/rpm-limit", requireAdmin, async (req, res) => {
  const rpmLimit = Number(req.body?.rpmLimit)
  if (!Number.isInteger(rpmLimit) || rpmLimit < 1 || rpmLimit > 600) {
    res.status(400).json({ error: "rpmLimit must be an integer between 1 and 600." })
    return
  }

  const client = await pool.connect()
  try {
    await client.query("BEGIN")
    const result = await client.query(
      `UPDATE api_keys
       SET rpm_limit = $2
       WHERE id = $1
       RETURNING id, special_request_id, key_prefix, status, rpm_limit`,
      [req.params.id, rpmLimit],
    )
    if (result.rowCount === 0) {
      await client.query("ROLLBACK")
      res.status(404).json({ error: "API key not found." })
      return
    }
    await client.query("UPDATE access_requests SET rpm_limit = $2, updated_at = now() WHERE special_request_id = $1", [
      result.rows[0].special_request_id,
      rpmLimit,
    ])
    await client.query("DELETE FROM api_request_rate_events WHERE api_key_id = $1", [req.params.id])
    await client.query("COMMIT")
    res.json({ apiKey: result.rows[0] })
  } catch (error) {
    await client.query("ROLLBACK")
    throw error
  } finally {
    client.release()
  }
})

app.post("/v1/admin/api-keys/:id/disable", requireAdmin, async (req, res) => {
  const result = await pool.query(
    `UPDATE api_keys
     SET status = 'disabled', revoked_at = now()
     WHERE id = $1
     RETURNING id, special_request_id, key_prefix, status, rpm_limit, revoked_at`,
    [req.params.id],
  )

  if (result.rowCount === 0) {
    res.status(404).json({ error: "API key not found." })
    return
  }

  await pool.query("DELETE FROM api_request_rate_events WHERE api_key_id = $1", [req.params.id])
  res.json({ apiKey: result.rows[0] })
})

app.post("/v1/admin/api-keys/:id/enable", requireAdmin, async (req, res) => {
  const result = await pool.query(
    `UPDATE api_keys
     SET status = 'active', revoked_at = null
     WHERE id = $1
       AND EXISTS (
         SELECT 1 FROM access_requests ar
         WHERE ar.special_request_id = api_keys.special_request_id AND ar.status = 'approved'
       )
     RETURNING id, special_request_id, key_prefix, status, rpm_limit, revoked_at`,
    [req.params.id],
  )

  if (result.rowCount === 0) {
    res.status(409).json({ error: "The API key was not found or its access request is permanently disabled." })
    return
  }

  await pool.query("DELETE FROM api_request_rate_events WHERE api_key_id = $1", [req.params.id])
  res.json({ apiKey: result.rows[0] })
})

app.get("/v1/access-requests/check-email", enforcePublicRateLimit("email_check", 60, 60), async (req, res) => {
  const email = String(req.query.email || "").trim().toLowerCase()
  const applicantType = String(req.query.applicantType || "developer_researcher").trim()
  if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    res.status(400).json({ error: "A valid email is required." })
    return
  }

  if (freeEmailDomains.has(email.split("@")[1])) {
    res.status(400).json({ error: "Free email domains are not accepted." })
    return
  }

  if (applicantType === "student" && !isEducationalEmail(email)) {
    res.status(400).json({ error: "A university or educational email address is required for student applicants." })
    return
  }

  if (!(await hasDeliverableMailDomain(email))) {
    res.status(400).json({ error: "The email domain cannot receive mail." })
    return
  }

  const result = await pool.query("SELECT 1 FROM access_requests WHERE lower(business_mail) = $1 LIMIT 1", [email])
  res.json({ exists: result.rowCount > 0 })
})

app.post("/v1/access-requests", enforcePublicRateLimit("access_request", 10, 3600), async (req, res) => {
  const parsed = accessRequestSchema.safeParse(req.body)
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid request", issues: parsed.error.issues })
    return
  }

  const data = parsed.data
  if (!(await hasDeliverableMailDomain(data.businessMail))) {
    res.status(400).json({ error: "The email domain cannot receive mail." })
    return
  }
  const profileValidationError = await validateProfileLinks(data)
  if (profileValidationError) {
    res.status(400).json({ error: profileValidationError })
    return
  }

  const specialRequestId = createSpecialRequestId()
  const fullName = `${data.firstName} ${data.secondName}`
  const rpmLimit = getDefaultRpmLimit(data.applicantType)
  const client = await pool.connect()

  try {
    await client.query("BEGIN")
    await client.query(
      `INSERT INTO access_requests (
        special_request_id, first_name, second_name, full_name, applicant_type, rpm_limit, company_name, company_website, job,
        business_mail, linkedin_profile_url, github_profile_url, portfolio_url, country_or_region, use_cases,
        responsible_use_accepted, responsible_use_accepted_at, metadata
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, true, now(), $16)`,
      [
        specialRequestId,
        data.firstName,
        data.secondName,
        fullName,
        data.applicantType,
        rpmLimit,
        data.companyName,
        data.companyWebsite,
        data.job,
        data.businessMail,
        data.linkedinProfileUrl,
        data.githubProfileUrl,
        data.portfolioUrl,
        data.countryOrRegion,
        data.useCases,
        {
          applicantType: data.applicantType,
          rpmLimit,
          approvalRequirements: {
            socialPostRequired: true,
            acceptedPlatforms: ["LinkedIn", "Facebook"],
            replyMustInclude: ["Special Request ID", "public post URL"],
            requiredMentions: ["TokenAI", "Horus Cyper Nano 1.0 BETA"],
            officialAccountMentionRequired: true,
            requiredHashtags: ["#HorusCyper"],
            realUseCaseRequired: true,
            postMustRemainPublic: true,
          },
        },
      ],
    )

    await client.query("COMMIT")
    const emailTemplate = mailTemplates.requestReceived({ firstName: data.firstName, specialRequestId })
    let emailStatus = "failed_to_queue"
    try {
      const email = await sendOrQueueEmail(pool, {
        specialRequestId,
        recipient: data.businessMail,
        subject: emailTemplate.subject,
        bodyText: emailTemplate.bodyText,
        bodyHtml: emailTemplate.bodyHtml,
      })
      emailStatus = email.status
    } catch (emailError) {
      console.error("Registration email could not be queued", emailError)
    }
    res.status(201).json({ specialRequestId, emailStatus })
  } catch (error) {
    await client.query("ROLLBACK")
    if (error.code === "23505") {
      res.status(409).json({ error: "This business email already has an access request." })
      return
    }
    console.error(error)
    res.status(500).json({ error: "Request could not be stored." })
  } finally {
    client.release()
  }
})

app.patch("/v1/admin/access-requests/:specialRequestId/social-post", requireAdmin, async (req, res) => {
  const parsed = socialPostReviewSchema.safeParse(req.body)
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid social-post review.", issues: parsed.error.issues })
    return
  }

  const result = await pool.query(
    `UPDATE access_requests
     SET social_post_url = $2, social_post_status = $3,
         metadata = metadata || $4::jsonb, updated_at = now()
     WHERE special_request_id = $1
     RETURNING special_request_id, social_post_url, social_post_status`,
    [
      req.params.specialRequestId,
      parsed.data.socialPostUrl,
      parsed.data.status,
      JSON.stringify({ socialPostReviewedBy: parsed.data.reviewedBy, socialPostReviewedAt: new Date().toISOString() }),
    ],
  )
  if (result.rowCount === 0) {
    res.status(404).json({ error: "Access request not found." })
    return
  }
  res.json({ request: result.rows[0] })
})

app.post("/v1/admin/access-requests/:specialRequestId/approve", requireAdmin, async (req, res) => {
  const specialRequestId = req.params.specialRequestId
  const reviewedBy = String(req.body?.reviewedBy || "dashboard").trim().slice(0, 120)
  const client = await pool.connect()
  let approval

  try {
    await client.query("BEGIN")
    const request = await client.query("SELECT * FROM access_requests WHERE special_request_id = $1 FOR UPDATE", [
      specialRequestId,
    ])
    if (request.rowCount === 0) {
      await client.query("ROLLBACK")
      res.status(404).json({ error: "Access request not found" })
      return
    }

    const row = request.rows[0]
    if (row.social_post_status !== "verified" || !row.social_post_url) {
      await client.query("ROLLBACK")
      res.status(409).json({ error: "Verify the applicant's public social post before approval." })
      return
    }
    if (row.status !== "pending") {
      await client.query("ROLLBACK")
      res.status(409).json({ error: `Only pending requests can be approved. Current status: ${row.status}.` })
      return
    }

    const existingKey = await client.query(
      "SELECT id, key_prefix, status FROM api_keys WHERE special_request_id = $1 LIMIT 1 FOR UPDATE",
      [specialRequestId],
    )
    if (existingKey.rowCount > 0) {
      await client.query("ROLLBACK")
      res.status(409).json({
        error: "This user already has an Access Token. Only one Access Token is allowed per user.",
        keyPrefix: existingKey.rows[0].key_prefix,
        status: existingKey.rows[0].status,
      })
      return
    }

    const rpmLimit = Number(row.rpm_limit || getDefaultRpmLimit(row.applicant_type))
    if (!Number.isInteger(rpmLimit) || rpmLimit < 1 || rpmLimit > 600) {
      throw new Error("The request has an invalid RPM limit.")
    }

    let apiKey
    let keyPrefix
    let keyHash
    for (let attempt = 0; attempt < 5; attempt += 1) {
      apiKey = createApiKey()
      keyPrefix = `${apiKey.slice(0, 12)}...`
      keyHash = hashSecret(apiKey)
      try {
        await client.query(
          `INSERT INTO api_keys (special_request_id, key_prefix, key_hash, key_ciphertext, rpm_limit)
           VALUES ($1, $2, $3, $4, $5)`,
          [specialRequestId, keyPrefix, keyHash, encryptSecret(apiKey), rpmLimit],
        )
        break
      } catch (error) {
        if (error.code !== "23505" || error.constraint !== "api_keys_key_hash_key" || attempt === 4) throw error
      }
    }

    await client.query(
      `UPDATE access_requests
       SET status = 'approved', rpm_limit = $3, api_key_prefix = $2, api_key_hash = NULL,
           reviewed_at = now(), reviewed_by = $4, updated_at = now()
       WHERE special_request_id = $1`,
      [specialRequestId, keyPrefix, rpmLimit, reviewedBy],
    )
    await client.query("COMMIT")
    approval = { row, apiKey, keyPrefix, rpmLimit }
  } catch (error) {
    await client.query("ROLLBACK")
    console.error(error)
    res.status(500).json({ error: "Approval failed." })
    return
  } finally {
    client.release()
  }

  const neuralNodeExample = String(req.body?.neuralNodeExample || [
    "import neuralnode as nn",
    "",
    "ai = nn.NeuralNode(",
    "    provider=\"horus-cyper-nano\",",
    "    model=\"horus-cyper-nano-1.0-beta\",",
    `    api_key="${approval.apiKey}",`,
    ")",
    "",
    "response = ai.chat(\"Review this code for defensive cybersecurity issues.\")",
    "print(response)",
  ].join("\n"))
  const emailTemplate = mailTemplates.approved({
    firstName: approval.row.first_name,
    specialRequestId,
    apiKey: approval.apiKey,
    rpmLimit: approval.rpmLimit,
    neuralNodeExample,
  })
  let emailStatus = "failed_to_queue"
  try {
    const email = await sendOrQueueEmail(pool, {
      specialRequestId,
      recipient: approval.row.business_mail,
      subject: emailTemplate.subject,
      bodyText: emailTemplate.bodyText,
      bodyHtml: emailTemplate.bodyHtml,
    })
    emailStatus = email.status
  } catch (emailError) {
    console.error("Approval email could not be queued", emailError)
  }
  res.json({ specialRequestId, keyPrefix: approval.keyPrefix, emailStatus })
})

app.post("/v1/admin/access-requests/:specialRequestId/disable", requireAdmin, async (req, res) => {
  const specialRequestId = req.params.specialRequestId
  const reason = String(req.body?.reason || "Access was disabled after a manual policy review.")
  const client = await pool.connect()
  let disabledUser

  try {
    await client.query("BEGIN")
    const request = await client.query("SELECT * FROM access_requests WHERE special_request_id = $1 FOR UPDATE", [
      specialRequestId,
    ])

    if (request.rowCount === 0) {
      await client.query("ROLLBACK")
      res.status(404).json({ error: "Access request not found" })
      return
    }

    const row = request.rows[0]
    await client.query("UPDATE access_requests SET status = 'disabled', updated_at = now() WHERE special_request_id = $1", [
      specialRequestId,
    ])
    await client.query("UPDATE api_keys SET status = 'disabled', revoked_at = now() WHERE special_request_id = $1", [
      specialRequestId,
    ])

    await client.query("COMMIT")
    disabledUser = row
  } catch (error) {
    await client.query("ROLLBACK")
    console.error(error)
    res.status(500).json({ error: "Disable failed." })
    return
  } finally {
    client.release()
  }

  const emailTemplate = mailTemplates.accessDisabled({ firstName: disabledUser.first_name, reason })
  let emailStatus = "failed_to_queue"
  try {
    const email = await sendOrQueueEmail(pool, {
      specialRequestId,
      recipient: disabledUser.business_mail,
      subject: emailTemplate.subject,
      bodyText: emailTemplate.bodyText,
      bodyHtml: emailTemplate.bodyHtml,
    })
    emailStatus = email.status
  } catch (emailError) {
    console.error("Access-disabled email could not be queued", emailError)
  }
  res.json({ specialRequestId, emailStatus })
})

app.post("/v1/admin/test-email", requireAdmin, async (req, res) => {
  const recipient = String(req.body?.recipient || "").trim()
  if (!recipient) {
    res.status(400).json({ error: "Recipient is required." })
    return
  }

  const client = await pool.connect()
  try {
    const emailTemplate = mailTemplates.socialPostReminder({ firstName: "Assem", specialRequestId: "HCN10-TEST" })
    const email = await sendOrQueueEmail(client, {
      specialRequestId: null,
      recipient,
      subject: `Test: ${emailTemplate.subject}`,
      bodyText: emailTemplate.bodyText,
      bodyHtml: emailTemplate.bodyHtml,
    })
    res.json({ recipient, emailStatus: email.status })
  } finally {
    client.release()
  }
})

app.post("/v1/chat-logs", (_req, res) => {
  res.status(410).json({ error: "Use POST /v1/chat/completions. Client-supplied chat logs are not accepted." })
})

registerModelGateway({
  app,
  pool,
  authenticateApiKey,
  enforceApiKeyRateLimit,
  enforceApiKeyConcurrency,
})

app.use((error, _req, res, _next) => {
  console.error(error)
  res.status(error.statusCode || error.status || 500).json({ error: error.expose ? error.message : "Internal server error" })
})

app.listen(port, "0.0.0.0", () => {
  console.log(`TokenAI early access API listening on ${port}`)
})
