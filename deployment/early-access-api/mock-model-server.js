import http from "node:http"

let requestCount = 0

http.createServer((req, res) => {
  if (req.method !== "POST" || req.url !== "/v1/chat/completions") {
    res.writeHead(404, { "Content-Type": "application/json" })
    res.end(JSON.stringify({ error: "Not found" }))
    return
  }

  let body = ""
  req.on("data", (chunk) => {
    body += chunk
  })
  req.on("end", () => {
    const payload = JSON.parse(body)
    requestCount += 1
    const lastMessage = payload.messages?.at(-1)
    const requestedTool = payload.tools?.[0]?.function
    const message = requestedTool
      ? {
          role: "assistant",
          content: null,
          tool_calls: [{
            id: `call_mock_${requestCount}`,
            type: "function",
            function: { name: requestedTool.name, arguments: JSON.stringify({ target: "gateway" }) },
          }],
        }
      : {
          role: "assistant",
          content: lastMessage?.role === "tool"
            ? `Tool result received: ${lastMessage.content}`
            : `Mock model response ${requestCount}.`,
        }
    res.writeHead(200, { "Content-Type": "application/json" })
    res.end(JSON.stringify({
      id: `mock-${requestCount}`,
      choices: [{
        message,
        finish_reason: requestedTool ? "tool_calls" : "stop",
      }],
      usage: { prompt_tokens: 8, completion_tokens: 4, total_tokens: 12 },
    }))
  })
}).listen(19090, "127.0.0.1")
