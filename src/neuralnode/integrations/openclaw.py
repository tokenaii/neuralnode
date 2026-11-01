"""OpenClaw Integration and OpenAI-Compatible API Server for NeuralNode.

This module provides:
- OpenClawServer: A local OpenAI-compatible API server wrapping any NeuralNode instance.
- OpenClawManager: Downloader, configurator, and runner for the OpenClaw gateway.
- OpenClawClient: A client to interact with the running OpenClaw REST API.
"""

import os
import secrets
import sys
import time
import socket
import logging
import threading
import subprocess
import shutil
import json
import hmac
import ipaddress
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# FastAPI and Uvicorn might be optional/lazy imported to avoid slow startup
_FASTAPI_AVAILABLE = False
try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore
    Header = None  # type: ignore
    HTTPException = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    StreamingResponse = None  # type: ignore


# ── Pydantic Models for OpenAI API ───────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


# ── OpenAI-Compatible FastAPI Server ─────────────────────────────────────────


class OpenClawServer:
    """FastAPI server exposing an OpenAI-compatible API for NeuralNode models."""

    def __init__(
        self,
        node: Any,
        port: int = 18788,
        host: str = "127.0.0.1",
        api_key: Optional[str] = None,
        cors_origins: Optional[List[str]] = None,
    ):
        if not _FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI and uvicorn are required for OpenClawServer. "
                "Install them via: pip install fastapi uvicorn"
            )
        self.node = node
        self.port = port
        self.host = host
        if api_key is not None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("api_key must be a non-empty string when provided.")
        if any(not isinstance(origin, str) or origin == "*" for origin in cors_origins or []):
            raise ValueError(
                "cors_origins must contain explicit origins; wildcard origins are unsafe."
            )
        self.api_key = api_key
        self.cors_origins = list(cors_origins or [])
        if not self.api_key and not self._is_loopback_host(host):
            raise ValueError(
                "api_key is required when OpenClawServer binds to a non-loopback host."
            )
        self.app = FastAPI(title="NeuralNode OpenAI-Compatible API Server")
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[Any] = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        # Browser access is opt-in. In particular, never combine wildcard
        # origins with credentials on an API that can execute model requests.
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

        def require_api_auth(authorization: Optional[str]) -> None:
            if not self.api_key:
                return
            scheme, separator, token = (authorization or "").partition(" ")
            if (
                not separator
                or scheme.lower() != "bearer"
                or not hmac.compare_digest(token, self.api_key)
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Bearer access token required.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        @self.app.get("/v1/models")
        async def list_models(authorization: Optional[str] = Header(default=None)):
            require_api_auth(authorization)
            return {
                "object": "list",
                "data": [
                    {
                        "id": getattr(self.node, "model", "neuralnode-model") or "neuralnode-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "neuralnode",
                    }
                ],
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            request: ChatCompletionRequest,
            authorization: Optional[str] = Header(default=None),
        ):
            require_api_auth(authorization)
            try:
                # Parse OpenAI messages into NeuralNode format
                last_message, system_prompt, history = self._parse_messages(request.messages)

                if request.stream:
                    return StreamingResponse(
                        self._stream_completion(
                            last_message, system_prompt, history, request.model
                        ),
                        media_type="text/event-stream",
                    )

                # Call NeuralNode chat function (synchronous blocking call)
                response_text = self.node.chat(
                    message=last_message, system_prompt=system_prompt, history=history, stream=False
                )

                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
                }
            except Exception:
                logger.exception("OpenClaw chat request failed")
                raise HTTPException(
                    status_code=500, detail="The request could not be completed."
                ) from None

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        if host.strip().lower() == "localhost":
            return True
        try:
            return ipaddress.ip_address(host.strip("[]")).is_loopback
        except ValueError:
            return False

    def _parse_messages(self, messages: List[ChatMessage]) -> tuple:
        system_prompt = None
        history = []
        last_message = ""

        # Extract system prompt, last message, and intermediate history
        for i, msg in enumerate(messages):
            role = msg.role
            content = msg.content

            if role == "system":
                system_prompt = content
            elif i == len(messages) - 1 and role == "user":
                last_message = content
            else:
                history.append({"role": role, "content": content})

        if not last_message and messages:
            last_message = messages[-1].content

        return last_message, system_prompt, history

    def _stream_completion(
        self,
        message: str,
        system_prompt: Optional[str],
        history: List[Dict[str, Any]],
        model_name: str,
    ) -> Generator[str, None, None]:
        import json

        chat_id = f"chatcmpl-{int(time.time())}"

        try:
            # Call streaming chat
            response_generator = self.node.chat(
                message=message, system_prompt=system_prompt, history=history, stream=True
            )

            for chunk in response_generator:
                data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Send stop choice
            stop_data = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(stop_data)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            logger.exception("OpenClaw streaming request failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "message": "The request could not be completed.",
                            "type": "server_error",
                        }
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

    def start(self) -> None:
        """Start the API server in a background thread."""
        import uvicorn

        # Check if port is already in use
        if self._is_port_in_use(self.port):
            logger.warning(
                f"Port {self.port} is already in use. Assuming server is already running."
            )
            return

        config = uvicorn.Config(app=self.app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server to bind
        retries = 10
        while retries > 0 and not self._is_port_in_use(self.port):
            time.sleep(0.5)
            retries -= 1

        logger.info(f"OpenAI-compatible local server started at http://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the local API server."""
        if self._server:
            self._server.should_exit = True
            if self._thread:
                self._thread.join(timeout=5)
            self._server = None
            self._thread = None
            logger.info("OpenAI-compatible local server stopped.")

    def _is_port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, port)) == 0


# ── OpenClaw REST Client ─────────────────────────────────────────────────────


class OpenClawClient:
    """Client for the OpenAI-compatible HTTP endpoints on an OpenClaw Gateway."""

    def __init__(self, gateway_url: str = "http://127.0.0.1:18789", api_key: Optional[str] = None):
        self.gateway_url = gateway_url.rstrip("/")
        self.api_url = (
            self.gateway_url if self.gateway_url.endswith("/v1") else f"{self.gateway_url}/v1"
        )
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.headers["Content-Type"] = "application/json"

    def check_health(self) -> bool:
        """Check if OpenClaw Gateway is online."""
        try:
            # Simple check to gateway url or api endpoint
            response = requests.get(f"{self.gateway_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            try:
                # Alternate ping check
                response = requests.get(f"{self.gateway_url}/", timeout=5)
                return response.status_code == 200
            except Exception:
                return False

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Validate and return a stable conversation id for later requests.

        OpenClaw's HTTP API creates a session implicitly when the first chat
        completion is sent with ``user=session_id``; it has no REST create route.
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        return {"sessionId": session_id, "status": "pending_first_message"}

    def send_task(self, session_id: str, prompt: str) -> Dict[str, Any]:
        """Send a prompt through OpenClaw's OpenAI-compatible chat endpoint."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        payload = {
            "model": "openclaw/default",
            "user": session_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            f"{self.api_url}/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "Failed to send task through OpenClaw's OpenAI-compatible API "
                f"(HTTP {response.status_code}): {response.text}"
            )
        return response.json()

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Raise because OpenClaw exposes session inspection over its WS RPC API."""
        raise NotImplementedError(
            "OpenClaw does not expose session status through its OpenAI-compatible HTTP API. "
            "Use the Gateway WebSocket sessions.list RPC."
        )


# ── OpenClaw Process Manager ─────────────────────────────────────────────────


class OpenClawManager:
    """Manages downloading, configuring, and executing the OpenClaw Gateway locally."""

    def __init__(
        self,
        node: Any,
        install_dir: Optional[str] = None,
        gateway_port: int = 18789,
        api_server_port: int = 18788,
        gateway_host: str = "127.0.0.1",
        api_key: Optional[str] = None,
    ):
        self.node = node
        self.install_dir = Path(install_dir or "./openclaw_runtime").resolve()
        self.gateway_port = gateway_port
        self.api_server_port = api_server_port
        self.gateway_host = gateway_host
        # Never ship a shared credential as a fallback.  Callers can provide
        # one explicitly or via the environment; otherwise generate a fresh
        # per-manager token for the local gateway.
        self.api_key = (
            api_key or os.getenv("NEURALNODE_OPENCLAW_TOKEN") or secrets.token_urlsafe(32)
        )

        # Initialize internal server
        self.server = OpenClawServer(
            node=self.node,
            port=self.api_server_port,
            api_key=self.api_key,
        )
        self.process: Optional[subprocess.Popen] = None

    def setup(self) -> None:
        """Download and install OpenClaw into the target directory."""
        if not self.install_dir.exists():
            self.install_dir.mkdir(parents=True, exist_ok=True)

        repo_dir = self.install_dir / "openclaw"

        # Step 1: Clone OpenClaw repository if not present
        if not repo_dir.exists():
            if shutil.which("git") is None:
                raise RuntimeError("git is required to download OpenClaw. Install git and retry.")
            logger.info(f"Cloning OpenClaw repository to {repo_dir}...")
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/openclaw/openclaw.git", str(repo_dir)],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone OpenClaw repository: {e.stderr.decode()}")

        # Step 2: Install Node.js dependencies
        logger.info("Installing OpenClaw Node.js dependencies...")
        # Check npm package manager
        try:
            # We check if pnpm or npm is available
            if shutil.which("pnpm"):
                pm_cmd = ["pnpm", "install"]
            elif shutil.which("npm"):
                pm_cmd = ["npm", "install"]
            else:
                raise RuntimeError(
                    "npm or pnpm is required to install OpenClaw dependencies. Install Node.js and retry."
                )
            install_result = subprocess.run(
                pm_cmd, cwd=str(repo_dir), check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(
                errors="replace"
            )
            raise RuntimeError(f"Failed to install node dependencies: {output.strip()}")

        package_json_path = repo_dir / "package.json"
        dist_entry_exists = (repo_dir / "dist" / "entry.mjs").exists() or (
            repo_dir / "dist" / "entry.js"
        ).exists()
        if package_json_path.exists() and not dist_entry_exists:
            try:
                with open(package_json_path, "r", encoding="utf-8") as f:
                    package_json = json.load(f)
                scripts = package_json.get("scripts", {})
            except Exception:
                scripts = {}

            if "build" in scripts:
                build_cmd = [pm_cmd[0], "run", "build"]
                logger.info(f"Building OpenClaw via: {' '.join(build_cmd)}")
                try:
                    subprocess.run(
                        build_cmd,
                        cwd=str(repo_dir),
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError as e:
                    output = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(
                        errors="replace"
                    )
                    raise RuntimeError(f"Failed to build OpenClaw source tree: {output.strip()}")

        # Step 3: Write configuration pointing to NeuralNode API server
        self.configure()

    def configure(self) -> None:
        """Write config files pointing to NeuralNode local API server."""
        repo_dir = self.install_dir / "openclaw"
        if not repo_dir.exists():
            raise FileNotFoundError(
                f"OpenClaw directory not found at {repo_dir}. Run setup() first."
            )

        # Write config.json
        import json

        config_path = repo_dir / "config.json"
        config_data = {
            "gateway": {
                "port": self.gateway_port,
                "host": self.gateway_host,
                "token": self.api_key,
            },
            "llm": {
                "provider": "openai",
                "baseURL": f"http://127.0.0.1:{self.api_server_port}/v1",
                "apiKey": self.api_key,
                "model": getattr(self.node, "model", "neuralnode-model") or "neuralnode-model",
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Write .env file for compatibility
        env_path = repo_dir / ".env"
        with open(env_path, "w") as f:
            f.write(f"PORT={self.gateway_port}\n")
            f.write(f"GATEWAY_HOST={self.gateway_host}\n")
            f.write(f"OPENAI_BASE_URL=http://127.0.0.1:{self.api_server_port}/v1\n")
            f.write(f"OPENAI_API_KEY={self.api_key}\n")
            if self.api_key:
                f.write(f"GATEWAY_TOKEN={self.api_key}\n")

        logger.info("OpenClaw configuration written successfully.")

    def start(self) -> None:
        """Start local NeuralNode API Server and launch OpenClaw Gateway."""
        # 1. Start NeuralNode API Server
        self.server.start()

        # 2. Start OpenClaw Gateway Process
        repo_dir = self.install_dir / "openclaw"
        if not repo_dir.exists():
            raise FileNotFoundError("OpenClaw repository not found. Run setup() first.")

        # Find starting command
        # Use node openclaw.mjs gateway or npm run dev/start
        startup_cmd = ["node", "openclaw.mjs", "gateway", "--allow-unconfigured"]

        # Fallback to dev command if js build not present
        if not (repo_dir / "openclaw.mjs").exists():
            # Check pnpm vs npm run dev
            if shutil.which("pnpm"):
                startup_cmd = ["pnpm", "run", "dev"]
            elif shutil.which("npm"):
                startup_cmd = ["npm", "run", "dev"]
            else:
                raise RuntimeError(
                    "npm or pnpm is required to start OpenClaw. Install Node.js and retry."
                )

        logger.info(f"Starting OpenClaw Gateway via: {' '.join(startup_cmd)}")
        process_env = os.environ.copy()
        process_env.update(
            {
                "PORT": str(self.gateway_port),
                "GATEWAY_HOST": self.gateway_host,
                "GATEWAY_TOKEN": self.api_key,
                "OPENAI_BASE_URL": f"http://127.0.0.1:{self.api_server_port}/v1",
                "OPENAI_API_KEY": self.api_key,
                "OPENAI_MODEL": getattr(self.node, "model", "neuralnode-model")
                or "neuralnode-model",
            }
        )
        self.process = subprocess.Popen(
            startup_cmd,
            cwd=str(repo_dir),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for gateway to start
        client = self.get_client()
        retries = 30
        while retries > 0:
            if client.check_health():
                logger.info(f"OpenClaw Gateway started successfully on port {self.gateway_port}.")
                return
            time.sleep(1)
            retries -= 1

        # Check process state
        if self.process.poll() is not None:
            stdout, stderr = self.process.communicate()
            raise RuntimeError(
                f"OpenClaw process exited immediately with code {self.process.returncode}. "
                f"Output: {(stdout or '') + (stderr or '')}"
            )

        logger.warning("OpenClaw Gateway start timed out, but process is still running.")

    def stop(self) -> None:
        """Stop the OpenClaw Gateway and the NeuralNode API Server."""
        # Stop OpenClaw process
        if self.process:
            logger.info("Stopping OpenClaw Gateway process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

        # Stop local API Server
        self.server.stop()

    def get_client(self) -> OpenClawClient:
        """Get an initialized OpenClawClient for the gateway."""
        url = f"http://{self.gateway_host}:{self.gateway_port}"
        return OpenClawClient(gateway_url=url, api_key=self.api_key)
