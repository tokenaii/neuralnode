"""OpenAI-compatible gateway and access-token management for NeuralNode."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import string
import time
from pathlib import Path
from typing import Any, Optional

import requests

H_PREVIEW_MODEL = "h-preview"
H_PREVIEW_HF_REPO = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
_TOKEN_ALPHABET = string.ascii_letters + string.digits


class APIKeyStore:
    """Persist token hashes and enforce per-token request limits."""

    def __init__(self, database: Optional[str] = None, rpm_limit: int = 60) -> None:
        if rpm_limit < 1:
            raise ValueError("rpm_limit must be at least 1.")
        self.database = str(database or os.getenv("NEURALNODE_API_DB", "neuralnode-api.sqlite3"))
        self.rpm_limit = rpm_limit
        Path(self.database).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS api_keys (
                    token_hash TEXT PRIMARY KEY,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    rpm_limit INTEGER,
                    created_at INTEGER NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0
                )""")
            connection.execute("""CREATE TABLE IF NOT EXISTS api_requests (
                    token_hash TEXT NOT NULL,
                    requested_at INTEGER NOT NULL
                )""")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_requests_key_time ON api_requests(token_hash, requested_at)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_api_keys_single_admin ON api_keys(role) WHERE role = 'admin'"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _new_token() -> str:
        return "nn-" + "".join(secrets.choice(_TOKEN_ALPHABET) for _ in range(21))

    def create_admin_token(self) -> str:
        """Create the one owner token. It is returned only at creation time."""
        token = self._new_token()
        with self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO api_keys(token_hash, role, rpm_limit, created_at) VALUES (?, 'admin', NULL, ?)",
                    (self._hash(token), int(time.time())),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError("The NeuralNode admin token already exists.") from exc
        return token

    def create_user_token(
        self,
        admin_token: str,
        rpm_limit: Optional[int] = None,
        *,
        tier: str = "basic",
    ) -> str:
        admin = self.authenticate(admin_token)
        if not admin or admin["role"] != "admin":
            raise PermissionError("A valid admin token is required.")
        tier = tier.lower()
        if tier not in {"basic", "elite"}:
            raise ValueError("tier must be 'basic' or 'elite'.")
        if tier == "elite":
            limit = None
        else:
            limit = self.rpm_limit if rpm_limit is None else rpm_limit
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("Basic tier rpm_limit must be an integer of at least 1.")
        token = self._new_token()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO api_keys(token_hash, role, rpm_limit, created_at) VALUES (?, 'user', ?, ?)",
                (self._hash(token), limit, int(time.time())),
            )
        return token

    def authenticate(self, token: str) -> Optional[dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT token_hash, role, rpm_limit FROM api_keys WHERE token_hash = ? AND revoked = 0",
                (self._hash(token),),
            ).fetchone()
        return dict(row) if row else None

    def authorize_request(self, token: str) -> tuple[Optional[dict[str, Any]], bool]:
        """Return token metadata and whether its RPM budget permits a request."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT token_hash, role, rpm_limit FROM api_keys WHERE token_hash = ? AND revoked = 0",
                (self._hash(token),),
            ).fetchone()
            if row is None:
                return None, False
            metadata = dict(row)
            if metadata["role"] == "admin" or metadata["rpm_limit"] is None:
                return metadata, True
            now = int(time.time())
            count = connection.execute(
                "SELECT COUNT(*) FROM api_requests WHERE token_hash = ? AND requested_at > ?",
                (metadata["token_hash"], now - 60),
            ).fetchone()[0]
            if count >= metadata["rpm_limit"]:
                return metadata, False
            connection.execute(
                "INSERT INTO api_requests(token_hash, requested_at) VALUES (?, ?)",
                (metadata["token_hash"], now),
            )
            return metadata, True

    def revoke_user_token(self, admin_token: str, token: str) -> bool:
        admin = self.authenticate(admin_token)
        if not admin or admin["role"] != "admin":
            raise PermissionError("A valid admin token is required.")
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE api_keys SET revoked = 1 WHERE token_hash = ? AND role = 'user' AND revoked = 0",
                (self._hash(token),),
            )
        return cursor.rowcount == 1


def create_api_app(
    *,
    key_store: Optional[APIKeyStore] = None,
    upstream_base_url: Optional[str] = None,
    upstream_api_key: Optional[str] = None,
    upstream_model: Optional[str] = None,
):
    """Build an authenticated API gateway that forwards H-Preview to vLLM-compatible servers."""
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise ImportError("Install the API extra with: pip install 'neuralnode[api]'") from exc

    store = key_store or APIKeyStore()
    base_url = (upstream_base_url or os.getenv("H_PREVIEW_UPSTREAM_URL", "")).rstrip("/")
    upstream_key = upstream_api_key or os.getenv("H_PREVIEW_UPSTREAM_API_KEY", "")
    remote_model = upstream_model or os.getenv("H_PREVIEW_UPSTREAM_MODEL", H_PREVIEW_HF_REPO)
    if not base_url:
        raise ValueError(
            "Set H_PREVIEW_UPSTREAM_URL to the remote server's OpenAI-compatible /v1 URL."
        )

    app = FastAPI(title="NeuralNode Model API", version="1.0.0")

    def get_token(authorization: Optional[str]) -> str:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(
                status_code=401, detail={"message": "Bearer access token required."}
            )
        return authorization[7:].strip()

    def require_admin(authorization: Optional[str]) -> str:
        token = get_token(authorization)
        metadata = store.authenticate(token)
        if not metadata or metadata["role"] != "admin":
            raise HTTPException(status_code=403, detail={"message": "Admin token required."})
        return token

    def request_access(authorization: Optional[str]) -> dict[str, Any]:
        token = get_token(authorization)
        metadata, allowed = store.authorize_request(token)
        if metadata is None:
            raise HTTPException(
                status_code=401, detail={"message": "Invalid or revoked access token."}
            )
        if not allowed:
            raise HTTPException(
                status_code=429, detail={"message": "Requests per minute limit exceeded."}
            )
        return metadata

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        request_access(authorization)
        return {
            "object": "list",
            "data": [{"id": H_PREVIEW_MODEL, "object": "model", "owned_by": "neuralnode"}],
        }

    @app.get("/v1/access")
    def access_info(authorization: Optional[str] = Header(default=None)):
        metadata = store.authenticate(get_token(authorization))
        if metadata is None:
            raise HTTPException(
                status_code=401, detail={"message": "Invalid or revoked access token."}
            )
        tier = (
            "Admin"
            if metadata["role"] == "admin"
            else ("elite" if metadata["rpm_limit"] is None else "basic")
        )
        return {"role": metadata["role"], "tier": tier.title(), "rpm_limit": metadata["rpm_limit"]}

    @app.post("/v1/chat/completions")
    def chat_completions(
        payload: dict[str, Any], authorization: Optional[str] = Header(default=None)
    ):
        request_access(authorization)
        requested_model = payload.get("model")
        if requested_model not in {H_PREVIEW_MODEL, remote_model, H_PREVIEW_HF_REPO}:
            raise HTTPException(
                status_code=404, detail={"message": f"Unknown model: {requested_model}"}
            )
        payload["model"] = remote_model
        headers = {"Content-Type": "application/json"}
        if upstream_key:
            headers["Authorization"] = f"Bearer {upstream_key}"
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(15, 600),
                stream=bool(payload.get("stream")),
            )
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502, detail={"message": f"Model upstream unavailable: {exc}"}
            ) from exc
        if not response.ok:
            try:
                detail = response.json()
            except ValueError:
                detail = {"message": response.text[:2000]}
            status = response.status_code if 400 <= response.status_code < 500 else 502
            response.close()
            raise HTTPException(status_code=status, detail=detail)
        if payload.get("stream"):

            def stream_body():
                try:
                    yield from response.iter_content(chunk_size=None)
                finally:
                    response.close()

            return StreamingResponse(stream_body(), media_type="text/event-stream")
        try:
            return response.json()
        finally:
            response.close()

    @app.post("/v1/admin/tokens")
    def create_token(payload: dict[str, Any], authorization: Optional[str] = Header(default=None)):
        admin_token = require_admin(authorization)
        try:
            if payload.get("role", "user") == "admin":
                raise ValueError(
                    "The owner admin token is created locally and can exist only once."
                )
            tier = str(payload.get("tier", "basic")).lower()
            rpm_limit = payload.get("rpm_limit")
            token = store.create_user_token(admin_token, rpm_limit, tier=tier)
        except (ValueError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
        effective_limit = None if tier == "elite" else (rpm_limit or store.rpm_limit)
        return {"token": token, "role": "user", "tier": tier.title(), "rpm_limit": effective_limit}

    @app.delete("/v1/admin/tokens")
    def revoke_token(payload: dict[str, Any], authorization: Optional[str] = Header(default=None)):
        admin_token = require_admin(authorization)
        try:
            revoked = store.revoke_user_token(admin_token, str(payload.get("token", "")))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail={"message": str(exc)}) from exc
        if not revoked:
            raise HTTPException(status_code=404, detail={"message": "Active user token not found."})
        return {"revoked": True}

    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError("Install the API extra with: pip install 'neuralnode[api]'") from exc
    uvicorn.run(create_api_app(**kwargs), host=host, port=port)


def generate_admin_token(database: Optional[str] = None) -> str:
    """Create the single owner token from a trusted local process."""
    return APIKeyStore(database=database).create_admin_token()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NeuralNode OpenAI-compatible model gateway")
    commands = parser.add_subparsers(dest="command", required=True)
    admin_command = commands.add_parser("create-admin", help="Create the one owner token")
    admin_command.add_argument("--database")
    serve_command = commands.add_parser("serve", help="Start the H-Preview API gateway")
    serve_command.add_argument("--host", default="0.0.0.0")
    serve_command.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "create-admin":
        print(generate_admin_token(args.database))
    else:
        run_api_server(host=args.host, port=args.port)


__all__ = [
    "APIKeyStore",
    "H_PREVIEW_MODEL",
    "H_PREVIEW_HF_REPO",
    "create_api_app",
    "generate_admin_token",
    "run_api_server",
]


if __name__ == "__main__":
    main()
