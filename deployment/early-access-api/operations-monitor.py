#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import smtplib
import ssl
import subprocess
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

ROOT = Path("/opt/tokenai-early-access")
STATE_PATH = ROOT / "monitor-state.json"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def run(command: list[str], timeout: int = 15) -> tuple[bool, str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def check_http(url: str, headers: dict[str, str] | None = None) -> tuple[bool, str]:
    request_headers = {"User-Agent": "TokenAI-Early-Access-Monitor/1.0", **(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read(20000).decode("utf-8", errors="replace")
            return response.status == 200, body
    except Exception as error:
        return False, str(error)


def collect(env: dict[str, str]) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    details: dict[str, object] = {}

    for container in ("tokenai-early-access-api-1", "tokenai-early-access-postgres-1", "tokenai-early-access-cloudflared-1"):
        ok, status = run(["docker", "inspect", "--format", "{{.State.Status}}", container])
        if not ok or status != "running":
            failures.append(f"Container {container} is not running: {status}")

    health_ok, health_body = check_http("http://127.0.0.1:8080/health")
    if not health_ok:
        failures.append(f"API health check failed: {health_body}")
    else:
        try:
            details["api"] = json.loads(health_body)
        except json.JSONDecodeError:
            failures.append("API health response was not valid JSON.")

    public_ok, public_body = check_http("https://api.tokenai.llc/health")
    if not public_ok:
        failures.append(f"Public Cloudflare API edge failed: {public_body}")
    else:
        try:
            if not json.loads(public_body).get("ok"):
                failures.append("Public Cloudflare API edge returned an unhealthy response.")
        except json.JSONDecodeError:
            failures.append("Public Cloudflare API edge did not return JSON.")

    model_private_ip = env.get("MODEL_PRIVATE_IP", "172.31.28.151")
    ok, container_route = run([
        "docker", "exec", "tokenai-early-access-api-1", "node", "-e",
        f"Promise.all([8011,8012].map(p=>fetch(`http://{model_private_ip}:${{p}}/health`).then(r=>{{if(!r.ok)throw new Error(String(r.status))}}))).catch(()=>process.exit(1))",
    ])
    if not ok:
        failures.append(f"API container cannot reach both model workers: {container_route}")

    for port in (8011, 8012):
        ok, body = check_http(f"http://{model_private_ip}:{port}/health")
        if not ok:
            failures.append(f"Model worker on private VPC port {port} failed: {body}")

    backup_key = env.get("BACKUP_REMOTE_SSH_KEY", "/home/ec2-user/.ssh/horus_backup_transfer")
    backup_host = env.get("BACKUP_REMOTE_HOST", "54.91.40.201")
    ok, metrics = run([
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=/home/ec2-user/.ssh/known_hosts",
        "-i", backup_key, f"ec2-user@{backup_host}", "metrics",
    ])
    if not ok:
        failures.append(f"Model host metrics failed: {metrics}")
    else:
        parsed = dict(line.split("=", 1) for line in metrics.splitlines() if "=" in line)
        details["model"] = parsed
        if parsed.get("WORKER_1") != "active" or parsed.get("WORKER_2") != "active":
            failures.append("One or more model workers are inactive.")
        if int(parsed.get("GPU_TEMPERATURE_C", "0") or 0) >= 85:
            failures.append(f"GPU temperature is high: {parsed.get('GPU_TEMPERATURE_C')} C")
        if int(parsed.get("DATA_DISK_USED_PERCENT", "0") or 0) >= 85:
            failures.append(f"Model data disk usage is high: {parsed.get('DATA_DISK_USED_PERCENT')}%")

    root_usage = os.statvfs(ROOT)
    root_used = 100 - int(root_usage.f_bavail * 100 / root_usage.f_blocks)
    details["apiRootDiskUsedPercent"] = root_used
    if root_used >= 80:
        failures.append(f"API root disk usage is high: {root_used}%")

    backups = sorted((ROOT / "backups").glob("early-access-*.tar.gz"), key=lambda path: path.stat().st_mtime)
    if not backups:
        failures.append("No encrypted off-instance backup has completed.")
    else:
        age_hours = (datetime.now(timezone.utc).timestamp() - backups[-1].stat().st_mtime) / 3600
        details["latestBackupAgeHours"] = round(age_hours, 2)
        if age_hours > 30:
            failures.append(f"Latest encrypted backup is {age_hours:.1f} hours old.")

    return failures, details


def send_alert(env: dict[str, str], subject: str, body: str) -> None:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = f"{env.get('SMTP_FROM_NAME', 'TokenAI')} <{env['SMTP_FROM_EMAIL']}>"
    message["To"] = env.get("ALERT_EMAIL", env.get("REPLY_TO_EMAIL", "support@tokenai.llc"))
    message["Reply-To"] = env.get("REPLY_TO_EMAIL", env["SMTP_FROM_EMAIL"])
    message.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(env["SMTP_HOST"], int(env.get("SMTP_PORT", "465")), context=context, timeout=20) as smtp:
        smtp.login(env["SMTP_USER"], env["SMTP_PASS"])
        smtp.send_message(message)


def main() -> None:
    env = load_env(ROOT / ".env")
    failures, details = collect(env)
    current = {"healthy": not failures, "failures": failures}
    previous = {}
    if STATE_PATH.exists():
        try:
            previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}

    if current != previous:
        if failures:
            subject = "TokenAI Early Access production alert"
            body = "The production monitor detected:\n\n" + "\n".join(f"- {item}" for item in failures)
        else:
            subject = "TokenAI Early Access service recovered"
            body = "All monitored Early Access API, database, model worker, GPU, disk, and backup checks are healthy."
        body += "\n\nUTC: " + datetime.now(timezone.utc).isoformat()
        body += "\n\nDetails:\n" + json.dumps(details, indent=2, default=str)
        send_alert(env, subject, body)

    STATE_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    os.chmod(STATE_PATH, 0o600)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
