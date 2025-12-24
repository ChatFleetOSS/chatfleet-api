#!/usr/bin/env python3
# generated-by: codex-agent 2025-12-24T00:00:00Z
"""
Verify login-time admin promotion end-to-end.

This script:
  1) Ensures the API health endpoint is reachable.
  2) Creates an admin promotion intent for a target email (via docker compose + mongosh).
  3) Logs in (or registers) with that email and password.
  4) Verifies the returned user role is `admin` immediately (no polling delay).

It supports both local dev compose (Mongo without auth) and the infra compose (Mongo with auth).

Usage examples:
  python backend/scripts/verify_admin_promotion.py \
    --email you@example.com --password 'StrongPass!234' --api http://localhost:8080/api

  # Using an infra install (default path), explicit compose dir
  python backend/scripts/verify_admin_promotion.py --compose-dir "$HOME/chatfleet-infra" \
    --email you@example.com --password 'StrongPass!234'

Exit codes: 0 on success; non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Tuple


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def http_get_json(url: str, timeout: float = 10.0) -> Tuple[int, Dict]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception:
        return 0, {}


def http_post_json(url: str, payload: Dict, timeout: float = 15.0) -> Tuple[int, Dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(body)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            return e.code, json.loads(err_body)
        except Exception:
            return e.code, {}
    except Exception:
        return 0, {}


def find_compose_dir(arg_dir: str | None) -> Path:
    if arg_dir:
        p = Path(arg_dir).expanduser().resolve()
        if not (p / "docker-compose.yml").exists():
            raise SystemExit(f"compose dir '{p}' missing docker-compose.yml")
        return p
    # Prefer infra install if present
    default_infra = Path(os.environ.get("HOME", "~")).expanduser() / "chatfleet-infra"
    if (default_infra / "docker-compose.yml").exists():
        return default_infra
    # Fallback to current repo root (this script assumed to run from ChatFleet repo)
    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "docker-compose.yml").exists():
        return repo_root
    raise SystemExit("Could not locate a docker-compose.yml. Provide --compose-dir.")


def parse_env(env_path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):  # comments/blank
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def compose_cmd(compose_dir: Path) -> Tuple[str, Path]:
    # Use integrated docker compose; fall back to docker-compose if needed
    exe = "docker"
    try:
        subprocess.run([exe, "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "docker compose", compose_dir
    except Exception:
        pass
    # Fallback
    exe2 = "docker-compose"
    try:
        subprocess.run([exe2, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "docker-compose", compose_dir
    except Exception:
        raise SystemExit("Neither 'docker compose' nor 'docker-compose' is available")


def create_promotion_intent(compose_dir: Path, email: str) -> None:
    env = parse_env(compose_dir / ".env")
    cmd_base, _ = compose_cmd(compose_dir)
    # Build mongosh command (auth if credentials present)
    js = (
        "db=db.getSiblingDB(\"chatfleet\"); var now=new Date(); var exp=new Date(now.getTime()+48*3600*1000); "
        f"db.admin_promotions.updateOne({{email:\"{email}\"}}, {{$setOnInsert:{{email:\"{email}\",created_at:now,expires_at:exp,redeemed:false}}}}, {{upsert:true}}); print(\"INTENT_OK\");"
    )
    if env.get("MONGO_ROOT_USER") and env.get("MONGO_ROOT_PASSWORD"):
        args = [
            *shlex.split(cmd_base),
            "-f",
            str(compose_dir / "docker-compose.yml"),
            "exec",
            "-T",
            "mongo",
            "mongosh",
            "--quiet",
            "-u",
            env["MONGO_ROOT_USER"],
            "-p",
            env["MONGO_ROOT_PASSWORD"],
            "--authenticationDatabase",
            "admin",
            "--eval",
            js,
        ]
    else:
        args = [
            *shlex.split(cmd_base),
            "-f",
            str(compose_dir / "docker-compose.yml"),
            "exec",
            "-T",
            "mongo",
            "mongosh",
            "--quiet",
            "--eval",
            js,
        ]
    proc = subprocess.run(args, cwd=str(compose_dir), capture_output=True, text=True)
    if proc.returncode != 0 or "INTENT_OK" not in (proc.stdout or ""):
        eprint("[verify] Failed to create promotion intent:")
        eprint(proc.stdout)
        eprint(proc.stderr)
        raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify login-time admin promotion works end-to-end")
    ap.add_argument("--api", default="http://localhost:8080/api", help="API base URL (default: %(default)s)")
    ap.add_argument("--compose-dir", default=None, help="Directory containing docker-compose.yml (infra or dev repo)")
    ap.add_argument("--email", required=True, help="Email to promote + use for login/register")
    ap.add_argument("--password", default=None, help="Password to use; if omitted, a random one is generated for register flow")
    ap.add_argument("--name", default="Admin Test", help="Name to use on register (default: %(default)s)")
    ap.add_argument("--timeout", type=int, default=60, help="Seconds to wait for API health (default: %(default)s)")
    args = ap.parse_args()

    api_base = args.api.rstrip("/")
    compose_dir = find_compose_dir(args.compose_dir)

    # 1) Wait for health
    health_url = f"{api_base}/health"
    print(f"[verify] Waiting for API health at {health_url} ...")
    deadline = time.time() + max(5, args.timeout)
    ok = False
    while time.time() < deadline:
        code, _ = http_get_json(health_url, timeout=5)
        if code == 200:
            ok = True
            break
        time.sleep(2)
    if not ok:
        raise SystemExit("API health not ready; aborting")
    print("[verify] Health OK")

    # 2) Create admin promotion intent for the email
    print(f"[verify] Creating admin promotion intent for {args.email} ...")
    create_promotion_intent(compose_dir, args.email)
    print("[verify] Intent created (48h TTL)")

    # 3) Login or register
    login_url = f"{api_base}/auth/login"
    register_url = f"{api_base}/auth/register"
    password = args.password or os.urandom(12).hex()

    # Try login first
    code, data = http_post_json(login_url, {"email": args.email, "password": password})
    if code == 200:
        token = data.get("token")
        user = data.get("user", {})
        print("[verify] Logged in successfully")
    elif code == 401:
        # Register
        print("[verify] Login failed (expected for new user). Registering...")
        code, data = http_post_json(
            register_url,
            {"email": args.email, "name": args.name, "password": password},
        )
        if code != 201:
            eprint("[verify] Register failed:", code, data)
            raise SystemExit(3)
        token = data.get("token")
        user = data.get("user", {})
        print("[verify] Registered successfully")
    else:
        eprint("[verify] Unexpected login response:", code, data)
        raise SystemExit(4)

    # 4) Verify admin role immediately
    role = (user or {}).get("role")
    if role != "admin":
        eprint(f"[verify] Expected role 'admin' after promotion, got: {role!r}")
        raise SystemExit(5)
    print("[verify] User role is admin (immediate)")

    # 5) Sanity check /me with the token
    me_url = f"{api_base}/auth/me"
    req = urllib.request.Request(me_url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        me = json.loads(resp.read().decode("utf-8"))
        if me.get("role") != "admin":
            eprint("[verify] /auth/me did not return admin role")
            raise SystemExit(6)
    print("[verify] /auth/me confirms admin role")

    print("[verify] SUCCESS — login-time admin promotion works as expected")


if __name__ == "__main__":
    main()

