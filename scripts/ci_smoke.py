"""
Minimal CI smoke test against a running API instance.

Checks:
- GET /api/health returns 200 and JSON
- POST /api/auth/register creates a user
- POST /api/auth/login returns a JWT
- GET /api/auth/me with the token returns the same email

Usage:
  python backend/scripts/ci_smoke.py --base http://localhost:8000/api
"""

from __future__ import annotations

import argparse
import os
import random
import string

import httpx


def _rand_email() -> str:
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"ci-smoke-{rand}@chatfleet.local"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("API_BASE", "http://localhost:8000/api"))
    args = ap.parse_args()

    base = args.base.rstrip("/")
    email = _rand_email()
    password = "Passw0rd!test"

    with httpx.Client(timeout=10.0) as client:
        # Health
        r = client.get(f"{base}/health")
        r.raise_for_status()
        assert r.headers.get("content-type", "").startswith("application/json")

        # Register
        r = client.post(
            f"{base}/auth/register",
            json={"email": email, "password": password, "name": "CI Smoke"},
        )
        r.raise_for_status()

        # Login
        r = client.post(
            f"{base}/auth/login",
            json={"email": email, "password": password},
        )
        r.raise_for_status()
        token = r.json()["token"]

        # Me
        r = client.get(f"{base}/auth/me", headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        assert r.json()["email"].lower() == email.lower()

    print("SMOKE_OK", email)


if __name__ == "__main__":
    main()

