"""Obtain a Rihla JWT via loginRihla and decode its claims.

Same flow as the manual curl: admin secret + sender-ip + origin=agency ->
access_token -> decode the payload for the user id. Reads YARVEL_* from .env.

    python login.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys

import httpx

from config import get_settings

_LOGIN = ("mutation($e: String!, $p: String!, $o: String!) "
          "{ loginRihla(email: $e, password: $p, origin: $o) { access_token } }")


async def get_token(email: str | None = None, password: str | None = None) -> str:
    s = get_settings()
    email = email or s.yarvel_username
    password = password or s.yarvel_password
    if not (s.yarvel_secret and email and password):
        raise RuntimeError("need YARVEL_SECRET + YARVEL_USERNAME + YARVEL_PASSWORD in .env")
    headers = {"x-hasura-admin-secret": s.yarvel_secret, "content-type": "application/json"}
    if s.sender_ip:
        headers["sender-ip"] = s.sender_ip   # loginRihla rejects requests without it
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(s.yarvel_url, headers=headers,
                              json={"query": _LOGIN, "variables": {"e": email, "p": password, "o": "agency"}})
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"][0].get("message", "login failed"))
    token = ((body.get("data") or {}).get("loginRihla") or {}).get("access_token")
    if not token:
        raise RuntimeError(f"no access_token in response: {str(body)[:200]}")
    return token


def decode_claims(token: str) -> dict:
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def user_id(claims: dict) -> str | None:
    ns = claims.get("https://hasura.io/jwt/claims") or {}
    return (ns.get("x-hasura-user-id") or claims.get("x-hasura-user-id")
            or claims.get("user_id") or claims.get("sub"))


async def main() -> int:
    token = await get_token()
    claims = decode_claims(token)
    print("access_token:", token[:40] + "…")
    print("user id:", user_id(claims))
    print("claims:", json.dumps(claims, indent=2)[:800])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
