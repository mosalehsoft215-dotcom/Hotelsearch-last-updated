"""Check the Hasura endpoint is reachable, the credential works, and data comes
back. Reads YARVEL_URL / YARVEL_SECRET (and TRIPON_SENDER_IP) from .env.

    python healthcheck.py
"""
from __future__ import annotations

import asyncio
import json
import sys

import httpx

from config import get_settings

_CHECKS = [
    ("connection + auth", "query { __typename }", {}, "__typename"),
    ("queue table access",
     "query { Core_BookingQueueStatus(limit: 1){ Id Status } }", {}, "Core_BookingQueueStatus"),
]


async def main() -> int:
    s = get_settings()
    if not s.yarvel_url or not s.yarvel_secret:
        print("FAIL  YARVEL_URL / YARVEL_SECRET not set in .env")
        return 2

    headers = {"content-type": "application/json", "x-hasura-admin-secret": s.yarvel_secret}
    if s.sender_ip:
        headers["sender-ip"] = s.sender_ip

    print(f"endpoint: {s.yarvel_url}")
    ok = True
    async with httpx.AsyncClient(timeout=20) as client:
        for label, query, variables, field in _CHECKS:
            try:
                r = await client.post(s.yarvel_url, headers=headers,
                                      json={"query": query, "variables": variables})
            except httpx.HTTPError as exc:
                print(f"FAIL  {label}: cannot connect — {exc}"); ok = False; continue
            if r.status_code >= 400:
                print(f"FAIL  {label}: HTTP {r.status_code} — {r.text[:160]}"); ok = False; continue
            body = r.json()
            if body.get("errors"):
                print(f"FAIL  {label}: {body['errors'][0].get('message')}"); ok = False; continue
            data = (body.get("data") or {}).get(field)
            sample = json.dumps(data)[:120] if data is not None else "null"
            print(f"OK    {label}: {sample}")
    try:
        from login import get_token, decode_claims, user_id
        if s.yarvel_username and s.yarvel_password:
            tok = await get_token()
            print(f"OK    loginRihla (JWT): user id = {user_id(decode_claims(tok))}")
        else:
            print("SKIP  loginRihla (JWT): set YARVEL_USERNAME / YARVEL_PASSWORD to test it")
    except Exception as exc:
        print(f"FAIL  loginRihla (JWT): {exc}"); ok = False

    print("\nALL GOOD" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
