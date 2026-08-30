"""Read the live GraphQL schema, so a query is checked against it rather than
against memory.

Every hotel query in `hotel_tools.py` names the type it selects from. When one of
those types changes shape, the call fails at runtime with a message that names
the field but not the fix — "field 'optionRefId' not found in type: 'RoomSearch'"
is the whole error. This prints the type, and the answer is in the listing.

    python scripts/introspect.py --type RoomSearch HotelObj
    python scripts/introspect.py --input HotelCriteriaSearchInput
    python scripts/introspect.py --query hotel          # query fields matching "hotel"
    python scripts/introspect.py --dump                 # scripts/schema_full.json

Reads YARVEL_URL / YARVEL_SECRET / TRIPON_SENDER_IP from .env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings

_TYPE_REF = """name kind ofType { name kind ofType { name kind ofType { name kind } } }"""

_Q_TYPE = f"""
query IntrospectType($name: String!) {{
  __type(name: $name) {{
    name kind description
    fields {{ name description type {{ {_TYPE_REF} }} }}
    enumValues {{ name }}
  }}
}}"""

_Q_INPUT = f"""
query IntrospectInput($name: String!) {{
  __type(name: $name) {{
    name kind
    inputFields {{ name type {{ {_TYPE_REF} }} }}
  }}
}}"""

_Q_QUERY_FIELDS = f"""
query QueryFields {{
  __schema {{ queryType {{ fields {{
    name
    args {{ name type {{ {_TYPE_REF} }} }}
    type {{ {_TYPE_REF} }}
  }} }} }}
}}"""

_Q_FULL = """
query FullSchema {
  __schema {
    types {
      name kind
      fields { name type { name kind ofType { name kind ofType { name kind } } } }
      inputFields { name type { name kind ofType { name kind ofType { name kind } } } }
      enumValues { name }
    }
  }
}"""


def render(ref: dict | None) -> str:
    """A type reference as it is written in a query: [Foo!]! rather than a tree."""
    if not ref:
        return "?"
    kind = ref.get("kind")
    if kind == "NON_NULL":
        return render(ref.get("ofType")) + "!"
    if kind == "LIST":
        return "[" + render(ref.get("ofType")) + "]"
    return ref.get("name") or "?"


def is_leaf(ref: dict | None) -> bool:
    """Whether a field takes a subselection. SCALAR and ENUM must not have one —
    asking for `medias { url }` on a [String] is rejected outright."""
    while ref and ref.get("kind") in ("NON_NULL", "LIST"):
        ref = ref.get("ofType")
    return (ref or {}).get("kind") in ("SCALAR", "ENUM")


async def fetch(query: str, variables: dict | None = None) -> dict:
    settings = get_settings()
    if not (settings.yarvel_url and settings.yarvel_secret):
        raise SystemExit("YARVEL_URL / YARVEL_SECRET are not set in .env")
    headers = {"content-type": "application/json",
               "x-hasura-admin-secret": settings.yarvel_secret}
    if settings.sender_ip:
        headers["sender-ip"] = settings.sender_ip
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(settings.yarvel_url, headers=headers,
                                     json={"query": query, "variables": variables or {}})
    if response.status_code >= 400:
        raise SystemExit(f"HTTP {response.status_code}: {response.text[:300]}")
    body = response.json()
    if body.get("errors"):
        raise SystemExit(body["errors"][0].get("message"))
    return body["data"]


async def show_type(name: str) -> None:
    data = await fetch(_Q_TYPE, {"name": name})
    node = data.get("__type")
    if not node:
        print(f"{name}: no such type")
        return
    print(f"{node['name']} ({node['kind']})")
    for field in node.get("fields") or []:
        mark = "" if is_leaf(field["type"]) else "   <- takes a subselection"
        print(f"    {field['name']}: {render(field['type'])}{mark}")
    for value in node.get("enumValues") or []:
        print(f"    {value['name']}")
    print()


async def show_input(name: str) -> None:
    data = await fetch(_Q_INPUT, {"name": name})
    node = data.get("__type")
    if not node:
        print(f"{name}: no such input type")
        return
    print(f"{node['name']} ({node['kind']})")
    for field in node.get("inputFields") or []:
        print(f"    {field['name']}: {render(field['type'])}")
    print()


async def show_query_fields(needle: str) -> None:
    data = await fetch(_Q_QUERY_FIELDS)
    fields = data["__schema"]["queryType"]["fields"]
    for field in fields:
        if needle.lower() not in field["name"].lower():
            continue
        args = ", ".join(f"{a['name']}: {render(a['type'])}" for a in field["args"])
        print(f"{field['name']}({args}) -> {render(field['type'])}")


async def dump(path: Path) -> None:
    data = await fetch(_Q_FULL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {path} ({len(data['__schema']['types'])} types)")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", nargs="+", metavar="NAME", help="object/enum types to print")
    parser.add_argument("--input", nargs="+", metavar="NAME", help="input types to print")
    parser.add_argument("--query", metavar="SUBSTRING", help="query fields whose name contains this")
    parser.add_argument("--dump", action="store_true", help="write scripts/schema_full.json")
    args = parser.parse_args()

    if not any((args.type, args.input, args.query, args.dump)):
        parser.print_help()
        return 2
    for name in args.type or []:
        await show_type(name)
    for name in args.input or []:
        await show_input(name)
    if args.query:
        await show_query_fields(args.query)
    if args.dump:
        await dump(Path(__file__).resolve().parent / "schema_full.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
