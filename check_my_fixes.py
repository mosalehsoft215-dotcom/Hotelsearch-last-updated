"""Which of the fixes are actually in this working copy?

Two lines of work were done on this codebase in parallel — GraphQL query
corrections in one copy, runtime and console fixes in another. Claude Code
reported "no new Python in the zip", which is true of the *old* zip and false of
the later ones, so the only way to know what this copy has is to look.

Run from the project root:

    python check_my_fixes.py

Each row names a marker in the source. Present means the fix is here.
Nothing is modified.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

CHECKS: list[tuple[str, str, str, str]] = [
    # (file, marker, what it is, why it matters)
    ("runtime.py", '"role": "assistant", "content": output',
     "multi-turn history",
     "without it the agent has no record of its own replies; turn 2 sees two user turns"),

    ("api.py", "tools_called_session",
     "per-turn tool chips",
     "otherwise the chip row shows every tool the session ever called"),
    ("api.py", "async def enrichment_view",
     "GET /enrichment",
     "the panel that shows match score and source; no endpoint, no panel"),
    ("api.py", "async def delegate_view",
     "POST /delegate",
     "the only path that exercises context isolation from the console"),
    ("api.py", "parent_unchanged",
     "isolation evidence in the response",
     "the before/after snapshot the demo line is read from"),
    ("api.py", "jsonable_encoder",
     "JSON contract",
     "without it an unencodable value returns a plain-text 500"),

    ("web_tools.py", "def index_stats",
     "index size accessor",
     "the panel header count"),
    ("web_tools.py", "minScore",
     "score floor passthrough",
     "lets a caller see what the floor filtered"),

    ("enrichment_index.py", "MIN_SCORE",
     "noise floor",
     "without it 'is it muggy' returns a weather claim with a real citation"),
    ("enrichment_index.py", "max(1, limit)",
     "limit clamp",
     "limit<=0 silently returned 'nothing found', which the agent states as fact"),

    ("web_enrich.py", "coverage_gap",
     "forecast coverage claim",
     "makes the API state when it did not cover the dates asked for"),

    ("agents/hotel_search_agent.py", "_ESTIMATED",
     "estimation check",
     "catches 'based on typical September patterns' passing as verified"),
    ("agents/hotel_search_agent.py", "_CONFIRMED",
     "unconfirmed-rate check",
     "catches a failed reprice being reported as a Confirmed Price"),
    ("agents/hotel_search_agent.py", "_OPTION_REF",
     "option reference guard",
     "stops the raw 33!~|a0!~|... string reaching the user"),
    ("agents/hotel_search_agent.py", "_PROMISED_MEMORY",
     "promised-memory check",
     "catches \"I'll remember that\" with no remember_preference call"),
    ("agents/hotel_search_agent.py", "Never fill a gap from your own knowledge",
     "no-estimation prompt rule",
     "the instruction the check above enforces"),
    ("agents/hotel_search_agent.py", "Decide first whether the message is a request to find hotels",
     "search-vs-enrichment rule",
     "keeps retrieval-first from swallowing an ordinary hotel search"),

    ("chat_ui.html", "function showPanel",
     "one panel at a time",
     "three stacked panels push the conversation off screen"),
    ("chat_ui.html", ".delrow[hidden]{display:none}",
     "delegate row hides",
     "an author display:flex beats the UA sheet's [hidden]"),
    ("chat_ui.html", "function safeLink",
     "link scheme whitelist",
     "a javascript: source URL would otherwise be clickable"),
    ("chat_ui.html", "function readable",
     "readable errors",
     "a 402 otherwise dumps a page of JSON mid-demo"),
    ("chat_ui.html", "function busy",
     "send/delegate lock",
     "a delegation during a chat reports a false 'parent CHANGED'"),
]


def main() -> int:
    root = Path.cwd()
    if not (root / "api.py").exists():
        print("Run this from the project root (the folder holding api.py).")
        return 2

    cache: dict[str, str] = {}
    missing: list[tuple[str, str, str]] = []
    present = 0

    print(f"\n  checking {root}\n")
    last_file = None
    for filename, marker, name, why in CHECKS:
        path = root / filename
        if filename not in cache:
            cache[filename] = path.read_text(encoding="utf-8") if path.exists() else ""
        if filename != last_file:
            print(f"  {filename}")
            last_file = filename
        ok = marker in cache[filename]
        print(f"    {'PRESENT' if ok else 'MISSING'}   {name}")
        if ok:
            present += 1
        else:
            missing.append((filename, name, why))

    print(f"\n  {present}/{len(CHECKS)} present\n")

    if missing:
        print("  Missing, and what each one lets through:\n")
        for filename, name, why in missing:
            print(f"    {name}  ({filename})")
            print(f"      {why}\n")
        print("  These are additive — none of them touches the GraphQL queries or")
        print("  models, so applying them will not undo the query corrections.\n")
        return 1

    print("  Everything is here. The suite should report 260 passed, 4 skipped.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
