"""A regex that cannot match is worse than a missing one: it reads as a check.

`_QUOTED_MONEY` shipped for months with a literal backspace where each `\\b`
belonged, because a patch wrote the pattern through a non-raw string. Both
currency-code alternatives were dead, so the price cross-check only ever saw
"$300" — never the "134.98 USD" form the agent actually writes. Nothing failed;
the guard was simply not there.

These tests are cheap and they hold the two things that went wrong: no stray
control characters in source, and the patterns that matter still match the text
they were written for.
"""
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Anything a source file has no business containing. \t (\x09), \n (\x0a) and
# \r (\x0d) are legitimate; the rest are the residue of a mangled escape.
CONTROL_CLASS = re.compile(b"[" + bytes(range(0, 9)) + bytes([0x0b, 0x0c])
                           + bytes(range(0x0e, 0x20)) + b"]")


def source_files():
    files = [p for p in ROOT.rglob("*.py")
             if ".venv" not in p.parts and "__pycache__" not in p.parts]
    files += [ROOT / "chat_ui.html", ROOT / "chat_render.js"]
    return [p for p in files if p.exists()]


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_no_source_file_carries_a_stray_control_character(path):
    """A backspace in a pattern is invisible in every editor and in every diff.
    It is only ever the wreck of an escape that lost its backslash."""
    found = CONTROL_CLASS.findall(path.read_bytes())
    assert not found, (
        f"{path.name} contains {len(found)} control character(s) {sorted(set(found))} — "
        "almost certainly a regex escape written through a non-raw string")


def test_the_money_pattern_matches_the_forms_the_agent_writes():
    """The live answer says "134.98 USD". If that does not match, the price
    cross-check is decoration."""
    from agents.hotel_search_agent import _QUOTED_MONEY

    for text, expected in [("134.98 USD", "134.98"), ("300 USD", "300"),
                           ("USD 300", "300"), ("$300", "300"),
                           ("1,234.56 SAR", "1,234.56"), ("SAR 1,234.56", "1,234.56"),
                           ("total is 90 EUR today", "90")]:
        found = [(m.group(1) or m.group(2)) for m in _QUOTED_MONEY.finditer(text)]
        assert found == [expected], f"{text!r} -> {found}"


def test_the_money_pattern_still_needs_a_word_boundary():
    """What the repaired \\b is for: a currency code glued to another word is
    not a price."""
    from agents.hotel_search_agent import _QUOTED_MONEY

    for text in ("300USD", "x300 USDy", "SARDINE 12", "12 USDX"):
        assert list(_QUOTED_MONEY.finditer(text)) == [], text


def test_every_regex_in_the_agent_and_runtime_compiles_and_is_not_inert():
    """A pattern whose alternatives can never match is the failure mode here, so
    each named pattern is asked to match something it was written for."""
    import agents.hotel_search_agent as agent
    import runtime

    samples = {
        "_ESTIMATED": "based on typical September patterns",
        "_CONFIRMED": "the confirmed price is 300 USD",
        "_NEGATED": "could not provide a",
        "_OPTION_REF": "33!~|a0",
        "_PROMISED_MEMORY": "I'll remember that",
        "_QUOTED_MONEY": "134.98 USD",
        "_QUOTED_MEASURE": "28-40 °C",
        "_DAY_REF": "2026-09-10",
        "_OFFICIAL_ADVISORY": "the official government travel advisory",
        "_MENTIONS_CHILDREN": "two children",
        "_STATES_AGE": "aged 8",
        "_AMBIGUOUS_ROOMS": "1 or 2 rooms",
    }
    for name, text in samples.items():
        pattern = getattr(agent, name)
        assert pattern.search(text), f"{name} no longer matches {text!r}"

    assert runtime._STORED_ONLY.search("using stored enrichment only")
    assert runtime._FETCH_PERMITTED.search("then fetch if missing")
    for pattern, _ in runtime._CLEAN_ERRORS:
        assert pattern.pattern, "an empty error pattern would swallow everything"
