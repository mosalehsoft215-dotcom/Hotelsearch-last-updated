"""Free-text search over everything enrichment has already fetched.

`enrich_hotel_info` and `enrich_destination` are exact-key lookups: you must know
the subject and the domain before you can ask. That is fine when the agent is
working through a hotel it just found, and useless when someone asks "which of
these places has a pool problem" or "what did we learn about Jeddah in September".

So every claim is embedded as it is written, and this searches those embeddings.

Two decisions worth stating:

* One row per claim. Claims arrive already split at their natural boundary — a
  field, a value, its sources — so there is nothing to chunk. Splitting them
  further would only break a fact away from its citation.
* Embedded at write time, in the same call that stores the claim, so an
  embedding can never be older than the claim it describes. A nightly batch would
  reopen exactly that gap.

The vectors come from the same hashed embedder the memory layer uses, so there is
one embedding scheme in the repo rather than two.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from graphiti_embedder import embed_text

INDEX_VERSION = 2

# Cosine over a bag of words is never exactly zero: an unrelated question still
# shares "it", "the", a stray digit. Measured on real claims, a question that
# belongs to a domain scores 0.74-0.85 and one that does not scores 0.03-0.13 —
# two clearly separated bands, because expand() puts a whole domain vocabulary on
# both sides of a genuine match. Anything under this floor is that second band,
# and returning it would hand the agent a sourced-looking claim about nothing.
MIN_SCORE = 0.35

# The embedder matches on words, not on meaning, so "how warm will it be" shares
# nothing with "29.2-34.5°C" and finds nothing. There are only seven domains and
# their field names are ours, so the words people actually use for each one are
# written down here and added to both sides — the stored fact and the question.
# This is a vocabulary, not comprehension: a word nobody listed will still miss.
DOMAIN_WORDS = {
    "weather": "weather forecast temperature degrees celsius hot warm cold cool "
               "rain rainy wet dry humid sunny climate",
    "reputation": "reputation rating rated review reviews score stars guests "
                  "opinion complaints complained praised recommended",
    "location": "location distance far near close nearby walk walking minutes "
                "airport landmark transport metro taxi central",
    "facilities": "facilities amenities pool swimming gym fitness spa parking "
                  "wifi internet restaurant breakfast family accessible",
    "risk": "risk renovation refurbishment refurbished closure closed construction "
            "works building rebranding ownership problem issue",
    "advisory": "advisory advice travel warning safety security government "
                "guidance restrictions entry",
    "news": "news recent events happening reported story update",
    "company_facts": "company corporate chain brand group owner owns owned ownership "
                     "parent subsidiary headquarters headquartered based legal name "
                     "founded established incorporated ceo chief executive website",
    "agency_facts": "agency agent travel operator company trading legal name country "
                    "headquarters address office website contact phone telephone email "
                    "accreditation accredited licence license iata atol registration",
}


def expand(text: str, domain: str | None = None) -> str:
    """Add the words that belong to a domain, so a question and a stored fact meet
    in the middle. On a fact we know the domain; on a question we look for one."""
    if domain and domain in DOMAIN_WORDS:
        return f"{text} {DOMAIN_WORDS[domain]}"
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = [terms for name, terms in DOMAIN_WORDS.items()
            if words & set(terms.split()) or name in words]
    return " ".join([text, *hits]) if hits else text
logger = logging.getLogger("tripon.enrichment.index")


@dataclass
class IndexedClaim:
    subject: str
    entity_type: str
    entity_ref: str
    domain: str
    field_name: str
    value: str
    status: str
    sources: list[dict[str, Any]]
    observed_at: str
    # When this claim stops being current, from the domain's own freshness
    # window. Without it the index served whatever it had ever been told,
    # forever — FRESH_FOR_SECONDS governed only the in-process re-fetch cache,
    # so a week-old rating and a three-hour-old forecast ranked alike.
    valid_until: str | None = None

    @property
    def is_stale(self) -> bool:
        if not self.valid_until:
            return False
        try:
            return datetime.fromisoformat(self.valid_until) < datetime.now(timezone.utc)
        except ValueError:
            return False

    @property
    def key(self) -> str:
        return f"{self.entity_type}|{self.entity_ref.lower()}|{self.domain}|{self.field_name}"

    @property
    def text(self) -> str:
        """What gets embedded: the subject, the domain and field, the value, and
        the domain's everyday words so a question phrased normally can reach it."""
        plain = f"{self.subject}. {self.domain}. {self.field_name.replace('_', ' ')}: {self.value}"
        return expand(plain, self.domain)

    def to_model(self, score: float) -> dict[str, Any]:
        return {"subject": self.subject, "entity_type": self.entity_type,
                "entity_ref": self.entity_ref, "domain": self.domain, "field": self.field_name,
                "value": self.value, "status": self.status, "sources": self.sources,
                "observed_at": self.observed_at, "valid_until": self.valid_until,
                "is_stale": self.is_stale, "match": round(score, 3)}


# Words that start with a capital in a question without naming a place.
_NOT_A_PLACE = frozenset("""
january february march april may june july august september october november
december monday tuesday wednesday thursday friday saturday sunday what when
where which how why who is are do does can could would will the a an my our
i it there hotel hotels room rooms night nights price prices weather forecast
""".split())

_CAPITALISED = re.compile(r"([A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*)")


def mentioned_entities(question: str) -> list[str]:
    """Proper nouns a question names — the city or hotel it is asking about.

    Cheap on purpose: capitalised runs, minus the words that start a sentence or
    name a month. It only has to be good enough to notice that "the weather in
    Aswan" is not a question about Jeddah.
    """
    phrases: list[str] = []
    singles: list[str] = []
    for phrase in _CAPITALISED.findall(question):
        # "What's" is "what" with a contraction stuck to it, and "what" is in
        # the list. Strip that before deciding, or every question beginning
        # "What's ..." names a place called What's.
        words = [w for w in phrase.split()
                 if re.sub(r"'\w*$", "", w).lower() not in _NOT_A_PLACE]
        if not words:
            continue
        if len(words) > 1:
            phrases.append(" ".join(words))
        # Each word on its own as well. A capitalised run is not always one
        # name: "For Makkah, will it be hot" gave the single candidate "For
        # Makkah", which matched no stored entity, so a question naming a city
        # the index held retrieved nothing at all and the agent answered from
        # the transcript instead. Phrases first, so a real multi-word name like
        # "Carawan Hotel" still wins over its parts.
        singles.extend(words)
    return [*phrases, *singles]


def resolve_entity(named: Iterable[str], known_refs: Iterable[str]) -> str | None:
    """Match what a question calls something to what the index stored it as.

    Exact first. Then on whole words, because a question almost never repeats a
    registered name in full: "who owns Hilton" against a record stored as
    "Hilton Worldwide", or "where is Acme Hotels based" — which reduces to the
    single candidate "Acme", since "hotels" is one of the generic words stripped
    before matching. Requiring equality meant every multi-word entity was
    unreachable unless the question spelled it out, and company and agency names
    are nearly all multi-word.

    Every word of the shorter name must appear in the longer one, so "Aswan"
    still fails to reach a record about Jeddah — the separation this guard
    exists for is unaffected.
    """
    known = {ref.lower(): ref for ref in sorted(known_refs)}   # sorted: one answer, not any
    for candidate in named:
        if candidate.lower() in known:
            return known[candidate.lower()]
    for candidate in named:
        words = set(re.findall(r"[\w']+", candidate.lower()))
        if not words:
            continue
        for lowered, ref in known.items():
            ref_words = set(re.findall(r"[\w']+", lowered))
            if ref_words and (words <= ref_words or ref_words <= words):
                return ref
    return None


_FORECAST_FIELD = re.compile(r"^forecast_(\d{4}-\d{2}-\d{2})$")


def _is_past_forecast(claim: "IndexedClaim") -> bool:
    """A forecast for a day that has already happened cannot answer a question
    about a stay. It stays in the index as a record; it is not offered as an
    answer."""
    match = _FORECAST_FIELD.match(claim.field_name or "")
    if not match:
        return False
    try:
        return date.fromisoformat(match.group(1)) < date.today()
    except ValueError:
        return False


class VectorStore(Protocol):
    def upsert(self, claims: Iterable[tuple[IndexedClaim, list[float]]]) -> int: ...

    def entity_refs(self) -> set[str]: ...

    def retire_superseded(self, entity_type: str, entity_ref: str, domain: str,
                          keep: set[str], observed_at: str) -> int: ...

    def nearest(self, vector: list[float], limit: int, subject: str | None,
                domain: str | None, entity_type: str | None = None,
                entity_ref: str | None = None,
                min_score: float = MIN_SCORE) -> list[tuple[float, IndexedClaim]]: ...

    def count(self) -> int: ...


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))       # both sides arrive unit length


class SqliteVectorStore:
    """Rows in SQLite, similarity in Python.

    A dedicated vector database earns its keep at a scale this does not have —
    enrichment records for the hotels and cities in play, not a document corpus.
    SQLite adds nothing to deploy, back up or secure, and `VectorStore` is the
    seam to move behind pgvector or anything else the day the numbers change.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                key TEXT PRIMARY KEY, subject TEXT, entity_type TEXT, entity_ref TEXT,
                domain TEXT, field TEXT, value TEXT, status TEXT, sources TEXT,
                observed_at TEXT, vector TEXT, version INTEGER, valid_until TEXT)""")
        # An index written before valid_until existed keeps its rows; they read
        # back with valid_until NULL, which means "no expiry known" rather than
        # "expired", so nothing that was already there disappears.
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(claims)")}
        if "valid_until" not in columns:
            self._db.execute("ALTER TABLE claims ADD COLUMN valid_until TEXT")
        self._backfill_valid_until()
        self._db.execute("CREATE INDEX IF NOT EXISTS claims_entity ON claims(entity_type, entity_ref)")
        self._db.execute("CREATE INDEX IF NOT EXISTS claims_group "
                         "ON claims(entity_type, entity_ref, domain)")
        self._db.commit()

    def _backfill_valid_until(self) -> int:
        """Give rows written before the column existed the expiry they should
        have had.

        Leaving them NULL would have read as "never expires", so the claims
        already in an index — the ones actually being served today — would have
        gone on being served. Both inputs are on record: the domain's window and
        the moment the claim was observed.
        """
        from web_enrich import FRESH_FOR_SECONDS

        rows = list(self._db.execute(
            "SELECT key, domain, observed_at FROM claims WHERE valid_until IS NULL"))
        updates = []
        for key, domain, observed_at in rows:
            fresh_for = FRESH_FOR_SECONDS.get(domain)
            if not (fresh_for and observed_at):
                continue
            try:
                seen = datetime.fromisoformat(observed_at)
            except (TypeError, ValueError):
                continue
            updates.append(((seen + timedelta(seconds=fresh_for)).isoformat(), key))
        if updates:
            self._db.executemany("UPDATE claims SET valid_until = ? WHERE key = ?", updates)
            self._db.commit()
            logger.info("backfilled valid_until on %d claim(s)", len(updates))
        return len(updates)

    def close(self) -> None:
        self._db.close()

    def upsert(self, claims: Iterable[tuple[IndexedClaim, list[float]]]) -> int:
        rows = [(c.key, c.subject, c.entity_type, c.entity_ref, c.domain, c.field_name,
                 c.value, c.status, json.dumps(c.sources), c.observed_at,
                 json.dumps(v), INDEX_VERSION, c.valid_until) for c, v in claims]
        if not rows:
            return 0
        # Columns named rather than positional, so adding one cannot silently
        # shift every value one place to the left.
        self._db.executemany(
            "INSERT INTO claims (key, subject, entity_type, entity_ref, domain, field, "
            "value, status, sources, observed_at, vector, version, valid_until) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, status=excluded.status, sources=excluded.sources, "
            "observed_at=excluded.observed_at, vector=excluded.vector, "
            "version=excluded.version, valid_until=excluded.valid_until",
            rows)
        self._db.commit()
        return len(rows)

    def retire_superseded(self, entity_type: str, entity_ref: str, domain: str,
                          keep: set[str], observed_at: str) -> int:
        """Drop what the previous fetch of this entity and domain left behind.

        A fetch is a snapshot of one subject in one domain. Keying rows by field
        meant a September window and an October window for the same city both
        persisted and competed, which is how a question about one stay was
        answered with the other one's forecast. The newest fetch is authoritative
        for that pair; anything older it did not re-state is retired.
        """
        placeholders = ",".join("?" * len(keep)) if keep else "''"
        cursor = self._db.execute(
            f"DELETE FROM claims WHERE entity_type = ? AND lower(entity_ref) = ? "
            f"AND domain = ? AND observed_at < ? AND key NOT IN ({placeholders})",
            [entity_type, entity_ref.lower(), domain, observed_at, *keep])
        self._db.commit()
        return cursor.rowcount or 0

    def nearest(self, vector: list[float], limit: int, subject: str | None = None,
                domain: str | None = None, entity_type: str | None = None,
                entity_ref: str | None = None,
                min_score: float = MIN_SCORE) -> list[tuple[float, IndexedClaim]]:
        sql = ("SELECT subject, entity_type, entity_ref, domain, field, value, status, "
               "sources, observed_at, vector, valid_until FROM claims")
        where, params = [], []
        if subject:
            where.append("lower(subject) LIKE ?")
            params.append(f"%{subject.lower()}%")
        if entity_type:
            where.append("entity_type = ?")
            params.append(entity_type)
        if entity_ref:
            where.append("lower(entity_ref) = ?")
            params.append(entity_ref.lower())
        if domain:
            where.append("domain = ?")
            params.append(domain)
        if where:
            sql += " WHERE " + " AND ".join(where)
        scored: list[tuple[float, IndexedClaim]] = []
        for row in self._db.execute(sql, params):
            claim = IndexedClaim(subject=row[0], entity_type=row[1], entity_ref=row[2],
                                 domain=row[3], field_name=row[4], value=row[5],
                                 status=row[6], sources=json.loads(row[7]), observed_at=row[8],
                                 valid_until=row[10])
            scored.append((cosine(vector, json.loads(row[9])), claim))
        # Score, then recency. Lexical similarity cannot tell yesterday's
        # forecast for the wrong week from today's for the right one, so the two
        # windows tied at 0.739 and came back interleaved in row order. The
        # fresher observation wins a tie.
        scored.sort(key=lambda pair: (pair[0], pair[1].observed_at or ""), reverse=True)
        # A limit of 0 or less would return nothing and read as "no match", which
        # the agent then states as fact. One result is the floor.
        return [pair for pair in scored[:max(1, limit)] if pair[0] >= min_score]

    def entity_refs(self) -> set[str]:
        """Every entity the index holds, so a question can be checked against
        what is actually stored rather than against a vocabulary."""
        rows = self._db.execute("SELECT DISTINCT entity_ref FROM claims").fetchall()
        return {r[0] for r in rows if r[0]}

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM claims").fetchone()[0]


def _by_group(rows: list[tuple[IndexedClaim, list[float]]]
              ) -> dict[tuple[str, str, str], list[IndexedClaim]]:
    """One fetch can span several fields of one subject and domain; that triple
    is the unit a later fetch supersedes."""
    grouped: dict[tuple[str, str, str], list[IndexedClaim]] = {}
    for claim, _ in rows:
        grouped.setdefault((claim.entity_type, claim.entity_ref, claim.domain), []).append(claim)
    return grouped


class EnrichmentIndex:
    """Writes claims in as they are fetched, and answers questions about them.

    The store opens on first use rather than at import, so a read-only working
    directory costs a warning and an in-memory index instead of a service that
    will not start.
    """

    def __init__(self, store: VectorStore | None = None, path: str | Path | None = None) -> None:
        self._store = store
        self._path = path

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            try:
                self._store = SqliteVectorStore(self._path or ":memory:")
            except sqlite3.Error as exc:
                logger.warning("enrichment index falling back to memory: %s", exc)
                self._store = SqliteVectorStore(":memory:")
        return self._store

    def add(self, enrichment: Any) -> int:
        """Called from the enricher once a result has been assessed. Anything
        without a source is skipped — an unsourced claim is not worth finding.

        Each claim is stamped with the moment it stops being current, and the
        write retires whatever the previous fetch of the same subject and domain
        left behind. Both were missing: the index accepted every claim it was
        ever given and kept them all, so a stay in one week could be answered
        with a forecast fetched for another.
        """
        from web_enrich import FRESH_FOR_SECONDS

        rows: list[tuple[IndexedClaim, list[float]]] = []
        for claim in getattr(enrichment, "claims", []) or []:
            if not claim.sources:
                continue
            fresh_for = FRESH_FOR_SECONDS.get(claim.domain)
            indexed = IndexedClaim(
                subject=enrichment.subject,
                entity_type=getattr(enrichment, "entity_type", "subject"),
                entity_ref=getattr(enrichment, "entity_ref", "") or enrichment.subject,
                domain=claim.domain, field_name=claim.field_name,
                value=claim.value, status=claim.status,
                sources=[{"url": s.url, "title": s.title, "tier": s.tier} for s in claim.sources],
                observed_at=claim.observed_at.isoformat(),
                valid_until=(claim.observed_at + timedelta(seconds=fresh_for)).isoformat()
                            if fresh_for else None)
            rows.append((indexed, embed_text(indexed.text)))
        written = self.store.upsert(rows)
        for (entity_type, entity_ref, domain), group in _by_group(rows).items():
            newest = max(claim.observed_at for claim in group)
            retired = self.store.retire_superseded(
                entity_type, entity_ref, domain, {claim.key for claim in group}, newest)
            if retired:
                logger.info("retired %d superseded %s claim(s) for %s:%s",
                            retired, domain, entity_type, entity_ref)
        return written

    def search(self, question: str, limit: int = 5, subject: str | None = None,
               domain: str | None = None, entity_type: str | None = None,
               entity_ref: str | None = None,
               min_score: float = MIN_SCORE,
               include_stale: bool = False) -> list[dict[str, Any]]:
        if not question.strip():
            return []
        # expand() puts a whole domain vocabulary on both sides, which is what
        # lets "how warm will it be" reach "29.2-34.5°C" — and also what made
        # "the weather in Aswan" score 0.757 against Jeddah. The floor separates
        # on-domain from off-domain; it cannot separate one city from another.
        # The names the question uses can.
        if entity_ref is None:
            named = mentioned_entities(question)
            if named:
                entity_ref = resolve_entity(named, self.store.entity_refs())
                if entity_ref is None:
                    return []      # asked about something this index has never seen
        # Ask for more than requested, drop what cannot answer, then trim. A
        # forecast for a date already past is dead weight that would otherwise
        # occupy one of the caller's slots forever.
        hits = self.store.nearest(embed_text(expand(question, domain)),
                                  max(1, limit) * 4, subject,
                                  domain, entity_type, entity_ref, min_score)
        usable = [(score, claim) for score, claim in hits
                  if not _is_past_forecast(claim)
                  and (include_stale or not claim.is_stale)]
        return [claim.to_model(score) for score, claim in usable[:max(1, limit)]]

    def size(self) -> int:
        return self.store.count()
