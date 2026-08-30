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
from datetime import datetime, timezone
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
                "observed_at": self.observed_at, "match": round(score, 3)}


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
    found: list[str] = []
    for phrase in _CAPITALISED.findall(question):
        # "What's" is "what" with a contraction stuck to it, and "what" is in
        # the list. Strip that before deciding, or every question beginning
        # "What's ..." names a place called What's.
        words = [w for w in phrase.split()
                 if re.sub(r"'\w*$", "", w).lower() not in _NOT_A_PLACE]
        if words:
            found.append(" ".join(words))
    return found


class VectorStore(Protocol):
    def upsert(self, claims: Iterable[tuple[IndexedClaim, list[float]]]) -> int: ...

    def entity_refs(self) -> set[str]: ...

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
                observed_at TEXT, vector TEXT, version INTEGER)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS claims_entity ON claims(entity_type, entity_ref)")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def upsert(self, claims: Iterable[tuple[IndexedClaim, list[float]]]) -> int:
        rows = [(c.key, c.subject, c.entity_type, c.entity_ref, c.domain, c.field_name,
                 c.value, c.status, json.dumps(c.sources), c.observed_at,
                 json.dumps(v), INDEX_VERSION) for c, v in claims]
        if not rows:
            return 0
        self._db.executemany(
            "INSERT INTO claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "value=excluded.value, status=excluded.status, sources=excluded.sources, "
            "observed_at=excluded.observed_at, vector=excluded.vector, version=excluded.version",
            rows)
        self._db.commit()
        return len(rows)

    def nearest(self, vector: list[float], limit: int, subject: str | None = None,
                domain: str | None = None, entity_type: str | None = None,
                entity_ref: str | None = None,
                min_score: float = MIN_SCORE) -> list[tuple[float, IndexedClaim]]:
        sql = ("SELECT subject, entity_type, entity_ref, domain, field, value, status, "
               "sources, observed_at, vector FROM claims")
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
                                 status=row[6], sources=json.loads(row[7]), observed_at=row[8])
            scored.append((cosine(vector, json.loads(row[9])), claim))
        scored.sort(key=lambda pair: pair[0], reverse=True)
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
        without a source is skipped — an unsourced claim is not worth finding."""
        rows: list[tuple[IndexedClaim, list[float]]] = []
        for claim in getattr(enrichment, "claims", []) or []:
            if not claim.sources:
                continue
            indexed = IndexedClaim(
                subject=enrichment.subject,
                entity_type=getattr(enrichment, "entity_type", "subject"),
                entity_ref=getattr(enrichment, "entity_ref", "") or enrichment.subject,
                domain=claim.domain, field_name=claim.field_name,
                value=claim.value, status=claim.status,
                sources=[{"url": s.url, "title": s.title, "tier": s.tier} for s in claim.sources],
                observed_at=claim.observed_at.isoformat())
            rows.append((indexed, embed_text(indexed.text)))
        return self.store.upsert(rows)

    def search(self, question: str, limit: int = 5, subject: str | None = None,
               domain: str | None = None, entity_type: str | None = None,
               entity_ref: str | None = None,
               min_score: float = MIN_SCORE) -> list[dict[str, Any]]:
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
                known = {ref.lower(): ref for ref in self.store.entity_refs()}
                matched = [known[n.lower()] for n in named if n.lower() in known]
                if matched:
                    entity_ref = matched[0]
                else:
                    return []      # asked about something this index has never seen
        hits = self.store.nearest(embed_text(expand(question, domain)), limit, subject,
                                  domain, entity_type, entity_ref, min_score)
        return [claim.to_model(score) for score, claim in hits]

    def size(self) -> int:
        return self.store.count()
