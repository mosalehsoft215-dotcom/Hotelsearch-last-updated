"""Embeddings for the Graphiti backend.

Graphiti needs two things from a provider: a chat model for extraction, and an
embedder for vectors. OpenRouter serves chat completions only — it has no
embeddings endpoint — so pointing the OpenAI embedder at it returns 404 and the
graph never builds.

So the default here is a local embedder: a deterministic hashed bag-of-words
vector, no API call, no model download, no extra key. It is lexical, not truly
semantic — "breakfast included" and "morning meal" will not land near each other
the way a trained model would put them. Graphiti's retrieval is hybrid (vector +
BM25 + a graph walk), so search still works; the vector part is just weaker.

Set GRAPHITI_EMBEDDER=openai with a real embeddings key when quality matters.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any

EMBEDDING_DIM = 1024
_WORD = re.compile(r"[a-z0-9]+")


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def embed_text(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Hash each word into a bucket, count it, then scale to unit length so
    cosine similarity behaves. Same text always gives the same vector."""
    vector = [0.0] * dim
    tokens = _WORD.findall((text or "").lower())
    for token in tokens:
        vector[_bucket(token, dim)] += 1.0
        if len(token) > 4:                      # a crude stem, so plurals overlap
            vector[_bucket(token[:4], dim)] += 0.5
    length = math.sqrt(sum(v * v for v in vector))
    if length == 0.0:
        return vector
    return [v / length for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def build_embedder(settings: Any):
    """Return an EmbedderClient for Graphiti. Local unless told otherwise."""
    from graphiti_core.embedder.client import EmbedderClient, EmbedderConfig

    dim = getattr(settings, "graphiti_embedding_dim", EMBEDDING_DIM)

    if getattr(settings, "graphiti_embedder", "local") == "openai":
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        if not settings.graphiti_embedder_api_key:
            raise RuntimeError("GRAPHITI_EMBEDDER=openai needs GRAPHITI_EMBEDDER_API_KEY")
        return OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key=settings.graphiti_embedder_api_key,
            base_url=settings.graphiti_embedder_base_url,
            embedding_model=settings.graphiti_embedder_model))

    class LocalHashEmbedder(EmbedderClient):
        """No network, no weights — deterministic and good enough for a demo."""

        def __init__(self, dimension: int) -> None:
            self.config = EmbedderConfig(embedding_dim=dimension)
            self.dimension = dimension

        async def create(self, input_data: Any) -> list[float]:
            if isinstance(input_data, str):
                return embed_text(input_data, self.dimension)
            if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
                return embed_text(" ".join(input_data), self.dimension)
            return embed_text(str(input_data), self.dimension)

        async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
            return [embed_text(text, self.dimension) for text in input_data_list]

    return LocalHashEmbedder(dim)
