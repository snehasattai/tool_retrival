"""Thin wrapper around the Gemini embedding API.

Kept separate from index.py so the embedding model/dimensionality can be
swapped (or pointed at a local sentence-transformers model for an
offline/no-API-cost setup) without touching any Chroma or retrieval code.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
_EMBED_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
# The Gemini free tier meters embed_content by request-per-minute, not by
# batch size, but large batches still occasionally trip rate limits. Small
# batches + a short pause between them + retry-with-backoff on 429s keeps a
# one-time 500+ tool seeding run reliable without needing a paid quota bump.
_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
_BATCH_PAUSE_SECONDS = float(os.getenv("EMBEDDING_BATCH_PAUSE_SECONDS", "1.0"))
_MAX_RETRIES = 6

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (copy .env.example to .env and fill it in)")
        _client = genai.Client(api_key=api_key)
    return _client


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _embed_with_retry(client: genai.Client, contents: list[str], task_type: str) -> list[list[float]]:
    delay = 2.0
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.models.embed_content(
                model=_EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=_EMBED_DIM),
            )
            return [e.values for e in resp.embeddings]
        except genai_errors.ClientError as exc:
            is_rate_limit = getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)
            if not is_rate_limit or attempt == _MAX_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    raise RuntimeError("unreachable")  # pragma: no cover


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of tool/document descriptions for indexing."""
    client = _get_client()
    vectors: list[list[float]] = []
    batches = list(_chunks(texts, _BATCH_SIZE))
    for i, batch in enumerate(batches):
        vectors.extend(_embed_with_retry(client, batch, "RETRIEVAL_DOCUMENT"))
        if i < len(batches) - 1:
            time.sleep(_BATCH_PAUSE_SECONDS)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single user query for similarity search against the index."""
    client = _get_client()
    return _embed_with_retry(client, [text], "RETRIEVAL_QUERY")[0]
