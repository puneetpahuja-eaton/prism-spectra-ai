"""Explicit Nebius, Google, or local document embeddings.

Requests are batched and retried with exponential backoff so bulk ingestion
tolerates per-minute embedding quota limits (HTTP 429 RESOURCE_EXHAUSTED).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time

from gemini_client import ModelUnavailable, is_google_live, nebius_post

_DIM = 256
_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
_NEBIUS_MODEL = "Qwen/Qwen3-Embedding-8B"
_TOKEN = re.compile(r"[a-z0-9]+")
_BATCH = int(os.getenv("EMBED_BATCH", "16"))
_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))
# text-embedding-004 allows up to 20,000 input tokens per request across the batch.
# Budget by characters (code tokenizes densely); keep a safe margin under the limit.
_CHAR_BUDGET = int(os.getenv("EMBED_CHAR_BUDGET", "24000"))


def _provider() -> str:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider and provider not in {"nebius", "google", "local"}:
        raise ModelUnavailable("EMBEDDING_PROVIDER must be 'nebius', 'google', or 'local'.")
    if os.getenv("FORCE_MOCK") == "1":
        return "local"
    if provider:
        return provider
    if os.getenv("AI_PROVIDER", "nebius").strip().lower() == "nebius":
        return "nebius" if os.getenv("NEBIUS_API_KEY") else "local"
    return "google" if is_google_live() else "local"


def backend() -> str:
    provider = _provider()
    if provider == "nebius":
        return f"nebius:{os.getenv('NEBIUS_EMBED_MODEL', _NEBIUS_MODEL)}"
    if provider == "google":
        prefix = "vertex" if os.getenv("USE_VERTEX") == "1" else "google-ai"
        return f"{prefix}:{_MODEL}"
    return "offline-hashed"


def _offline_embed(text: str) -> list[float]:
    vector = [0.0] * _DIM
    for token in _TOKEN.findall(text.lower()):
        index = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % _DIM
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


def _batches(texts: list[str]):
    batch, chars = [], 0
    for text in texts:
        if batch and (chars + len(text) > _CHAR_BUDGET or len(batch) >= _BATCH):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


def embed_texts(texts: list[str], on_progress=None) -> list[list[float]]:
    provider = _provider()
    if not texts:
        return []
    if provider == "local":
        result = [_offline_embed(text) for text in texts]
        if on_progress:
            on_progress(len(result), len(texts))
        return result
    if provider == "nebius":
        return _nebius_embed(texts, on_progress)
    if not is_google_live():
        raise ModelUnavailable("Google embeddings are selected but Google AI credentials are not configured.")
    from gemini_client import _client  # Reuse the configured Vertex/AI-Studio client.
    from google.genai import errors

    out: list[list[float]] = []
    with _client() as client:
        for batch in _batches(texts):
            for attempt in range(_MAX_RETRIES):
                try:
                    result = client.models.embed_content(model=_MODEL, contents=batch)
                    out.extend([list(embedding.values) for embedding in result.embeddings])
                    break
                except errors.APIError as error:
                    retryable = error.code in (429, 500, 503)
                    if retryable and attempt < _MAX_RETRIES - 1:
                        time.sleep(min(60, 8 * (attempt + 1)))  # Back off on quota/rate limits.
                        continue
                    raise
            if on_progress:
                on_progress(len(out), len(texts))
    return out


def _nebius_embed(texts: list[str], on_progress=None) -> list[list[float]]:
    model = os.getenv("NEBIUS_EMBED_MODEL", _NEBIUS_MODEL)
    out: list[list[float]] = []
    dimension = None
    for batch in _batches(texts):
        payload = nebius_post("embeddings", {
            "model": model, "input": batch, "encoding_format": "float",
        })
        if payload.get("model") != model:
            raise ModelUnavailable("Nebius embeddings returned a different model than requested.")
        rows = payload.get("data")
        if not isinstance(rows, list) or len(rows) != len(batch):
            raise ModelUnavailable("Nebius embeddings returned the wrong number of vectors.")
        indexed: dict[int, list[float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ModelUnavailable("Nebius embeddings returned an invalid vector entry.")
            index, vector = row.get("index"), row.get("embedding")
            if (type(index) is not int or index < 0 or index >= len(batch) or index in indexed):
                raise ModelUnavailable("Nebius embeddings returned invalid or duplicate input indexes.")
            if (not isinstance(vector, list) or not vector
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
                raise ModelUnavailable("Nebius embeddings returned an invalid or non-finite vector.")
            if dimension is not None and len(vector) != dimension:
                raise ModelUnavailable("Nebius embeddings returned inconsistent vector dimensions.")
            dimension = len(vector)
            indexed[index] = vector
        out.extend(indexed[index] for index in range(len(batch)))
        if on_progress:
            on_progress(len(out), len(texts))
    return out


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
