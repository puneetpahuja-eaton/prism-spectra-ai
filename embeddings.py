"""Nebius embeddings; deterministic test doubles require FORCE_MOCK explicitly."""
from __future__ import annotations

import hashlib
import math
import os
import re

from nebius_client import ModelUnavailable, nebius_post

_DIM = 256
_NEBIUS_MODEL = "Qwen/Qwen3-Embedding-8B"
_TOKEN = re.compile(r"[a-z0-9]+")
_BATCH = int(os.getenv("EMBED_BATCH", "16"))
_CHAR_BUDGET = int(os.getenv("EMBED_CHAR_BUDGET", "24000"))


def _provider() -> str:
    provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if provider and provider != "nebius":
        raise ModelUnavailable("EMBEDDING_PROVIDER must be 'nebius'; local/Google inference is not supported.")
    if os.getenv("FORCE_MOCK") == "1":
        return "test-double"
    return "nebius"


def backend() -> str:
    provider = _provider()
    if provider == "nebius":
        return f"nebius:{os.getenv('NEBIUS_EMBED_MODEL', _NEBIUS_MODEL)}"
    return "test-double"


def _test_embed(text: str) -> list[float]:
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
    if provider == "test-double":
        result = [_test_embed(text) for text in texts]
        if on_progress:
            on_progress(len(result), len(texts))
        return result
    return _nebius_embed(texts, on_progress)


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
