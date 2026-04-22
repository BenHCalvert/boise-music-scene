"""Unit tests for embeddings.index and embeddings.query.

The real sentence-transformer model is never loaded; encode() is patched
with a deterministic stub so tests run without GPU/network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest

DIM = 8

ARTISTS = [
    {"id": "artist-a", "name": "Artist A", "bio": "Indie rock from Boise.", "origin": "local"},
    {"id": "artist-b", "name": "Artist B", "bio": "Folk music with mellow vibes.", "origin": "regional"},
    {"id": "artist-c", "name": "Artist C", "bio": "Electronic beats and experimental sounds.", "origin": "local"},
]


def _stub_encode(texts: list[str]) -> np.ndarray:
    """One-hot stub: text i gets a 1 in position (i % DIM)."""
    vecs = np.zeros((len(texts), DIM), dtype="float32")
    for i in range(len(texts)):
        vecs[i, i % DIM] = 1.0
    return vecs


@pytest.fixture()
def index_paths(tmp_path: Path):
    return tmp_path / "artists.index", tmp_path / "artist_ids.json"


def test_build_index_ntotal(index_paths):
    from embeddings.index import build_index

    with patch("embeddings.index.encode", side_effect=_stub_encode):
        index, ids, vecs = build_index(ARTISTS)

    assert index.ntotal == len(ARTISTS)
    assert ids == ["artist-a", "artist-b", "artist-c"]
    assert vecs.shape == (len(ARTISTS), DIM)


def test_save_load_roundtrip(index_paths):
    from embeddings.index import build_index, load, save

    idx_path, ids_path = index_paths
    with patch("embeddings.index.encode", side_effect=_stub_encode):
        index, ids, _ = build_index(ARTISTS)
        save(index, ids, idx_path, ids_path)
        loaded_index, loaded_ids = load(idx_path, ids_path)

    assert loaded_ids == ["artist-a", "artist-b", "artist-c"]
    assert loaded_index.ntotal == 3


def test_search_returns_k_results(index_paths):
    from embeddings.index import build_index, save
    from embeddings.query import search

    idx_path, ids_path = index_paths
    with patch("embeddings.index.encode", side_effect=_stub_encode):
        index, ids, _ = build_index(ARTISTS)
        save(index, ids, idx_path, ids_path)

    artists_by_id = {a["id"]: a for a in ARTISTS}
    with patch("embeddings.query.encode", side_effect=_stub_encode):
        results = search("indie rock boise", k=3, index=index, ids=ids, artists_by_id=artists_by_id)

    assert len(results) == 3
    assert all("id" in r and "score" in r and "name" in r for r in results)


def test_search_scores_descending(index_paths):
    from embeddings.index import build_index, save
    from embeddings.query import search

    idx_path, ids_path = index_paths
    with patch("embeddings.index.encode", side_effect=_stub_encode):
        index, ids, _ = build_index(ARTISTS)
        save(index, ids, idx_path, ids_path)

    artists_by_id = {a["id"]: a for a in ARTISTS}
    with patch("embeddings.query.encode", side_effect=_stub_encode):
        results = search("query text", k=3, index=index, ids=ids, artists_by_id=artists_by_id)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_k_smaller_than_corpus(index_paths):
    from embeddings.index import build_index, save
    from embeddings.query import search

    idx_path, ids_path = index_paths
    with patch("embeddings.index.encode", side_effect=_stub_encode):
        index, ids, _ = build_index(ARTISTS)
        save(index, ids, idx_path, ids_path)

    artists_by_id = {a["id"]: a for a in ARTISTS}
    with patch("embeddings.query.encode", side_effect=_stub_encode):
        results = search("query", k=2, index=index, ids=ids, artists_by_id=artists_by_id)

    assert len(results) == 2
