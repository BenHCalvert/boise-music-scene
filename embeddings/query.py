"""Natural-language query over the FAISS artist index.

Powers both the search bar and the "sounds like" recommendations on the
artist detail page.

Usage (CLI):
    python -m embeddings.query "mellow local act playing this weekend"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

from embeddings.index import INDEX_PATH, META_PATH, MODEL_NAME


@dataclass
class Hit:
    row_id: int
    id: str
    name: str
    score: float
    top_genre: str | None
    top_mood: str | None
    top_vibe: str | None
    origin: str | None
    treefort_worthy: bool | None
    spotify_url: str | None
    lastfm_url: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _load():
    import faiss  # type: ignore
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "FAISS index or metadata missing. Run `python -m embeddings.index` first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    meta = pd.read_parquet(META_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model


def search(text: str, k: int = 10) -> list[Hit]:
    index, meta, model = _load()
    import numpy as np

    vec = model.encode([text], normalize_embeddings=True).astype("float32")
    k = min(k, index.ntotal)
    scores, idx = index.search(vec, k)
    hits: list[Hit] = []
    for score, row_id in zip(scores[0].tolist(), idx[0].tolist()):
        if row_id < 0:
            continue
        row = meta.iloc[row_id]
        hits.append(
            Hit(
                row_id=int(row_id),
                id=row.get("id"),
                name=row.get("name"),
                score=float(score),
                top_genre=row.get("top_genre"),
                top_mood=row.get("top_mood"),
                top_vibe=row.get("top_vibe"),
                origin=row.get("origin"),
                treefort_worthy=bool(row.get("treefort_worthy")) if row.get("treefort_worthy") is not None else None,
                spotify_url=row.get("spotify_url"),
                lastfm_url=row.get("lastfm_url"),
            )
        )
    return hits


def sounds_like(artist_id: str, k: int = 5) -> list[Hit]:
    """Top-k neighbours for an indexed artist (excludes self)."""
    index, meta, _ = _load()
    import numpy as np

    matches = meta.index[meta["id"] == artist_id].tolist()
    if not matches:
        return []
    row_id = int(matches[0])
    vec = index.reconstruct(row_id).reshape(1, -1).astype("float32")
    scores, idx = index.search(vec, min(k + 1, index.ntotal))
    hits: list[Hit] = []
    for score, rid in zip(scores[0].tolist(), idx[0].tolist()):
        if rid == row_id or rid < 0:
            continue
        row = meta.iloc[rid]
        hits.append(
            Hit(
                row_id=int(rid),
                id=row.get("id"),
                name=row.get("name"),
                score=float(score),
                top_genre=row.get("top_genre"),
                top_mood=row.get("top_mood"),
                top_vibe=row.get("top_vibe"),
                origin=row.get("origin"),
                treefort_worthy=bool(row.get("treefort_worthy")) if row.get("treefort_worthy") is not None else None,
                spotify_url=row.get("spotify_url"),
                lastfm_url=row.get("lastfm_url"),
            )
        )
        if len(hits) >= k:
            break
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural language query")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    hits = search(args.query, k=args.k)
    for h in hits:
        print(f"  {h.score:.3f}  {h.name}  [{h.top_genre} / {h.top_mood}]  ({h.origin})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
