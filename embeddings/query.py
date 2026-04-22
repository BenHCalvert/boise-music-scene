"""Nearest-neighbor artist search against the FAISS index.

Usage:
    python -m embeddings.query "energetic indie rock with a gritty sound" [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from embeddings.index import INDEX_PATH, IDS_PATH, TAGGED, encode, load

ROOT = Path(__file__).resolve().parents[1]


def search(
    query: str,
    k: int = 5,
    index: faiss.IndexFlatIP | None = None,
    ids: list[str] | None = None,
    artists_by_id: dict[str, dict] | None = None,
) -> list[dict]:
    if index is None or ids is None:
        index, ids = load()
    if artists_by_id is None:
        data = json.loads(TAGGED.read_text())
        artists_by_id = {a["id"]: a for a in data["artists"]}

    vec = encode([query])
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        artist_id = ids[idx]
        artist = artists_by_id.get(artist_id, {})
        results.append(
            {
                "id": artist_id,
                "name": artist.get("name", artist_id),
                "score": float(score),
                "top_genre": (artist.get("tags_genre") or {}).get("top"),
                "top_mood": (artist.get("tags_mood") or {}).get("top"),
                "treefort_worthy": artist.get("treefort_worthy"),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="natural-language search query")
    parser.add_argument("--k", type=int, default=5, help="number of results (default 5)")
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        print(
            f"error: {INDEX_PATH} not found. Run python -m embeddings.index first.",
            file=sys.stderr,
        )
        return 1

    results = search(args.query, k=args.k)
    for i, r in enumerate(results, 1):
        treefort = "yes" if r["treefort_worthy"] else "no"
        print(
            f"{i}. {r['name']}  [{r['top_genre']} / {r['top_mood']}]"
            f"  treefort={treefort}  score={r['score']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
