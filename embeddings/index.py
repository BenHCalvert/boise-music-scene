"""Build and persist a FAISS inner-product index of artist embeddings.

Reads data/artists_tagged.json, encodes each artist's text with
sentence-transformers/all-MiniLM-L6-v2 (after L2 normalisation, inner
product == cosine similarity), and writes:
  data/artists.index    - FAISS IndexFlatIP
  data/artist_ids.json  - ordered list of artist ids matching index rows

Usage:
    python -m embeddings.index [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from classify.zeroshot import build_artist_text

ROOT = Path(__file__).resolve().parents[1]
TAGGED = ROOT / "data" / "artists_tagged.json"
INDEX_PATH = ROOT / "data" / "artists.index"
IDS_PATH = ROOT / "data" / "artist_ids.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> np.ndarray:
    vecs = _get_model().encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return vecs.astype("float32")


def build_index(artists: list[dict]) -> tuple[faiss.IndexFlatIP, list[str], np.ndarray]:
    texts = [build_artist_text(a) for a in artists]
    ids = [a["id"] for a in artists]
    vecs = encode(texts)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, ids, vecs


def save(
    index: faiss.IndexFlatIP,
    ids: list[str],
    index_path: Path = INDEX_PATH,
    ids_path: Path = IDS_PATH,
) -> None:
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids, indent=2) + "\n")


def load(
    index_path: Path = INDEX_PATH,
    ids_path: Path = IDS_PATH,
) -> tuple[faiss.IndexFlatIP, list[str]]:
    index = faiss.read_index(str(index_path))
    ids = json.loads(ids_path.read_text())
    return index, ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not TAGGED.exists():
        print(
            f"error: {TAGGED} not found. Run classify/tagger.py first.",
            file=sys.stderr,
        )
        return 1

    data = json.loads(TAGGED.read_text())
    artists = data["artists"]
    if args.limit:
        artists = artists[: args.limit]

    index, ids, _ = build_index(artists)
    save(index, ids)
    print(f"wrote {INDEX_PATH} and {IDS_PATH} ({len(ids)} artists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
