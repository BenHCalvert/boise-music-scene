"""Build a FAISS index over artist embeddings.

Embedding input per artist:
    "{name}. Genres: {genres_raw}. {bio} Tags: {top_tags}"

Top tags are the zero-shot top-1 for genre/mood/vibe; including them lets
natural-language queries like "mellow folk" rank correctly even when the
bio is sparse.

Outputs:
    data/faiss_index/artists.index            FAISS IndexFlatIP (cosine)
    data/faiss_index/artists_meta.parquet     row_id -> artist metadata

Usage:
    python -m embeddings.index [--input data/artists_tagged.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "artists_tagged.json"
INDEX_DIR = ROOT / "data" / "faiss_index"
INDEX_PATH = INDEX_DIR / "artists.index"
META_PATH = INDEX_DIR / "artists_meta.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_text(artist: dict) -> str:
    name = artist.get("name", "")
    tags = artist.get("genres_raw") or artist.get("mb_tags") or []
    tag_str = ", ".join(tags[:8])
    bio = artist.get("bio") or artist.get("lastfm_bio_summary") or ""
    derived = []
    for axis in ("tags_genre", "tags_mood", "tags_vibe"):
        top = (artist.get(axis) or {}).get("top")
        if top:
            derived.append(top)
    derived_str = ", ".join(derived)
    parts = [name]
    if tag_str:
        parts.append(f"Genres: {tag_str}.")
    if bio:
        parts.append(bio)
    if derived_str:
        parts.append(f"Tags: {derived_str}.")
    return " ".join(parts).strip()


def build_index(input_path: Path = DEFAULT_INPUT) -> None:
    import faiss  # type: ignore
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    if not input_path.exists():
        print(f"error: {input_path} not found. Run classify.tagger first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text())
    artists = data["artists"]
    if not artists:
        print("error: no artists to index", file=sys.stderr)
        sys.exit(1)

    texts = [build_text(a) for a in artists]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine on L2-normalized vectors
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    meta_rows = []
    for row_id, a in enumerate(artists):
        meta_rows.append(
            {
                "row_id": row_id,
                "id": a.get("id"),
                "name": a.get("name"),
                "origin": a.get("origin"),
                "top_genre": (a.get("tags_genre") or {}).get("top"),
                "top_mood": (a.get("tags_mood") or {}).get("top"),
                "top_vibe": (a.get("tags_vibe") or {}).get("top"),
                "top_venue": (a.get("venue_fit") or {}).get("top"),
                "treefort_worthy": a.get("treefort_worthy"),
                "spotify_url": a.get("spotify_url"),
                "spotify_image_url": a.get("spotify_image_url"),
                "lastfm_url": a.get("lastfm_url"),
                "bio": a.get("bio") or a.get("lastfm_bio_summary") or "",
                "indexed_text": texts[row_id],
            }
        )
    pd.DataFrame(meta_rows).to_parquet(META_PATH, index=False)
    print(f"wrote {INDEX_PATH} ({len(artists)} vectors, dim={dim})")
    print(f"wrote {META_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    build_index(args.input)
    return 0


if __name__ == "__main__":
    sys.exit(main())
