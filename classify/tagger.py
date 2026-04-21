"""Reconciler: read artists_enriched.json, zero-shot classify, apply Treefort
rule, write artists_tagged.json.

Usage:
    python -m classify.tagger [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from classify import treefort_rules, zeroshot

ROOT = Path(__file__).resolve().parents[1]
ENRICHED = ROOT / "data" / "artists_enriched.json"
TAGGED = ROOT / "data" / "artists_tagged.json"


def tag_one(artist: dict) -> dict:
    out = dict(artist)
    out.update(zeroshot.classify_artist(artist))
    out.update(treefort_rules.is_treefort_worthy(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not ENRICHED.exists():
        print(f"error: {ENRICHED} not found. Run scripts/run_ingest.py first.", file=sys.stderr)
        return 1

    data = json.loads(ENRICHED.read_text())
    artists = data["artists"]
    if args.limit:
        artists = artists[: args.limit]

    tagged: list[dict] = []
    for i, a in enumerate(artists, 1):
        print(f"[{i}/{len(artists)}] {a['name']}")
        tagged.append(tag_one(a))

    TAGGED.write_text(json.dumps({"artists": tagged}, indent=2) + "\n")
    print(f"wrote {TAGGED} ({len(tagged)} artists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
