"""End-to-end pipeline: ingest -> tag -> embed.

Usage:
    python scripts/run_all.py [--limit N] [--skip-spotify] [--skip-events]
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-spotify", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    args = parser.parse_args()

    ingest_cmd = [sys.executable, "scripts/run_ingest.py"]
    tag_cmd = [sys.executable, "-m", "classify.tagger"]
    embed_cmd = [sys.executable, "-m", "embeddings.index"]

    if args.limit:
        for cmd in (ingest_cmd, tag_cmd, embed_cmd):
            cmd += ["--limit", str(args.limit)]
    if args.skip_spotify:
        ingest_cmd.append("--skip-spotify")
    if args.skip_events:
        ingest_cmd.append("--skip-events")

    _run(ingest_cmd)
    _run(tag_cmd)
    _run(embed_cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
