"""End-to-end pipeline: ingest -> tag -> index.

Usage:
    python scripts/run_all.py [--limit N] [--skip-spotify] [--skip-events]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-spotify", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    args = parser.parse_args()

    limit_args = ["--limit", str(args.limit)] if args.limit else []

    ingest_cmd = [sys.executable, "scripts/run_ingest.py", *limit_args]
    if args.skip_spotify:
        ingest_cmd.append("--skip-spotify")
    if args.skip_events:
        ingest_cmd.append("--skip-events")
    run(ingest_cmd)

    run([sys.executable, "-m", "classify.tagger", *limit_args])
    run([sys.executable, "-m", "embeddings.index"])
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
