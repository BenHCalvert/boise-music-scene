"""Resolve `spotify_id` for each artist in data/artists_seed.json via Spotify search.

Usage:
    python scripts/resolve_spotify_ids.py [--write]

Without --write, prints a dry-run report. With --write, updates artists_seed.json in place.
Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

SEED_PATH = Path(__file__).resolve().parents[1] / "data" / "artists_seed.json"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"


def get_access_token(client_id: str, client_secret: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def search_artist(name: str, token: str) -> dict | None:
    resp = requests.get(
        SEARCH_URL,
        params={"q": name, "type": "artist", "limit": 5},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("artists", {}).get("items", [])
    if not items:
        return None
    exact = next((a for a in items if a["name"].lower() == name.lower()), None)
    return exact or items[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Persist updates to artists_seed.json")
    args = parser.parse_args()

    load_dotenv()
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    csec = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not csec:
        print("error: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in .env", file=sys.stderr)
        return 1

    token = get_access_token(cid, csec)
    data = json.loads(SEED_PATH.read_text())

    updated = 0
    for artist in data["artists"]:
        if artist.get("spotify_id"):
            continue
        hit = search_artist(artist["name"], token)
        if hit is None:
            print(f"  miss: {artist['name']}")
            continue
        match_quality = "exact" if hit["name"].lower() == artist["name"].lower() else "fuzzy"
        print(f"  {match_quality}: {artist['name']} -> {hit['id']} ({hit['name']}, {hit.get('followers', {}).get('total', 0)} followers)")
        artist["spotify_id"] = hit["id"]
        updated += 1

    print(f"\nresolved {updated} artist(s).")
    if args.write and updated:
        SEED_PATH.write_text(json.dumps(data, indent=2) + "\n")
        print(f"wrote {SEED_PATH}")
    elif updated:
        print("(dry run — pass --write to persist)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
