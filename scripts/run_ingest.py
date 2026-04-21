"""Enrich artists_seed.json with MusicBrainz + Last.fm + Spotify UI assets,
and fetch Bandsintown events.

Outputs:
    data/artists_enriched.json
    data/events.json

Usage:
    python scripts/run_ingest.py [--limit N] [--skip-spotify] [--skip-events]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ingest import bandsintown, lastfm, musicbrainz, spotify

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "artists_seed.json"
ENRICHED = ROOT / "data" / "artists_enriched.json"
EVENTS = ROOT / "data" / "events.json"


def enrich_one(artist: dict, spotify_token: str | None) -> dict:
    name = artist["name"]
    out = dict(artist)

    mb = musicbrainz.resolve(name)
    if mb:
        out["mb_mbid"] = mb.mbid
        out["mb_country"] = mb.country
        out["mb_area"] = mb.area
        out["mb_tags"] = mb.tags
        out["mb_begin_date"] = mb.begin_date
    else:
        out["mb_mbid"] = None
        out["mb_tags"] = []

    try:
        lfm = lastfm.get_artist(name, mbid=out.get("mb_mbid"))
    except Exception as exc:  # noqa: BLE001
        print(f"  lastfm error for {name}: {exc}", file=sys.stderr)
        lfm = None
    if lfm:
        out["lastfm_bio_summary"] = lfm.bio_summary
        out["lastfm_bio_full"] = lfm.bio_full
        out["lastfm_tags"] = lfm.tags
        out["lastfm_listeners"] = lfm.listeners
        out["lastfm_url"] = lfm.url
        if not out.get("bio"):
            out["bio"] = lfm.bio_summary
    else:
        out.setdefault("lastfm_tags", [])

    if spotify_token and artist.get("spotify_id"):
        try:
            assets = spotify.fetch_assets(artist["spotify_id"], spotify_token)
            out["spotify_url"] = assets.spotify_url
            out["spotify_image_url"] = assets.image_url
            out["spotify_preview_url"] = assets.preview_url
            out["spotify_top_track_name"] = assets.top_track_name
        except Exception as exc:  # noqa: BLE001
            print(f"  spotify error for {name}: {exc}", file=sys.stderr)

    tags = sorted(set((out.get("mb_tags") or []) + (out.get("lastfm_tags") or [])))
    out["genres_raw"] = tags
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-spotify", action="store_true")
    parser.add_argument("--skip-events", action="store_true")
    args = parser.parse_args()

    seed = json.loads(SEED.read_text())
    artists = seed["artists"]
    if args.limit:
        artists = artists[: args.limit]

    spotify_token: str | None = None
    if not args.skip_spotify:
        try:
            spotify_token = spotify.load_token_from_env()
        except KeyError:
            print("note: SPOTIFY_CLIENT_ID / SECRET missing — skipping Spotify assets")

    enriched: list[dict] = []
    for i, artist in enumerate(artists, 1):
        print(f"[{i}/{len(artists)}] {artist['name']}")
        enriched.append(enrich_one(artist, spotify_token))

    ENRICHED.write_text(json.dumps({"artists": enriched}, indent=2) + "\n")
    print(f"wrote {ENRICHED} ({len(enriched)} artists)")

    if args.skip_events:
        return 0

    all_events: list[dict] = []
    for a in enriched:
        try:
            events = bandsintown.get_events(a["name"])
        except Exception as exc:  # noqa: BLE001
            print(f"  bandsintown error for {a['name']}: {exc}", file=sys.stderr)
            continue
        for e in events:
            d = e.to_dict()
            d["artist_id"] = a["id"]
            all_events.append(d)
    EVENTS.write_text(json.dumps({"events": all_events}, indent=2) + "\n")
    print(f"wrote {EVENTS} ({len(all_events)} events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
