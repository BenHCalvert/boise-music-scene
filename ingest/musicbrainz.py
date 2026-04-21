"""MusicBrainz client — CC0-licensed artist metadata (area, country, tags).

MusicBrainz data is ML-safe (public domain / CC0). Rate limit: 1 req/sec;
a descriptive User-Agent is required (configure via MUSICBRAINZ_USER_AGENT).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

BASE = "https://musicbrainz.org/ws/2"
_last_call: float = 0.0


@dataclass
class MBArtist:
    mbid: str
    name: str
    country: str | None
    area: str | None
    tags: list[str]
    begin_date: str | None


def _throttle() -> None:
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_call = time.monotonic()


def _headers() -> dict[str, str]:
    load_dotenv()
    ua = os.getenv("MUSICBRAINZ_USER_AGENT", "boise-music-scene/0.1")
    return {"User-Agent": ua, "Accept": "application/json"}


def search_artist(name: str) -> dict | None:
    _throttle()
    resp = requests.get(
        f"{BASE}/artist",
        params={"query": f'artist:"{name}"', "fmt": "json", "limit": 5},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    items = resp.json().get("artists", [])
    if not items:
        return None
    exact = next((a for a in items if a["name"].lower() == name.lower()), None)
    return exact or items[0]


def get_artist(mbid: str) -> MBArtist:
    _throttle()
    resp = requests.get(
        f"{BASE}/artist/{mbid}",
        params={"fmt": "json", "inc": "tags"},
        headers=_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    tags = [t["name"] for t in sorted(data.get("tags", []), key=lambda t: -t.get("count", 0))]
    return MBArtist(
        mbid=data["id"],
        name=data["name"],
        country=data.get("country"),
        area=(data.get("area") or {}).get("name"),
        tags=tags,
        begin_date=(data.get("life-span") or {}).get("begin"),
    )


def resolve(name: str) -> MBArtist | None:
    hit = search_artist(name)
    if hit is None:
        return None
    return get_artist(hit["id"])
