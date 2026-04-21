"""Bandsintown client — upcoming events for an artist."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from urllib.parse import quote

import requests
from dotenv import load_dotenv

BASE = "https://rest.bandsintown.com/artists"


@dataclass
class Event:
    id: str
    datetime: str
    venue_name: str | None
    venue_city: str | None
    venue_region: str | None
    venue_country: str | None
    lineup: list[str]
    url: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _app_id() -> str:
    load_dotenv()
    return os.getenv("BANDSINTOWN_APP_ID", "boise-music-scene")


def get_events(artist_name: str) -> list[Event]:
    url = f"{BASE}/{quote(artist_name, safe='')}/events"
    resp = requests.get(
        url,
        params={"app_id": _app_id()},
        headers={"Accept": "application/json"},
        timeout=15,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        return []
    events: list[Event] = []
    for e in payload:
        venue = e.get("venue") or {}
        events.append(
            Event(
                id=str(e.get("id", "")),
                datetime=e.get("datetime", ""),
                venue_name=venue.get("name"),
                venue_city=venue.get("city"),
                venue_region=venue.get("region"),
                venue_country=venue.get("country"),
                lineup=e.get("lineup") or [],
                url=e.get("url"),
            )
        )
    return events
