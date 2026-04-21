"""Last.fm client — artist bio summary + top tags for the ML text pipeline.

Requires LASTFM_API_KEY. Last.fm bios are sourced from a Last.fm-maintained
wiki and are licensed CC-BY-SA; attribution is shown in the Gradio UI.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

BASE = "https://ws.audioscrobbler.com/2.0/"
_LINK_RE = re.compile(r'<a href="[^"]*">.*?Read more on Last\.fm</a>\.?', re.DOTALL)


@dataclass
class LastfmArtist:
    bio_summary: str
    bio_full: str
    tags: list[str]
    listeners: int | None
    url: str | None


def _key() -> str:
    load_dotenv()
    key = os.getenv("LASTFM_API_KEY")
    if not key:
        raise RuntimeError("LASTFM_API_KEY not set")
    return key


def _strip_read_more(text: str) -> str:
    cleaned = _LINK_RE.sub("", text or "").strip()
    return cleaned


def get_artist(name: str, mbid: str | None = None) -> LastfmArtist | None:
    params = {
        "method": "artist.getinfo",
        "artist": name,
        "api_key": _key(),
        "format": "json",
        "autocorrect": 1,
    }
    if mbid:
        params["mbid"] = mbid

    resp = requests.get(BASE, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "artist" not in data:
        return None
    a = data["artist"]
    bio = a.get("bio") or {}
    tags = [t["name"] for t in (a.get("tags") or {}).get("tag", [])]
    listeners_raw = (a.get("stats") or {}).get("listeners")
    listeners = int(listeners_raw) if listeners_raw and listeners_raw.isdigit() else None
    return LastfmArtist(
        bio_summary=_strip_read_more(bio.get("summary", "")),
        bio_full=_strip_read_more(bio.get("content", "")),
        tags=tags,
        listeners=listeners,
        url=a.get("url"),
    )
