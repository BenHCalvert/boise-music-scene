"""Spotify client — UI assets only (preview URL, image, external link).

Per Spotify Developer ToS, Spotify content may not be used to train or
ingest into ML models. This module therefore only fetches fields used for
display in the Gradio UI, not for classification or embedding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

TOKEN_URL = "https://accounts.spotify.com/api/token"
ARTIST_URL = "https://api.spotify.com/v1/artists/{id}"
TOP_TRACKS_URL = "https://api.spotify.com/v1/artists/{id}/top-tracks"


@dataclass
class SpotifyAssets:
    spotify_url: str | None
    image_url: str | None
    preview_url: str | None
    top_track_name: str | None


def get_access_token(client_id: str, client_secret: str) -> str:
    resp = requests.post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_assets(spotify_id: str, token: str, market: str = "US") -> SpotifyAssets:
    headers = {"Authorization": f"Bearer {token}"}
    artist_resp = requests.get(ARTIST_URL.format(id=spotify_id), headers=headers, timeout=10)
    artist_resp.raise_for_status()
    artist = artist_resp.json()
    image_url = artist["images"][0]["url"] if artist.get("images") else None
    spotify_url = artist.get("external_urls", {}).get("spotify")

    tracks_resp = requests.get(
        TOP_TRACKS_URL.format(id=spotify_id),
        headers=headers,
        params={"market": market},
        timeout=10,
    )
    tracks_resp.raise_for_status()
    tracks = tracks_resp.json().get("tracks", [])
    preview_url = None
    top_name = None
    for t in tracks:
        if t.get("preview_url"):
            preview_url = t["preview_url"]
            top_name = t.get("name")
            break
    if top_name is None and tracks:
        top_name = tracks[0].get("name")

    return SpotifyAssets(
        spotify_url=spotify_url,
        image_url=image_url,
        preview_url=preview_url,
        top_track_name=top_name,
    )


def load_token_from_env() -> str:
    load_dotenv()
    cid = os.environ["SPOTIFY_CLIENT_ID"]
    csec = os.environ["SPOTIFY_CLIENT_SECRET"]
    return get_access_token(cid, csec)
