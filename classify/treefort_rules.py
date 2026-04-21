"""Rule-based 'Treefort-worthy' classifier (v1).

v1 rule: artist is Treefort-worthy if
  - zero-shot top genre is in an indie/folk/experimental cluster, AND
  - origin is local or regional, AND
  - zero-shot top mood is not "aggressive" (proxy for mid-energy,
    since Spotify audio features are excluded from the ML path in v1).

v2 (stretch): train a binary classifier on labeled Treefort lineups.
"""

from __future__ import annotations

TREEFORT_GENRES = {"indie rock", "folk", "Americana", "experimental"}
LOCAL_ORIGINS = {"local", "regional"}
EXCLUDED_MOODS = {"aggressive"}


def is_treefort_worthy(artist: dict) -> dict:
    top_genre = (artist.get("tags_genre") or {}).get("top")
    top_mood = (artist.get("tags_mood") or {}).get("top")
    origin = artist.get("origin")

    reasons: list[str] = []
    if top_genre not in TREEFORT_GENRES:
        reasons.append(f"genre '{top_genre}' not in Treefort cluster")
    if origin not in LOCAL_ORIGINS:
        reasons.append(f"origin '{origin}' not local/regional")
    if top_mood in EXCLUDED_MOODS:
        reasons.append(f"mood '{top_mood}' excluded")

    verdict = len(reasons) == 0
    return {
        "treefort_worthy": verdict,
        "treefort_reasons": reasons,
    }
