"""Zero-shot classification over a fixed taxonomy of genre / mood / vibe / venue.

Uses facebook/bart-large-mnli via transformers' zero-shot-classification
pipeline. Results are cached to disk keyed on (text, label-set, hypothesis)
so rerunning the tagger only re-scores artists whose bio or tags changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / ".cache" / "zeroshot"
MODEL_NAME = "facebook/bart-large-mnli"

GENRE_LABELS = [
    "indie rock", "folk", "Americana", "hip-hop",
    "electronic", "punk", "jazz", "country", "experimental",
]
MOOD_LABELS = [
    "energetic", "mellow", "dark", "celebratory",
    "introspective", "danceable", "aggressive",
]
VIBE_LABELS = [
    "gritty", "polished", "lo-fi", "psychedelic",
    "nostalgic", "earnest", "playful", "cinematic",
]
VENUE_LABELS = [
    "intimate seated", "standing room", "dance floor",
    "outdoor festival", "all-ages",
]

_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline  # local import keeps module import cheap

        _pipeline = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device_map="auto",
        )
    return _pipeline


def _cache_key(text: str, labels: tuple[str, ...], hypothesis: str) -> str:
    payload = json.dumps(
        {"m": MODEL_NAME, "t": text, "l": list(labels), "h": hypothesis},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


@dataclass
class ZeroShotResult:
    labels: list[str]
    scores: list[float]

    @property
    def top(self) -> str:
        return self.labels[0]

    @property
    def top_score(self) -> float:
        return self.scores[0]

    def as_dict(self) -> dict:
        return {"labels": self.labels, "scores": self.scores, "top": self.top, "top_score": self.top_score}


def classify(
    text: str,
    labels: Iterable[str],
    hypothesis: str = "This artist's music is {}.",
    multi_label: bool = False,
) -> ZeroShotResult:
    labels_t = tuple(labels)
    key = _cache_key(text, labels_t, hypothesis)
    path = _cache_path(key)
    if path.exists():
        cached = json.loads(path.read_text())
        return ZeroShotResult(labels=cached["labels"], scores=cached["scores"])

    pipe = _get_pipeline()
    out = pipe(
        text,
        candidate_labels=list(labels_t),
        hypothesis_template=hypothesis,
        multi_label=multi_label,
    )
    result = ZeroShotResult(labels=out["labels"], scores=out["scores"])
    path.write_text(json.dumps(result.as_dict()))
    return result


def build_artist_text(artist: dict) -> str:
    name = artist.get("name", "")
    tags = artist.get("genres_raw") or artist.get("mb_tags") or []
    tag_str = ", ".join(tags[:8]) if tags else ""
    bio = artist.get("bio") or artist.get("lastfm_bio_summary") or ""
    parts = [name]
    if tag_str:
        parts.append(f"Genres: {tag_str}.")
    if bio:
        parts.append(bio)
    return " ".join(parts).strip()


def classify_artist(artist: dict) -> dict:
    text = build_artist_text(artist)
    return {
        "tags_genre": classify(text, GENRE_LABELS, "This artist plays {} music.").as_dict(),
        "tags_mood": classify(text, MOOD_LABELS, "This artist's music feels {}.").as_dict(),
        "tags_vibe": classify(text, VIBE_LABELS, "This artist's sound is {}.").as_dict(),
        "venue_fit": classify(text, VENUE_LABELS, "This artist is best suited for a {} venue.").as_dict(),
    }
