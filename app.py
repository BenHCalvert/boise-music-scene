"""Boise Music Scene Tagger — Gradio UI.

Three tabs:
    1. Search        — natural-language query → ranked artist cards
    2. Upcoming Shows — event feed, filter by genre/mood/origin
    3. Artist Detail — bio, tags, top-5 "sounds like" neighbours

Deployable to HF Spaces (SDK: gradio). Requires data/artists_tagged.json,
data/events.json, and data/faiss_index/ to be populated — run
`python scripts/run_all.py` locally first, then commit the artifacts.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd

from embeddings.query import search, sounds_like

ROOT = Path(__file__).resolve().parent
TAGGED = ROOT / "data" / "artists_tagged.json"
EVENTS = ROOT / "data" / "events.json"


@lru_cache(maxsize=1)
def _artists() -> dict[str, dict]:
    if not TAGGED.exists():
        return {}
    data = json.loads(TAGGED.read_text())
    return {a["id"]: a for a in data["artists"]}


@lru_cache(maxsize=1)
def _events_df() -> pd.DataFrame:
    if not EVENTS.exists():
        return pd.DataFrame()
    data = json.loads(EVENTS.read_text())
    rows = []
    artists = _artists()
    for e in data.get("events", []):
        a = artists.get(e.get("artist_id"), {})
        rows.append(
            {
                "when": e.get("datetime", "")[:10],
                "artist": a.get("name", e.get("artist_id")),
                "venue": e.get("venue_name") or "",
                "city": e.get("venue_city") or "",
                "region": e.get("venue_region") or "",
                "genre": (a.get("tags_genre") or {}).get("top"),
                "mood": (a.get("tags_mood") or {}).get("top"),
                "origin": a.get("origin"),
                "treefort": a.get("treefort_worthy"),
                "url": e.get("url") or "",
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("when")
    return df


# ---------- Search tab ----------

def _card(hit) -> str:
    tf = "🌲 Treefort-worthy" if hit.treefort_worthy else ""
    links = []
    if hit.spotify_url:
        links.append(f"[Spotify]({hit.spotify_url})")
    if hit.lastfm_url:
        links.append(f"[Last.fm]({hit.lastfm_url})")
    return (
        f"### {hit.name}  \n"
        f"**score** {hit.score:.3f} · **genre** {hit.top_genre or '—'} · "
        f"**mood** {hit.top_mood or '—'} · **vibe** {hit.top_vibe or '—'} · "
        f"**origin** {hit.origin or '—'}  \n"
        f"{tf}  \n"
        f"{' · '.join(links)}"
    )


def do_search(query: str, k: int) -> str:
    if not query.strip():
        return "_enter a query above_"
    try:
        hits = search(query, k=int(k))
    except FileNotFoundError as exc:
        return f"**Index not built.** {exc}"
    if not hits:
        return "_no results_"
    return "\n\n---\n\n".join(_card(h) for h in hits)


# ---------- Shows tab ----------

def filter_shows(genre: str, mood: str, origin: str) -> pd.DataFrame:
    df = _events_df()
    if df.empty:
        return df
    if genre and genre != "any":
        df = df[df["genre"] == genre]
    if mood and mood != "any":
        df = df[df["mood"] == mood]
    if origin and origin != "any":
        df = df[df["origin"] == origin]
    return df


def _options(col: str) -> list[str]:
    df = _events_df()
    if df.empty or col not in df:
        return ["any"]
    vals = sorted({v for v in df[col].dropna().unique() if v})
    return ["any", *vals]


# ---------- Artist detail tab ----------

def artist_names() -> list[str]:
    return sorted([a["name"] for a in _artists().values()])


def _artist_by_name(name: str) -> dict | None:
    for a in _artists().values():
        if a.get("name") == name:
            return a
    return None


def artist_detail(name: str) -> tuple[str, str]:
    if not name:
        return "_pick an artist_", ""
    a = _artist_by_name(name)
    if not a:
        return "_unknown artist_", ""

    bio = a.get("bio") or a.get("lastfm_bio_summary") or "_(no bio)_"
    tags = (a.get("genres_raw") or [])[:8]
    links = []
    if a.get("spotify_url"):
        links.append(f"[Spotify]({a['spotify_url']})")
    if a.get("lastfm_url"):
        links.append(f"[Last.fm]({a['lastfm_url']})")

    header = (
        f"# {a['name']}\n\n"
        f"**genre** {(a.get('tags_genre') or {}).get('top') or '—'} · "
        f"**mood** {(a.get('tags_mood') or {}).get('top') or '—'} · "
        f"**vibe** {(a.get('tags_vibe') or {}).get('top') or '—'} · "
        f"**venue fit** {(a.get('venue_fit') or {}).get('top') or '—'}  \n"
        f"**origin** {a.get('origin') or '—'} · "
        f"{'🌲 Treefort-worthy' if a.get('treefort_worthy') else '—'}  \n\n"
        f"{' · '.join(links)}\n\n"
        f"**Raw tags:** {', '.join(tags) if tags else '—'}\n\n"
        f"{bio}"
    )

    try:
        neighbours = sounds_like(a["id"], k=5)
    except FileNotFoundError:
        return header, "_index not built_"
    if not neighbours:
        return header, "_no neighbours_"

    rec_md = "## Sounds like\n\n" + "\n".join(
        f"- **{h.name}** ({h.top_genre or '—'} / {h.top_mood or '—'}) · score {h.score:.3f}"
        for h in neighbours
    )
    return header, rec_md


# ---------- App ----------

def build() -> gr.Blocks:
    with gr.Blocks(title="Boise Music Scene Tagger") as demo:
        gr.Markdown(
            "# Boise Music Scene Tagger\n"
            "Zero-shot tagged + embedded artists from the Boise scene. "
            "Data: MusicBrainz + Last.fm (CC-BY-SA bios, attribution preserved). "
            "Spotify links are UI only — Spotify content is not used for ML."
        )

        with gr.Tab("Search"):
            q = gr.Textbox(label="Natural-language query", placeholder="mellow local act with cinematic vibes")
            k = gr.Slider(1, 20, value=10, step=1, label="results")
            btn = gr.Button("Search", variant="primary")
            out = gr.Markdown()
            btn.click(do_search, inputs=[q, k], outputs=out)
            q.submit(do_search, inputs=[q, k], outputs=out)

        with gr.Tab("Upcoming shows"):
            with gr.Row():
                g = gr.Dropdown(choices=_options("genre"), value="any", label="genre")
                m = gr.Dropdown(choices=_options("mood"), value="any", label="mood")
                o = gr.Dropdown(choices=_options("origin"), value="any", label="origin")
            shows = gr.Dataframe(value=_events_df(), label="shows", wrap=True)
            for ctrl in (g, m, o):
                ctrl.change(filter_shows, inputs=[g, m, o], outputs=shows)

        with gr.Tab("Artist detail"):
            names = artist_names()
            initial_header, initial_recs = artist_detail(names[0]) if names else ("_no artists indexed_", "")
            picker = gr.Dropdown(choices=names, label="artist", value=names[0] if names else None)
            header = gr.Markdown(value=initial_header)
            recs = gr.Markdown(value=initial_recs)
            picker.change(artist_detail, inputs=picker, outputs=[header, recs])

    demo.queue(default_concurrency_limit=4)
    return demo


if __name__ == "__main__":
    build().launch()
