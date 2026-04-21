# Boise Music Scene Tagger

A Hugging Face pipeline that ingests Boise-area artist and event data, tags artists with genre/mood/vibe via zero-shot NLP, computes similarity via sentence embeddings, and exposes everything through a natural-language search UI.

## Status

Phase 1 of 6 — scaffold + seed dataset.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SPOTIFY_CLIENT_ID / SECRET and BANDSINTOWN_APP_ID
```

Later phases will add:

```bash
python scripts/resolve_spotify_ids.py   # Phase 1 helper — populate spotify_id fields
python scripts/run_all.py               # Phase 2+: ingest → classify → embed
python app.py                           # Phase 5: Gradio UI
```

## Data sources

- **Spotify Web API** — artist metadata, genres, audio features (avg over top tracks).
- **Bandsintown Public API** — upcoming shows by artist.

## Models

- `facebook/bart-large-mnli` — zero-shot classification over genre / mood / vibe / venue-fit labels.
- `sentence-transformers/all-MiniLM-L6-v2` — artist embeddings for FAISS similarity.

## Known limitations (v1)

- Requires a Spotify ID per seeded artist; purely-local acts without a Spotify profile are excluded in v1.
- "Treefort-worthy" is a hand-tuned rule, not a learned model.
- Zero-shot tags are noisy on short bios; confidence scores are surfaced in the UI.
- Bandsintown coverage is uneven for very small local acts.

## Repo layout

```
data/             seed / enriched / tagged artist JSON, FAISS index
ingest/           spotify.py, bandsintown.py
classify/         zeroshot.py, audio_rules.py, treefort_rules.py, tagger.py
embeddings/       index.py, query.py
scripts/          run_all.py, resolve_spotify_ids.py
app.py            Gradio UI (Phase 5)
```
