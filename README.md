# Boise Music Scene Tagger

A Hugging Face pipeline that ingests Boise-area artist and event data, tags artists with genre/mood/vibe via zero-shot NLP, computes similarity via sentence embeddings, and exposes everything through a natural-language search UI.

## Status

Phase 5 of 6 — Gradio UI built, ready to deploy once data artifacts are generated.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SPOTIFY_CLIENT_ID / SECRET and BANDSINTOWN_APP_ID
```

Later phases will add:

```bash
python scripts/resolve_spotify_ids.py --write   # populate spotify_id via Spotify search
python scripts/run_all.py                       # ingest → tag → index (all phases)
python app.py                                   # launch the Gradio UI locally
```

## Deploying to HF Spaces

1. Run `python scripts/run_all.py` locally to generate `data/artists_tagged.json`, `data/events.json`, and `data/faiss_index/`.
2. Commit the generated data artifacts (the `.gitignore` excludes them by default — remove or comment those lines before committing).
3. Create a new Gradio Space on Hugging Face, point it at this repo, and copy `README_SPACE.md` into the Space's `README.md` (or use Space settings to set the frontmatter).
4. Push; Spaces will `pip install -r requirements.txt` and run `app.py`.

## Data sources

The project separates **ML-ingested sources** (fed into classification + embeddings) from **UI-only sources** (rendered in the app but never ingested into models).

- **MusicBrainz** (CC0, ML-safe) — artist area, country, genre tags.
- **Last.fm API** (CC-BY-SA bios, attribution shown in UI) — artist bio summary + top tags. ML side.
- **Spotify Web API** — UI only. Fetches top-track preview, artist image, and the external Spotify link for the artist card. **Not used for classification or embeddings**, per Spotify's developer ToS prohibiting use of Spotify content for ML training.
- **Bandsintown Public API** — upcoming shows by artist name.

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
ingest/           spotify.py (UI), musicbrainz.py, lastfm.py, bandsintown.py
classify/         zeroshot.py, treefort_rules.py, tagger.py
embeddings/       index.py, query.py
scripts/          run_ingest.py, resolve_spotify_ids.py, run_all.py
app.py            Gradio UI (Phase 5)
```
