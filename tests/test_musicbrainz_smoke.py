"""Live smoke test against MusicBrainz (no API key required).

Marked as a network test — run with `pytest tests/test_musicbrainz_smoke.py`.
"""

from __future__ import annotations

import pytest

from ingest import musicbrainz


@pytest.mark.network
def test_resolve_built_to_spill() -> None:
    artist = musicbrainz.resolve("Built to Spill")
    assert artist is not None
    assert artist.name.lower() == "built to spill"
    assert artist.country in {"US", None}
    assert len(artist.mbid) == 36
