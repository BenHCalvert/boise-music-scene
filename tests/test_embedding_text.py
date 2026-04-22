from embeddings.index import build_text


def test_includes_name_genres_bio_and_derived_tags() -> None:
    artist = {
        "name": "Built to Spill",
        "genres_raw": ["indie rock", "rock"],
        "bio": "Boise indie rock band led by Doug Martsch.",
        "tags_genre": {"top": "indie rock"},
        "tags_mood": {"top": "mellow"},
        "tags_vibe": {"top": "earnest"},
    }
    text = build_text(artist)
    assert "Built to Spill" in text
    assert "Genres: indie rock, rock" in text
    assert "Doug Martsch" in text
    assert "Tags: indie rock, mellow, earnest" in text


def test_handles_missing_bio() -> None:
    artist = {"name": "Magic Sword", "genres_raw": ["synthwave"]}
    text = build_text(artist)
    assert "Magic Sword" in text
    assert "Genres: synthwave" in text


def test_handles_minimal_artist() -> None:
    text = build_text({"name": "Unknown Act"})
    assert text == "Unknown Act"
