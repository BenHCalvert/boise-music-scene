from classify.treefort_rules import is_treefort_worthy


def _artist(genre: str, mood: str, origin: str) -> dict:
    return {
        "tags_genre": {"top": genre},
        "tags_mood": {"top": mood},
        "origin": origin,
    }


def test_local_indie_rock_mellow_is_worthy() -> None:
    v = is_treefort_worthy(_artist("indie rock", "mellow", "local"))
    assert v["treefort_worthy"] is True
    assert v["treefort_reasons"] == []


def test_touring_hiphop_is_not_worthy() -> None:
    v = is_treefort_worthy(_artist("hip-hop", "energetic", "touring"))
    assert v["treefort_worthy"] is False
    assert len(v["treefort_reasons"]) == 2


def test_aggressive_mood_excluded() -> None:
    v = is_treefort_worthy(_artist("punk", "aggressive", "local"))
    assert v["treefort_worthy"] is False
    assert any("mood" in r for r in v["treefort_reasons"])


def test_regional_folk_is_worthy() -> None:
    v = is_treefort_worthy(_artist("folk", "introspective", "regional"))
    assert v["treefort_worthy"] is True
