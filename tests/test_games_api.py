"""The saved-games API: what the games list says about a game without opening it.

These run against a temporary games directory, so they never touch real games.

The endpoint functions are called directly rather than over HTTP. Starlette's
TestClient would need `httpx`, and a whole extra dependency to re-test FastAPI's own
routing is a poor trade - the part worth testing is what `_summary` decides, which is
right here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import games as games_module
from app.config import settings

BAR = settings.low_conf_threshold


def cell(row, col, letter, confidence=None, is_blank=False):
    return {"row": row, "col": col, "letter": letter,
            "isBlank": is_blank, "confidence": confidence, "source": "scan"}


def document(*turns):
    return {
        "schemaVersion": 1,
        "sessionId": "test-game",
        "turns": [{"index": i, "player": i % 2, "annotation": "play", "cells": list(cells),
                   "frameRef": None, "capturedAt": "2026-01-01T00:00:00Z", "note": ""}
                  for i, cells in enumerate(turns)],
        "currentTurn": len(turns) - 1,
        "players": ["A", "B"],
        "editLog": [],
    }


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A games directory of its own, so no test can see or touch a real game."""
    monkeypatch.setattr(settings, "games_dir", str(tmp_path))
    return tmp_path


def write(_store, game_id, doc):
    assert games_module.save_game(game_id, doc)["ok"]


def only_game(_store):
    games = games_module.list_games()["games"]
    assert len(games) == 1
    return games[0]


# -- what is still waiting on a person ---------------------------------------


def test_a_confidently_read_board_has_nothing_to_confirm(store):
    write(store, "g", document([cell(7, 7, "A", 0.99), cell(7, 8, "B", 0.90)]))
    assert only_game(store)["toConfirm"] == 0


def test_a_doubtful_reading_is_counted(store):
    write(store, "g", document([cell(7, 7, "A", 0.99), cell(7, 8, "B", BAR - 0.01)]))
    assert only_game(store)["toConfirm"] == 1


def test_the_threshold_itself_is_not_doubtful(store):
    """The bar is "below 0.75", not "at or below", and it is the same bar the reader
    flags against. A disagreement of one cell here would show up as a count the board
    page could never clear."""
    write(store, "g", document([cell(7, 7, "A", BAR)]))
    assert only_game(store)["toConfirm"] == 0


def test_a_tile_with_no_letter_is_counted_however_confident(store):
    """The reader saw a tile and could not name it. Confidence in *that* is not
    confidence in a letter, and the tile is invisible in a tile count."""
    write(store, "g", document([cell(7, 7, None, 0.99)]))
    assert only_game(store)["toConfirm"] == 1


def test_an_undesignated_blank_is_counted(store):
    """A blank's designation is not in the photograph at all, so it always waits."""
    write(store, "g", document([cell(7, 7, None, None, is_blank=True)]))
    assert only_game(store)["toConfirm"] == 1


def test_a_typed_or_confirmed_tile_never_counts(store):
    """Null confidence means not from OCR: typed in by hand, or checked by a person.
    That is the most trustworthy state a tile has."""
    doc = document([cell(7, 7, "A", None), cell(7, 8, "B", None)])
    doc["turns"][0]["cells"][1]["source"] = "confirmed"
    write(store, "g", doc)
    assert only_game(store)["toConfirm"] == 0


def test_only_the_last_turn_counts(store):
    """Every turn carries the whole board, so counting them all would count the same
    tile twenty times. The last turn is the board as it stands."""
    write(store, "g", document(
        [cell(7, 7, "A", BAR - 0.1)],
        [cell(7, 7, "A", None), cell(7, 8, "B", 0.99)],
    ))
    assert only_game(store)["toConfirm"] == 0


# -- the shape of a game ------------------------------------------------------


def test_shape_is_the_tile_count_per_turn(store):
    write(store, "g", document(
        [cell(7, 7, "A", 0.9)],
        [cell(7, 7, "A", 0.9), cell(7, 8, "B", 0.9)],
        [cell(7, 7, "A", 0.9), cell(7, 8, "B", 0.9), cell(7, 9, "C", 0.9)],
    ))
    game = only_game(store)
    assert game["shape"] == [1, 2, 3]
    assert game["turns"] == 3
    assert game["tiles"] == 3


def test_shape_stays_short_for_a_very_long_game(store):
    """A list of games should not carry a hundred numbers per row."""
    turns = [[cell(7, 7, "A", 0.9)] for _ in range(60)]
    write(store, "g", document(*turns))
    assert len(only_game(store)["shape"]) == 40


# -- a game that cannot be read ----------------------------------------------


def test_a_broken_game_is_listed_rather_than_taking_the_list_down(store, tmp_path):
    write(store, "good", document([cell(7, 7, "A", 0.9)]))
    broken = Path(tmp_path) / "bad"
    broken.mkdir()
    (broken / "game.json").write_text("{not json", encoding="utf-8")

    games = {g["id"]: g for g in games_module.list_games()["games"]}
    assert games["bad"]["broken"] is True
    assert games["good"]["broken"] is False
    # The page has to survive the missing keys rather than be handed guesses.
    assert "toConfirm" not in games["bad"] or games["bad"]["toConfirm"] == 0


def test_the_list_needs_a_token():
    """A doorbell, but it does have to ring."""
    with pytest.raises(HTTPException) as refused:
        games_module.require_auth("")
    assert refused.value.status_code == 401
    with pytest.raises(HTTPException):
        games_module.require_auth("Bearer not-a-real-token")
