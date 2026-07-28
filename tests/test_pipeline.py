"""End-to-end tests against synthetic boards.

No image fixtures are committed: `scripts/make_fake_board.py` renders them, so
these run anywhere and the test set can be widened by adding tilts and seeds.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.vision import pipeline, registration  # noqa: E402
from scripts.make_fake_board import _place_words, photograph, render_flat  # noqa: E402

TILTS = [0, 8, 15, 22, 28]
SEEDS = [1, 2, 3]
CASES = [(t, s) for t in TILTS for s in SEEDS]


def make_board(tilt: float, seed: int, blanks: int = 0):
    rng = random.Random(seed)
    np.random.seed(seed)
    rows = _place_words(rng, blanks)
    photo = photograph(render_flat(rows, rng), tilt, rng)
    return photo, ["".join(r) for r in rows]


@pytest.mark.parametrize("tilt,seed", CASES)
def test_every_letter_is_read_correctly(tilt, seed):
    photo, truth = make_board(tilt, seed)
    result = pipeline.detect(photo)

    wrong = [
        (r, c, truth[r][c], result.board[r][c])
        for r in range(15)
        for c in range(15)
        if truth[r][c].isalpha() and truth[r][c] != result.board[r][c]
    ]
    assert not wrong, f"misread letters at tilt={tilt} seed={seed}: {wrong}"


@pytest.mark.parametrize("tilt,seed", CASES)
def test_no_phantom_tiles(tilt, seed):
    photo, truth = make_board(tilt, seed)
    result = pipeline.detect(photo)

    phantom = [
        (r, c, result.board[r][c])
        for r in range(15)
        for c in range(15)
        if truth[r][c] == "." and result.board[r][c] != "."
    ]
    assert not phantom, f"tiles reported on empty squares: {phantom}"


@pytest.mark.parametrize("tilt,seed", [(0, 1), (15, 2), (25, 3)])
def test_blank_tiles_are_found(tilt, seed):
    photo, truth = make_board(tilt, seed, blanks=2)
    result = pipeline.detect(photo)

    expected = {(r, c) for r in range(15) for c in range(15) if truth[r][c] == "?"}
    found = {(r, c) for r in range(15) for c in range(15) if result.board[r][c] == "?"}
    assert found == expected, f"blanks expected {sorted(expected)}, got {sorted(found)}"


@pytest.mark.parametrize("tilt,seed", CASES)
def test_grid_locks_onto_the_printed_grid(tilt, seed):
    """The pitch must land near the true playable width, or every cell shifts."""
    photo, _ = make_board(tilt, seed)
    reg = registration.find_board(photo)
    _, info = registration.warp_board(photo, reg.corners)

    assert info["locked"], info
    # The synthetic board is 15 * 48 px of grid inside a 34 px border, so the
    # playable area is 720/788 of the detected quad.
    expected = (720 / 15) * (720 / 788)
    assert info["grid_period"] == pytest.approx(expected, rel=0.02), info


def test_registration_fails_loudly_on_a_blank_image():
    with pytest.raises(registration.RegistrationError):
        pipeline.detect(np.full((600, 600, 3), 128, np.uint8))


def test_decode_rejects_rubbish():
    with pytest.raises(pipeline.DecodeError):
        pipeline.decode(b"not an image")
    with pytest.raises(pipeline.DecodeError):
        pipeline.decode(b"")


def test_response_shape_is_self_consistent():
    photo, truth = make_board(15, 1)
    result = pipeline.detect(photo)

    assert len(result.board) == 15
    assert all(len(row) == 15 for row in result.board)
    assert result.tile_count == len(result.tiles)
    assert result.tile_count == sum(ch != "." for row in result.board for ch in row)
    for cell in result.tiles:
        assert 0 <= cell.row < 15 and 0 <= cell.col < 15
        assert 0.0 <= cell.confidence <= 1.0
    # every low-confidence cell must also appear in tiles
    keys = {(t.row, t.col) for t in result.tiles}
    assert all((c.row, c.col) in keys for c in result.low_confidence)


def test_generator_script_runs(tmp_path):
    out = subprocess.run(
        [sys.executable, "scripts/make_fake_board.py", str(tmp_path), "--tilt", "12"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "board.jpg").exists()
    truth = json.loads((tmp_path / "board.json").read_text())
    assert len(truth["board"]) == 15
    assert cv2.imread(str(tmp_path / "board.jpg")) is not None
