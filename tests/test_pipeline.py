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

from app.config import settings  # noqa: E402
from app.vision import classify, pipeline, registration  # noqa: E402
from scripts.make_fake_board import (  # noqa: E402
    SCHEMES,
    _place_words,
    photograph,
    render_flat,
)

TILTS = [0, 8, 15, 22, 28]
SEEDS = [1, 2, 3]
CASES = [(t, s) for t in TILTS for s in SEEDS]
# Every colour scheme, at a tilt that is not the easy one.
SCHEME_CASES = [(name, 15, 1) for name in sorted(SCHEMES)]


def make_board(tilt: float, seed: int, blanks: int = 0, scheme: str = "classic"):
    """Returns (photo, truth rows). Use `make_board_with_corners` for the corners."""
    photo, truth, _ = make_board_with_corners(tilt, seed, blanks, scheme)
    return photo, truth


def make_board_with_corners(
    tilt: float, seed: int, blanks: int = 0, scheme: str = "classic"
):
    rng = random.Random(seed)
    np.random.seed(seed)
    rows = _place_words(rng, blanks)
    photo, corners = photograph(render_flat(rows, rng, scheme), tilt, rng)
    return photo, ["".join(r) for r in rows], corners


def misread_letters(truth: list[str], result) -> list[tuple[int, int, str, str]]:
    return [
        (r, c, truth[r][c], result.board[r][c])
        for r in range(15)
        for c in range(15)
        if truth[r][c].isalpha() and truth[r][c] != result.board[r][c]
    ]


@pytest.mark.parametrize("tilt,seed", CASES)
def test_letters_are_read_correctly(tilt, seed):
    """At most one letter per board may be misread.

    Not zero, and the exception is specific rather than slack. These boards are
    drawn with a *font*, and a 'Q' rendered small in a bold sans is an 'O' plus a
    two-pixel tail: at 25 letters a board it is the one pair overlap cannot
    reliably separate, and it goes both ways with the camera angle.

    That is a property of the reference glyph's shape, not of this pipeline. Given
    reference glyphs taken from real tile artwork rather than a font - see
    `settings.template_tiles` - the same 'Q' reads correctly, which is why the
    allowance is one letter and not a percentage. Two would mean something else
    has broken.
    """
    photo, truth = make_board(tilt, seed)
    wrong = misread_letters(truth, pipeline.detect(photo))
    assert len(wrong) <= 1, f"misread letters at tilt={tilt} seed={seed}: {wrong}"


@pytest.mark.parametrize("tilt,seed", CASES)
def test_no_letter_is_misread_confidently(tilt, seed):
    """A misread must arrive flagged, whatever else happens.

    This is the property the board state is trusted on. A wrong letter reported
    confidently is a corrupted game record that nobody looks at twice; a wrong
    letter reported doubtfully is one tap in the operator UI.
    """
    photo, truth = make_board(tilt, seed)
    result = pipeline.detect(photo)

    flagged = {(c.row, c.col) for c in result.low_confidence}
    silent = [w for w in misread_letters(truth, result) if (w[0], w[1]) not in flagged]
    assert not silent, f"misread without a low-confidence flag: {silent}"


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


@pytest.mark.parametrize("scheme,tilt,seed", SCHEME_CASES)
def test_any_tile_and_board_colours_are_read(scheme, tilt, seed):
    """Nothing may depend on tiles being pale or letters being dark.

    `red_tiles` and `dark_tiles` invert the ink polarity, which used to make the
    board come back empty rather than misread - the pipeline looked for dark marks
    on light tiles and there were none to find. So what this test watches for is a
    scheme reading *worse than the others*, not a perfect read.

    Hence the same one-letter allowance as `test_letters_are_read_correctly`, and
    for the same reason: a small handful of letter pairs are separated only by the
    shape of one feature, which is a property of the reference glyph and has
    nothing to do with what colour the tiles are. Holding this test to zero made a
    typeface question look like a colour regression.
    """
    photo, truth = make_board(tilt, seed, scheme=scheme)
    result = pipeline.detect(photo)

    wrong = misread_letters(truth, result)
    assert len(wrong) <= 1, f"misread on {scheme}: {wrong}"

    flagged = {(c.row, c.col) for c in result.low_confidence}
    silent = [w for w in wrong if (w[0], w[1]) not in flagged]
    assert not silent, f"misread on {scheme} without a low-confidence flag: {silent}"


def test_reading_the_point_value_does_no_harm(monkeypatch):
    """The point-value path must stay working, and must not make things worse.

    It is off by default because at this resolution the digit is about eight pixels
    tall and reads as a coin flip - see `settings.use_point_value`. It stays in the
    code because it becomes a strong signal with more pixels on the board, and this
    exercises it so it cannot quietly rot in the meantime.

    Asserted as "no worse", not "better", which is what the measurements support.
    """
    photo, truth = make_board(15, 1)

    classify.reset()
    without = misread_letters(truth, pipeline.detect(photo))

    monkeypatch.setattr(settings, "use_point_value", True)
    classify.reset()
    try:
        result = pipeline.detect(photo)
        assert classify.values() is not None, "no digit references were built"
        assert len(misread_letters(truth, result)) <= len(without)
    finally:
        # The banks outlive the monkeypatch, so they have to be dropped explicitly.
        monkeypatch.undo()
        classify.reset()


@pytest.mark.parametrize("scheme,tilt,seed", SCHEME_CASES)
def test_no_phantom_tiles_on_any_colours(scheme, tilt, seed):
    """An empty square must stay empty whatever colour the board is painted."""
    photo, truth = make_board(tilt, seed, scheme=scheme)
    result = pipeline.detect(photo)

    phantom = [
        (r, c, result.board[r][c])
        for r in range(15)
        for c in range(15)
        if truth[r][c] == "." and result.board[r][c] != "."
    ]
    assert not phantom, f"tiles reported on empty squares of {scheme}: {phantom}"


@pytest.mark.parametrize("scheme,tilt,seed", SCHEME_CASES)
def test_registration_needs_no_colour_hints(scheme, tilt, seed):
    """The board is found by its grid, so the border's colour is irrelevant.

    `classic` has a red border, `red_tiles` a dark wooden one, `green_board` a
    near-black frame and `dark_tiles` a mid-grey one. A detector keyed to red
    found only the first.
    """
    photo, _ = make_board(tilt, seed, scheme=scheme)
    reg = registration.find_board(photo)
    board, info = registration.warp_board(photo, reg.corners)

    assert reg.grid_score >= settings.min_grid_score, (scheme, reg.considered)
    assert info["locked"], (scheme, info)
    assert grid_alignment(board) > 3.0, (scheme, info)


def grid_alignment(board: np.ndarray) -> float:
    """Edge energy on the assumed cell boundaries, over that at cell centres.

    Measured on the rectified output, so it asks the question that matters
    directly - do the printed grid lines land where this code will slice cells? -
    without caring which quad the detector proposed or what pitch it fitted.
    """
    size = board.shape[0]
    step = size / 15.0
    lines = np.clip(np.rint(np.arange(16) * step).astype(int), 0, size - 1)
    centres = np.rint((np.arange(15) + 0.5) * step).astype(int)
    worst = 1e9
    for proj in registration._edge_projections(board):
        worst = min(worst, float(proj[lines].mean() / max(proj[centres].mean(), 1e-6)))
    return worst


@pytest.mark.parametrize("tilt,seed", CASES)
def test_grid_locks_onto_the_printed_grid(tilt, seed):
    """The printed grid must land on the cell boundaries, or every cell shifts."""
    photo, _ = make_board(tilt, seed)
    reg = registration.find_board(photo)
    board, info = registration.warp_board(photo, reg.corners)

    assert info["locked"], info
    assert grid_alignment(board) > 3.0, (info, grid_alignment(board))


@pytest.mark.parametrize("tilt,seed", [(0, 1), (15, 2), (28, 3)])
def test_boundary_lines_alone_find_the_board(tilt, seed):
    """The line detector must find the board without help from any region mask.

    It exists for the photographs where no colour or texture blob outlines the board
    correctly, so it is worth knowing it stands on its own. Judged against the true
    playable corners the generator returns, not against another detector.

    Tolerant on distance - a whole cell - because these quads bound the printed
    border rather than the playable area, and closing that gap is the snapping and
    grid-lock stages' job, not this one's. What is being tested is that the four
    lines found are the board's, not a table's.
    """
    photo, _, truth = make_board_with_corners(tilt, seed)
    quads = registration._line_quads(photo, settings.line_quad_limit)
    assert quads, f"no quad from boundary lines at tilt={tilt} seed={seed}"

    step = np.linalg.norm(truth[1] - truth[0]) / 15.0
    best = min(np.abs(registration.order_corners(q) - truth).max() for q in quads)
    assert best <= step, f"nearest line quad is {best / step:.2f} cells out"


@pytest.mark.parametrize("dx,dy", [(1, 0), (-1, 0), (0, 1), (0, -1)])
def test_a_board_a_whole_cell_out_is_put_back(dx, dy):
    """Handed a board shifted by one cell, the pipeline must find the right one.

    The failure being guarded against is the only one that hides. A shifted lattice
    fits as well as the true one, reads a full complement of tiles and clears every
    geometric check, so what comes back is a complete and confident transcript of
    the wrong cells. The printed premium squares are the one board feature that does
    not repeat, and so the only thing that can tell the difference.

    Started from the true corners rather than the detector's, so that the shift is
    exactly one cell and the test is about recovering from it rather than about
    which quad some detector happened to propose.

    What is asserted is the transcript, not the direction taken to reach it. Without
    the correction these inputs agree with the printed layout at 0.08 and misread
    the whole board, so the property is real; but warping re-fits the cell comb, and
    a quad perturbed by about a cell can be pulled back from more than one
    direction, so which probe is adopted is not meaningful.
    """
    photo, truth, corners = make_board_with_corners(8, 1)
    result = pipeline.detect(photo, registration.shift_quad(corners, dx, dy))

    assert not misread_letters(truth, result), (
        f"shifted by ({dx},{dy}) and read {result.registration.method}"
    )


def test_registration_fails_loudly_on_a_blank_image():
    with pytest.raises(registration.RegistrationError):
        pipeline.detect(np.full((600, 600, 3), 128, np.uint8))


def test_decode_rejects_rubbish():
    with pytest.raises(pipeline.DecodeError):
        pipeline.decode(b"not an image")
    with pytest.raises(pipeline.DecodeError):
        pipeline.decode(b"")


def test_response_shape_is_self_consistent():
    photo, _ = make_board(15, 1)
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
