#!/usr/bin/env python3
"""Render a synthetic Scrabble board photo plus ground truth.

Useful before you have real photos, and useful afterwards as a regression test
you can run in CI without shipping image fixtures.

    python scripts/make_fake_board.py out/ --tilt 18 --seed 3

Writes out/board.jpg and out/board.json (the expected 15-row board state).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Run directly as a script as well as imported by the tests. sys.path[0] is this
# file's own directory when run directly, so the repo root has to be added.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.classify import resolve_font  # noqa: E402

N = 15
CELL = 48
BORDER = 34

@dataclass(frozen=True)
class Scheme:
    """One board's colour scheme, all RGB.

    Boards in the wild vary in every one of these independently, and the vision
    code is not allowed to know any of them in advance, so the test set has to
    vary them too. `red_tiles` in particular exists because a board whose tiles
    are darker than its face inverts the ink polarity, which was the one case the
    pipeline could not read at all.
    """
    tile: tuple[int, int, int]
    ink: tuple[int, int, int]
    tile_edge: tuple[int, int, int]
    board: tuple[int, int, int]
    border: tuple[int, int, int]
    gridline: tuple[int, int, int]
    premium: dict[str, tuple[int, int, int]]
    premium_ink: tuple[int, int, int]


_PALE_PREMIUM = {
    "TW": (226, 88, 74), "DW": (240, 170, 170),
    "TL": (58, 130, 190), "DL": (168, 208, 230),
}

SCHEMES = {
    # Cream tiles, dark letters, pale board, red border. The common tournament
    # board, and the only one the pipeline used to handle.
    "classic": Scheme(
        tile=(232, 218, 186), ink=(25, 22, 20), tile_edge=(170, 155, 125),
        board=(232, 232, 224), border=(200, 40, 34), gridline=(170, 170, 165),
        premium=_PALE_PREMIUM, premium_ink=(250, 250, 250),
    ),
    # Dark red tiles with white letters on a cream board, edged in dark wood.
    "red_tiles": Scheme(
        tile=(158, 38, 46), ink=(248, 244, 240), tile_edge=(110, 26, 32),
        board=(238, 232, 210), border=(74, 46, 32), gridline=(150, 146, 132),
        premium={"TW": (214, 96, 88), "DW": (226, 158, 150),
                 "TL": (120, 130, 186), "DL": (170, 178, 214)},
        premium_ink=(250, 250, 250),
    ),
    # Cream tiles on a green board with a near-black frame.
    "green_board": Scheme(
        tile=(236, 214, 158), ink=(28, 24, 20), tile_edge=(176, 156, 108),
        board=(86, 148, 96), border=(38, 40, 38), gridline=(52, 96, 62),
        premium={"TW": (222, 92, 96), "DW": (238, 168, 176),
                 "TL": (74, 140, 196), "DL": (170, 210, 232)},
        premium_ink=(250, 250, 250),
    ),
    # Near-black tiles with white letters on a light grey board. The hardest
    # polarity case, and a check that nothing depends on tiles being warm.
    "dark_tiles": Scheme(
        tile=(38, 38, 42), ink=(240, 240, 236), tile_edge=(20, 20, 24),
        board=(206, 206, 202), border=(120, 120, 118), gridline=(150, 150, 148),
        premium=_PALE_PREMIUM, premium_ink=(40, 40, 40),
    ),
}

# Standard premium-square layout, quarter-symmetric.
_TW = [(0, 0), (0, 7), (7, 0)]
_DW = [(1, 1), (2, 2), (3, 3), (4, 4)]
_TL = [(1, 5), (5, 1), (5, 5)]
_DL = [(0, 3), (3, 0), (2, 6), (6, 2), (6, 6), (7, 3), (3, 7)]

WORDS = [
    ("RETAINS", 7, 4, "H"), ("QUIZ", 3, 7, "V"), ("JOKE", 10, 2, "H"),
    ("VEXED", 5, 9, "V"), ("BOX", 12, 6, "H"), ("FLY", 2, 11, "V"),
]

# The real English point values. A tile prints the value of the letter on it, so
# printing a random one - as this generator used to - produces a board that cannot
# be used to test whether the printed value helps identify the letter.
POINTS = {
    **dict.fromkeys("AEILNORSTU", 1), **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3), **dict.fromkeys("FHVWY", 4),
    "K": 5, **dict.fromkeys("JX", 8), **dict.fromkeys("QZ", 10),
}


def _premium_map() -> dict[tuple[int, int], str]:
    out: dict[tuple[int, int], str] = {}
    for kind, coords in (("TW", _TW), ("DW", _DW), ("TL", _TL), ("DL", _DL)):
        for r, c in coords:
            for rr, cc in ((r, c), (r, N - 1 - c), (N - 1 - r, c), (N - 1 - r, N - 1 - c)):
                out[(rr, cc)] = kind
                out[(cc, rr)] = kind
                out[(N - 1 - cc, rr)] = kind
                out[(cc, N - 1 - rr)] = kind
    out[(7, 7)] = "DW"
    return out


def _place_words(rng: random.Random, blanks: int = 0) -> list[list[str]]:
    rows = [["." for _ in range(N)] for _ in range(N)]
    for word, r, c, direction in WORDS:
        for i, ch in enumerate(word):
            rr = r + i if direction == "V" else r
            cc = c + i if direction == "H" else c
            if 0 <= rr < N and 0 <= cc < N:
                rows[rr][cc] = ch

    # Turn some placed letters into blanks played as that letter. The expected
    # output for a blank is '?', because pixels cannot tell you which letter a
    # blank was played as - only the game record can.
    if blanks:
        placed = [(r, c) for r in range(N) for c in range(N) if rows[r][c] != "."]
        for r, c in rng.sample(placed, min(blanks, len(placed))):
            rows[r][c] = "?"
    return rows


def render_flat(
    rows: list[list[str]], rng: random.Random, scheme: str | Scheme = "classic"
) -> Image.Image:
    s = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    size = N * CELL + 2 * BORDER
    img = Image.new("RGB", (size, size), s.border)
    draw = ImageDraw.Draw(img)
    draw.rectangle([BORDER, BORDER, size - BORDER, size - BORDER], fill=s.board)

    premium = _premium_map()
    # The same per-platform font search the classifier uses, so these boards
    # render on Windows and macOS and not only in the container. A hard-coded
    # Linux path here meant the whole test suite errored out on a developer
    # machine, which is the one place it most needs to run.
    _, font_path = resolve_font()
    letter_font = ImageFont.truetype(font_path, int(CELL * 0.62))
    small_font = ImageFont.truetype(font_path, int(CELL * 0.22))
    tiny_font = ImageFont.truetype(font_path, int(CELL * 0.17))

    for r in range(N):
        for c in range(N):
            x0 = BORDER + c * CELL
            y0 = BORDER + r * CELL
            x1, y1 = x0 + CELL - 1, y0 + CELL - 1
            kind = premium.get((r, c))
            draw.rectangle([x0, y0, x1, y1],
                           fill=s.premium.get(kind, s.board), outline=s.gridline)
            if kind and rows[r][c] == ".":
                draw.text((x0 + CELL / 2, y0 + CELL / 2), kind, fill=s.premium_ink,
                          font=tiny_font, anchor="mm")

            ch = rows[r][c]
            if ch == ".":
                continue
            # tile, drawn slightly inset and slightly jittered
            j = CELL * 0.05
            dx = np.random.uniform(-j, j)
            dy = np.random.uniform(-j, j)
            pad = CELL * 0.06
            draw.rectangle([x0 + pad + dx, y0 + pad + dy, x1 - pad + dx, y1 - pad + dy],
                           fill=s.tile, outline=s.tile_edge)
            if ch == "?":
                continue                      # a blank tile: face, no markings
            draw.text((x0 + CELL / 2 + dx, y0 + CELL / 2 + dy - CELL * 0.03),
                      ch, fill=s.ink, font=letter_font, anchor="mm")
            # the little point value, printed in the corner as on a real tile
            draw.text((x1 - pad - CELL * 0.10 + dx, y1 - pad - CELL * 0.06 + dy),
                      str(POINTS[ch]), fill=s.ink, font=small_font, anchor="rs")
    return img


def photograph(
    flat: Image.Image, tilt_deg: float, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp, light unevenly, and add sensor noise.

    Returns the photograph and where the playable area's four corners ended up in
    it, ordered top-left, top-right, bottom-right, bottom-left. The corners are
    worth returning because they let a test measure registration error in pixels
    against the exact answer, rather than inferring it from whether the letters
    came out right - by which point a board that is one cell out looks the same as
    a board that could not be read at all.
    """
    bgr = cv2.cvtColor(np.asarray(flat), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    out_w, out_h = int(w * 1.5), int(h * 1.5)

    k = np.tan(np.radians(tilt_deg))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    inset = k * w * 0.30
    pad_x, pad_y = (out_w - w) / 2, (out_h - h) / 2
    dst = np.float32([
        [pad_x + inset, pad_y],
        [pad_x + w - inset, pad_y],
        [pad_x + w + inset * 0.35, pad_y + h],
        [pad_x - inset * 0.35, pad_y + h],
    ])
    # a little in-plane rotation too
    m = cv2.getPerspectiveTransform(src, dst)
    rot = cv2.getRotationMatrix2D((out_w / 2, out_h / 2), rng.uniform(-4, 4), 1.0)
    warped = cv2.warpPerspective(bgr, m, (out_w, out_h), borderValue=(120, 118, 115))
    warped = cv2.warpAffine(warped, rot, (out_w, out_h), borderValue=(120, 118, 115))

    # uneven lighting: a broad diagonal gradient plus vignette
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    grad = 0.80 + 0.35 * (xx / out_w) + 0.12 * (yy / out_h)
    cy, cx = out_h / 2, out_w / 2
    rad = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    grad *= np.clip(1.05 - 0.22 * rad ** 2, 0, None)
    lit = np.clip(warped.astype(np.float32) * grad[..., None], 0, 255)

    noisy = np.clip(lit + np.random.normal(0, 3.0, lit.shape), 0, 255).astype(np.uint8)

    inner = N * CELL
    playable = np.float32([
        [BORDER, BORDER], [BORDER + inner, BORDER],
        [BORDER + inner, BORDER + inner], [BORDER, BORDER + inner],
    ]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(playable, m)
    moved = cv2.transform(moved, rot).reshape(4, 2)

    return cv2.GaussianBlur(noisy, (3, 3), 0), moved.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--tilt", type=float, default=15.0, help="perspective tilt, degrees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blanks", type=int, default=0, help="how many tiles to make blank")
    ap.add_argument("--scheme", default="classic", choices=sorted(SCHEMES),
                    help="board and tile colours")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rows = _place_words(rng, args.blanks)
    photo, corners = photograph(render_flat(rows, rng, args.scheme), args.tilt, rng)

    os.makedirs(args.outdir, exist_ok=True)
    jpg = os.path.join(args.outdir, "board.jpg")
    cv2.imwrite(jpg, photo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    with open(os.path.join(args.outdir, "board.json"), "w") as fh:
        json.dump({"board": ["".join(r) for r in rows], "scheme": args.scheme,
                   "corners": corners.tolist()}, fh, indent=2)

    print(f"wrote {jpg}  ({photo.shape[1]}x{photo.shape[0]}, "
          f"tilt {args.tilt} deg, scheme {args.scheme})")


if __name__ == "__main__":
    main()
