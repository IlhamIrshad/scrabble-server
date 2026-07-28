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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

N = 15
CELL = 48
BORDER = 34
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

TILE = (232, 218, 186)          # cream, RGB
BOARD_BG = (232, 232, 224)
PREMIUM = {
    "TW": (226, 88, 74), "DW": (240, 170, 170),
    "TL": (58, 130, 190), "DL": (168, 208, 230),
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


def render_flat(rows: list[list[str]], rng: random.Random) -> Image.Image:
    size = N * CELL + 2 * BORDER
    img = Image.new("RGB", (size, size), (200, 40, 34))     # red border
    draw = ImageDraw.Draw(img)
    draw.rectangle([BORDER, BORDER, size - BORDER, size - BORDER], fill=BOARD_BG)

    premium = _premium_map()
    letter_font = ImageFont.truetype(FONT, int(CELL * 0.62))
    small_font = ImageFont.truetype(FONT, int(CELL * 0.22))
    tiny_font = ImageFont.truetype(FONT, int(CELL * 0.17))

    for r in range(N):
        for c in range(N):
            x0 = BORDER + c * CELL
            y0 = BORDER + r * CELL
            x1, y1 = x0 + CELL - 1, y0 + CELL - 1
            kind = premium.get((r, c))
            draw.rectangle([x0, y0, x1, y1],
                           fill=PREMIUM.get(kind, BOARD_BG), outline=(170, 170, 165))
            if kind and rows[r][c] == ".":
                draw.text((x0 + CELL / 2, y0 + CELL / 2), kind, fill=(250, 250, 250),
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
                           fill=TILE, outline=(170, 155, 125))
            if ch == "?":
                continue                      # a blank tile: face, no markings
            draw.text((x0 + CELL / 2 + dx, y0 + CELL / 2 + dy - CELL * 0.03),
                      ch, fill=(25, 22, 20), font=letter_font, anchor="mm")
            # the little point value, which the pipeline must ignore
            draw.text((x1 - pad - CELL * 0.10 + dx, y1 - pad - CELL * 0.06 + dy),
                      str(rng.choice([1, 1, 2, 3, 4, 8, 10])), fill=(40, 36, 32),
                      font=small_font, anchor="rs")
    return img


def photograph(flat: Image.Image, tilt_deg: float, rng: random.Random) -> np.ndarray:
    """Perspective-warp, light unevenly, and add sensor noise."""
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
    return cv2.GaussianBlur(noisy, (3, 3), 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--tilt", type=float, default=15.0, help="perspective tilt, degrees")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blanks", type=int, default=0, help="how many tiles to make blank")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rows = _place_words(rng, args.blanks)
    photo = photograph(render_flat(rows, rng), args.tilt, rng)

    os.makedirs(args.outdir, exist_ok=True)
    jpg = os.path.join(args.outdir, "board.jpg")
    cv2.imwrite(jpg, photo, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    with open(os.path.join(args.outdir, "board.json"), "w") as fh:
        json.dump({"board": ["".join(r) for r in rows]}, fh, indent=2)

    print(f"wrote {jpg}  ({photo.shape[1]}x{photo.shape[0]}, tilt {args.tilt} deg)")


if __name__ == "__main__":
    main()
