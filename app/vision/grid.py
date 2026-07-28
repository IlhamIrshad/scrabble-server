"""Split a rectified board into 15x15 cells and find the glyph in each.

The board grid is fixed and known, so we never search for cells. We threshold
the whole board once, collect every dark blob, and assign each blob to the
nearest grid position. That tolerates tiles sitting slightly off-square, which
is the normal case on a real table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import settings

N = 15


@dataclass
class Cell:
    row: int
    col: int
    # tight, normalised binary glyph (glyph_size square), or None if empty
    glyph: np.ndarray | None = None
    occupied: bool = False
    is_blank: bool = False
    mean_v: float = 0.0
    cov: float = 0.0
    mean_s: float = 0.0
    mean_h: float = 0.0
    bbox: tuple[int, int, int, int] | None = None   # in board coords
    notes: list[str] = field(default_factory=list)


def cell_bounds(size: int, row: int, col: int) -> tuple[int, int, int, int]:
    step = size / N
    return (
        int(round(col * step)),
        int(round(row * step)),
        int(round((col + 1) * step)),
        int(round((row + 1) * step)),
    )


def flatten(board: np.ndarray) -> np.ndarray:
    """Divide out low-frequency brightness variation.

    A single camera beside the board always produces a brightness gradient
    across it. Removing that here means every downstream threshold is a global
    constant instead of something that has to be retuned per photograph.
    """
    if not settings.flatten_illumination:
        return board
    lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(np.float32)

    # The background estimate is deliberately very low frequency, so it is
    # computed at a small size and scaled back up. Blurring at full resolution
    # with an equivalent sigma costs 14x more and differs by under 3 grey levels.
    n = settings.flatten_grid
    small = cv2.resize(lightness, (n, n), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(1.0, settings.flatten_sigma_frac * n))
    background = cv2.resize(
        small, (lightness.shape[1], lightness.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    corrected = lightness / np.maximum(background, 1.0) * float(background.mean())
    lab[:, :, 0] = np.clip(corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def binarize(board: np.ndarray) -> np.ndarray:
    """Dark-glyph mask over the whole rectified board."""
    v = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)[:, :, 2]
    v = cv2.GaussianBlur(v, (3, 3), 0)
    block = settings.thresh_block | 1        # force odd
    return cv2.adaptiveThreshold(
        v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, settings.thresh_c,
    )


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Crop to content, scale preserving aspect, centre in a square canvas.

    Returns a uint8 array of 0/1 with shape (glyph_size, glyph_size). The same
    function is used on template glyphs, so the two are directly comparable.
    """
    size, pad = settings.glyph_size, settings.glyph_pad
    out = np.zeros((size, size), np.uint8)

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return out
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    crop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255

    inner = size - 2 * pad
    h, w = crop.shape
    s = inner / float(max(h, w))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    ox = (size - nw) // 2
    oy = (size - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = (crop > 127).astype(np.uint8)
    return out


def _shed_satellites(sub: np.ndarray, step: float) -> np.ndarray:
    """Split a blob that has swallowed the tile's point-value digit.

    Tiles carry a small number in one corner. At board resolution its ink often
    touches the letter's ink, so the two arrive as one connected component, and
    the classifier then sees a letter with a spur on it. An 'O' with a '1' stuck
    to its lower right really is an excellent 'Q'. Left unhandled this costs
    about five points of letter accuracy.

    Eroding by a pixel parts the thin bridge. Deciding what to keep then needs
    two signals, because either alone fails:

    * **Relative area** keeps anything comparable to the biggest fragment. Alone
      it cannot separate a digit that happens to be a sixth of the letter's ink
      from the arm of a 'K'.
    * **Proximity to the biggest fragment's centre** is the decisive one. A
      letter's strokes are all near its own centre; the digit sits out in a
      corner. This is what pushed letter accuracy from 95% to 99.9%.
    """
    eroded = cv2.erode(sub, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=4)
    if n <= 2:                                  # nothing separated
        return sub

    areas = stats[1:, 4]
    biggest = 1 + int(np.argmax(areas))
    bx, by = centroids[biggest]
    min_area = settings.glyph_keep_frac * float(areas.max())
    max_dist = settings.glyph_merge_radius_frac * step

    keep = np.zeros_like(eroded)
    for j in range(1, n):
        if stats[j, 4] < min_area:
            continue
        if float(np.hypot(centroids[j][0] - bx, centroids[j][1] - by)) > max_dist:
            continue
        keep |= (labels == j).astype(np.uint8)

    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(sub, keep * 255)


def _glyph_candidates(mask: np.ndarray, step: float) -> list[dict]:
    """Blobs that plausibly are a single printed letter on a tile.

    Connected components rather than contours, so each candidate carries exact
    pixel membership and can be handed to the classifier in isolation. Cropping
    the raw mask to a bounding box instead drags in whatever else is nearby.

    Four-connectivity, because the bridge between a letter and the tile's corner
    digit is usually a single diagonal touch, which four-connectivity ignores for
    free.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    cell_area = step * step
    out = []
    for label in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[label])
        if max(w, h) > settings.glyph_max_side_frac * step:
            continue

        sub = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255

        if max(w, h) > settings.glyph_split_side_frac * step:
            sub = _shed_satellites(sub, step)
            ys, xs = np.nonzero(sub)
            if len(xs) == 0:
                continue
            x, y = x + int(xs.min()), y + int(ys.min())
            sub = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            h, w = sub.shape
            area = int(np.count_nonzero(sub))

        long_side = max(w, h)
        if not (settings.glyph_min_side_frac * step <= long_side
                <= settings.glyph_max_side_frac * step):
            continue
        if not (settings.glyph_min_area_frac * cell_area
                <= area <= settings.glyph_max_area_frac * cell_area):
            continue
        if w * h <= 0 or area / float(w * h) < settings.glyph_min_extent:
            continue
        if max(w / float(h), h / float(w)) > settings.glyph_max_aspect:
            continue

        out.append({
            "bbox": (x, y, w, h),
            "mask": sub,
            # The bounding-box centre, not the centroid. For a letter with
            # uneven ink distribution the centroid sits off the glyph's visual
            # centre, and it is the visual centre that lands on a cell.
            "cx": x + w / 2.0,
            "cy": y + h / 2.0,
            "area": float(area),
            "long_side": float(long_side),
        })
    return out


def _assign_to_cells(cands: list[dict], step: float) -> dict[tuple[int, int], dict]:
    """One glyph per cell, within max_glyph_shift_frac of the cell centre.

    Where a cell has more than one candidate, the physically largest wins. On a
    tile the letter is always the biggest mark, so this is what discards the
    small point-value digit in the corner - which is otherwise a real hazard,
    because a '0' matched against templates comes back a confident 'D'.
    """
    limit = settings.max_glyph_shift_frac * step
    best: dict[tuple[int, int], tuple[tuple[float, float], dict]] = {}
    for cand in cands:
        col = int(cand["cx"] // step)
        row = int(cand["cy"] // step)
        if not (0 <= row < N and 0 <= col < N):
            continue
        ccx = (col + 0.5) * step
        ccy = (row + 0.5) * step
        d = float(np.hypot(cand["cx"] - ccx, cand["cy"] - ccy))
        if d > limit:
            continue
        key = (row, col)
        rank = (-cand["long_side"], d)        # biggest first, then nearest
        if key not in best or rank < best[key][0]:
            best[key] = (rank, cand)
    return {k: v[1] for k, v in best.items()}


def _cell_stats(
    board_hsv: np.ndarray, row: int, col: int, size: int
) -> tuple[float, float, float, float]:
    """Mean value, coefficient of variance, mean saturation and mean hue."""
    x0, y0, x1, y1 = cell_bounds(size, row, col)
    mx = int((x1 - x0) * 0.28)
    my = int((y1 - y0) * 0.28)
    patch = board_hsv[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if patch.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    v = patch[:, :, 2].astype(np.float32)
    mean = float(v.mean())
    cov = float(v.std()) / mean if mean > 1e-6 else 0.0
    return mean, cov, float(patch[:, :, 1].mean()), float(np.median(patch[:, :, 0]))


def _detect_blanks(cells: dict[tuple[int, int], Cell]) -> None:
    """Mark smooth, warm, saturated empty cells as blank tiles.

    A blank tile face is featureless, so it has a low coefficient of variance -
    but so does bare board, which is the whole difficulty. The separating signal
    is colour, and specifically saturation: on test boards a cream tile face
    reads about 50 against about 10 for plain board, while brightness does not
    separate them at all. Premium squares are also saturated, so hue gates them
    out: cream is warm, premium squares are red or blue.

    Comparison is against *nearby* plain cells, so that a lighting gradient
    across the board does not make every cell on the dim side an outlier.

    This is the least reliable part of the pipeline and it is off by default.
    Note also that even a perfect detection only tells you a blank is present -
    which letter it was played as is not in the pixels at all.
    """
    lo_hue, hi_hue = settings.blank_hue_range
    plain = [c for c in cells.values()
             if not c.occupied and c.mean_s <= settings.blank_plain_max_sat]
    if len(plain) < settings.blank_min_neighbours:
        return

    radius = settings.blank_neighbourhood
    for c in cells.values():
        if c.occupied or c.cov > settings.blank_max_cov:
            continue
        if not (lo_hue <= c.mean_h <= hi_hue):
            continue
        near = [o.mean_s for o in plain
                if o is not c
                and abs(o.row - c.row) <= radius
                and abs(o.col - c.col) <= radius]
        if len(near) < settings.blank_min_neighbours:
            continue
        baseline = float(np.median(near))
        delta = c.mean_s - baseline
        if delta >= settings.blank_min_sat_delta:
            c.occupied = True
            c.is_blank = True
            c.notes.append(f"blank: saturation +{delta:.0f} over local board")


def extract_cells(board: np.ndarray) -> tuple[list[Cell], np.ndarray]:
    """Return 225 Cells (row-major) and the binary mask used, for debugging."""
    size = board.shape[0]
    step = size / N
    even = flatten(board)
    mask = binarize(even)
    hsv = cv2.cvtColor(even, cv2.COLOR_BGR2HSV)

    assigned = _assign_to_cells(_glyph_candidates(mask, step), step)

    cells: dict[tuple[int, int], Cell] = {}
    for row in range(N):
        for col in range(N):
            cell = Cell(row=row, col=col)
            cell.mean_v, cell.cov, cell.mean_s, cell.mean_h = _cell_stats(hsv, row, col, size)
            cand = assigned.get((row, col))
            if cand is not None and cell.mean_s <= settings.max_tile_saturation:
                cell.bbox = cand["bbox"]
                cell.glyph = normalize_glyph(cand["mask"])
                cell.occupied = True
            elif cand is not None:
                cell.notes.append(f"rejected: saturation {cell.mean_s:.0f}")
            cells[(row, col)] = cell

    if settings.detect_blanks:
        _detect_blanks(cells)

    return [cells[(r, c)] for r in range(N) for c in range(N)], mask
