"""Split a rectified board into 15x15 cells and find the glyph in each.

The board grid is fixed and known, so we never search for cells. We threshold
the whole board once, collect every ink blob, and assign each blob to the
nearest grid position. That tolerates tiles sitting slightly off-square, which
is the normal case on a real table.

Nothing here is told what colour anything is. Two assumptions replaced the
colour constants that used to be baked in, and both hold for any tile set:

* **Ink contrasts with its tile, in one direction or the other.** Both polarities
  are thresholded and both are searched, so dark letters on cream tiles and white
  letters on dark red tiles are found by the same code. Assuming dark-on-light
  found nothing at all on a red-tile board - not a misread letter, a blank board.

* **Every tile in a set has the same face colour.** So the tile face need not be
  described in advance; it can be measured from the photograph, by taking the
  colour that the cells holding letters agree on. That measurement then does the
  work an absolute saturation threshold used to do - rejecting premium-square
  lettering, and telling a blank tile from bare board - without needing to know
  whether this board's tiles are cream, red, black or green.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import settings

N = 15

# The standard premium-square layout, given as one eighth of the board and
# unfolded by symmetry. This is the only thing on a board that is *not* periodic,
# which makes it the only thing that can pin the board's absolute position: a
# lattice shifted by a whole cell still lands on every printed grid line and still
# centres every glyph, so nothing else in this pipeline can tell it from the truth.
# Shift this pattern by one cell and it stops matching at all.
_TRIPLE_WORD = [(0, 0), (0, 7), (7, 0)]
_DOUBLE_WORD = [(1, 1), (2, 2), (3, 3), (4, 4)]
_TRIPLE_LETTER = [(1, 5), (5, 1), (5, 5)]
_DOUBLE_LETTER = [(0, 3), (3, 0), (2, 6), (6, 2), (6, 6), (7, 3), (3, 7)]


def _unfold(seed: list[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in seed:
        for rr, cc in ((r, c), (r, N - 1 - c), (N - 1 - r, c), (N - 1 - r, N - 1 - c)):
            out.add((rr, cc))
            out.add((cc, rr))
            out.add((N - 1 - cc, rr))
            out.add((cc, N - 1 - rr))
    return frozenset(out)


# Kept per type rather than merged, because the types carry far more information
# than their union does: "is this square premium?" narrows a cell to one of 61
# positions, where "is it a triple word?" narrows it to one of eight.
#
# That was tried, as a way of *solving* for the board's position rather than only
# checking it - group the visible premium squares by colour, then search jointly
# over every assignment of colour groups to types and every offset, so the layout
# decides which colour is which rather than the other way round. It measured well
# in the one respect that matters for safety: on twelve correctly-registered boards
# it reported "no shift" twelve times, never inventing a correction. But given a
# board that really was shifted it recovered the right offset only 19 times in 48,
# and confidence did not separate its right answers from its wrong ones - one board
# was confidently wrong at a score of 0.74. As a veto it scored exactly the same 71
# of 75 as the plain fraction below, at several times the cost, so it was withdrawn.
#
# Two ambiguities are intrinsic and would have to be dealt with first: the layout is
# symmetric under transposition, so a row shift can look like a column shift; and
# the double-word diagonal maps onto itself under a diagonal shift, so (1,1) is
# genuinely hard to distinguish from no shift at all. The crude colour grouping is
# the other limit, and the easier of the two to improve.
PREMIUM_TYPES: dict[str, frozenset[tuple[int, int]]] = {
    "TW": _unfold(_TRIPLE_WORD),
    "DW": _unfold(_DOUBLE_WORD) | {(7, 7)},   # the centre star is a double word
    "TL": _unfold(_TRIPLE_LETTER),
    "DL": _unfold(_DOUBLE_LETTER),
}
PREMIUM = frozenset().union(*PREMIUM_TYPES.values())


@dataclass
class Cell:
    row: int
    col: int
    # tight, normalised binary glyph (glyph_size square), or None if empty
    glyph: np.ndarray | None = None
    # the tile's printed point value, normalised the same way, when any mark that
    # could be one was found beside the letter
    value_glyph: np.ndarray | None = None
    occupied: bool = False
    is_blank: bool = False
    mean_l: float = 0.0
    # Spread of lightness across the face, in CIELAB L units. Absolute, not
    # relative to the mean: the same speck of noise on a near-black tile and on a
    # cream one is equally smooth, and dividing by the mean says the dark tile is
    # seven times rougher, which is how dark tiles' blanks came to be missed.
    roughness: float = 0.0
    # Approximate CIELAB of the cell's face, ink excluded. None if unmeasurable.
    face: np.ndarray | None = None
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


def cielab(board: np.ndarray) -> np.ndarray:
    """Approximate CIELAB, so that colour distances are perceptual.

    OpenCV packs L into 0-255 and offsets a and b by 128; undoing both puts
    distances in roughly the units CIE Delta-E 1976 is quoted in, which is what
    makes a single tolerance meaningful across quite different tile colours.
    """
    lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    return lab


def delta_e(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32)))


def binarize(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ink masks for both polarities: (darker than its surroundings, lighter).

    Thresholding lightness rather than HSV value is what makes white-on-red
    legible. Value is the largest RGB channel, so a saturated red tile reads 180
    against white ink's 255 - barely a step, and adaptive thresholding at that
    contrast finds the tile's edges instead of its letter. In lightness the same
    pair is 45 against 100.

    The offset is added for the light mask and subtracted for the dark one, so
    that in both cases a flat region falls on the *empty* side of the threshold.
    Reusing the dark mask's sign would mark every flat area as light ink, which
    merges each letter into its own tile and loses it to the size filter.
    """
    lightness = cv2.cvtColor(board, cv2.COLOR_BGR2LAB)[:, :, 0]
    lightness = cv2.GaussianBlur(lightness, (3, 3), 0)
    # Sized as a fraction of a cell, not in pixels. Adaptive thresholding compares
    # a pixel against a window around it, so the window has to mean the same thing
    # relative to a letter whatever resolution the board was warped to - otherwise
    # raising `warp_size` silently halves it and changes what counts as ink.
    step = board.shape[0] / N
    block = max(3, int(round(settings.thresh_block_frac * step)) | 1)
    dark = cv2.adaptiveThreshold(
        lightness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, settings.thresh_c,
    )
    light = cv2.adaptiveThreshold(
        lightness, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block, -settings.thresh_c,
    )
    return dark, light


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


def _shed_satellites(
    sub: np.ndarray, step: float, centre: tuple[float, float] | None
) -> np.ndarray:
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
    * **Proximity to the cell's centre** is the decisive one. A letter is centred
      on its tile; the digit sits out in a corner.

      Measured from the biggest fragment's centre instead, this test throws away
      the tail of a 'Q'. The tail is in the same corner as the digit and about as
      far from the bowl, so there is nothing to tell them apart from there - and a
      'Q' without its tail is an excellent 'O', which is a ten-point tile read as a
      one-point tile. From the cell's centre the two separate cleanly, because the
      tail belongs to a centred glyph and the digit does not.

    On *reading* the digit rather than discarding it: it is the obvious way to
    settle the pairs that overlap cannot, since a tile states its letter twice and
    the second statement is unambiguous - Q is 10 where O is 1, D is 2, C is 3, and
    S is 1 against B's 3. It was tried and measured, and it does not survive this
    resolution. At the ~46 pixels per cell these photographs give, the printed value
    is about eight pixels tall; extracted with no speckle-removal at all, and again
    at twice the warp size, a digit-sized mark could be recovered from only 27% of
    tiles - and the width that should mark a two-digit value apart was matched just
    as often by stray ink from the tile's edge or its neighbour. Acting on it would
    have cost more cells than it fixed.

    It is worth revisiting at around three times the pixels on the board - a
    2500-pixel-wide photograph rather than 960 - where the digit is about 28 pixels
    and can simply be classified. Until then a letter is identified from its own
    shape alone, which is why the *shape* of the reference glyph matters as much as
    it does; see `settings.template_tiles`.

    The erosion is repeated if one pass does not part them. How thick the bridge
    is depends on the photograph, and one pass leaves it intact often enough to
    matter: an unsplit 'E' plus its digit reads as a confident 'B', an 'F' plus its
    digit as a 'P', and a 'B' or 'Q' plus its digit is simply too big to be a
    letter and is dropped, so the tile disappears altogether.
    """
    kernel = np.ones((3, 3), np.uint8)
    for rounds in range(1, settings.glyph_split_rounds + 1):
        eroded = cv2.erode(sub, kernel, iterations=rounds)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            eroded, connectivity=4
        )
        if n > 2:
            break
    else:
        return sub                              # nothing separated

    areas = stats[1:, 4]
    biggest = 1 + int(np.argmax(areas))
    bx, by = centre if centre is not None else centroids[biggest]
    min_area = settings.glyph_keep_frac * float(areas.max())
    max_dist = settings.glyph_merge_radius_frac * step

    keep = np.zeros_like(eroded)
    for j in range(1, n):
        if stats[j, 4] < min_area:
            continue
        if float(np.hypot(centroids[j][0] - bx, centroids[j][1] - by)) > max_dist:
            continue
        keep |= (labels == j).astype(np.uint8)

    keep = cv2.dilate(keep, kernel, iterations=rounds)
    return cv2.bitwise_and(sub, keep * 255)


def _glyph_candidates(
    mask: np.ndarray, step: float, polarity: str,
    centre: tuple[float, float] | None = None,
) -> list[dict]:
    """Blobs that plausibly are a single printed letter on a tile.

    Connected components rather than contours, so each candidate carries exact
    pixel membership and can be handed to the classifier in isolation. Cropping
    the raw mask to a bounding box instead drags in whatever else is nearby.

    Four-connectivity, because the bridge between a letter and the tile's corner
    digit is usually a single diagonal touch, which four-connectivity ignores for
    free.

    `centre` is where in `mask` the cell's middle is, when that is known; it lets
    the split pass tell a letter's own stroke from the tile's point-value digit.
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
            local = None if centre is None else (centre[0] - x, centre[1] - y)
            sub = _shed_satellites(sub, step, local)
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
            "polarity": polarity,
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
    because a '0' matched against templates comes back a confident 'D'. It also
    settles the case of the same tile producing a blob in both polarities: the
    wrong polarity's blob is the tile's own outline or a highlight on it, and
    loses on size or was never a candidate at all.
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


def _cell_glyph(
    lightness: np.ndarray, cell: Cell, size: int, step: float, ink: np.ndarray
) -> dict | None:
    """Extract one tile's letter, thresholded against that tile's own face.

    This is the second of two passes, and the one that actually reads the board.
    A single threshold over the whole board cannot: adaptive thresholding compares
    each pixel against a window around it, and for a tile at the edge of the
    played cluster that window is half bare board. On a board where the tiles are
    darker than the face, the bright board next door lifts the local mean above
    the tile's white lettering, and the letter comes out as a few disconnected
    slivers - which then match as 'I' or 'J' with total confidence. Every tile on
    the outer rows failed this way, and only those tiles, which is what gave the
    game away.

    A tile's own face colour is the right reference and it is already measured, so
    the comparison here is local by construction and needs no window at all.

    Both polarities are tried and the larger blob wins. On a tile the letter is
    the biggest mark either way, so this settles which direction the ink runs
    without assuming it, per tile rather than per board.
    """
    if cell.face is None:
        return None
    x0, y0, x1, y1 = cell_bounds(size, cell.row, cell.col)
    grow = int(round(step * settings.cell_patch_grow_frac))
    px0, py0 = max(0, x0 - grow), max(0, y0 - grow)
    px1, py1 = min(size, x1 + grow), min(size, y1 + grow)
    patch = lightness[py0:py1, px0:px1]
    if patch.size == 0:
        return None

    face_l = float(cell.face[0])
    margin = settings.ink_contrast_min
    limit = settings.max_glyph_shift_frac * step
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    kernel = np.ones((3, 3), np.uint8)

    best: dict | None = None
    for polarity, sign in (("dark", -1.0), ("light", 1.0)):
        # Two steps. The margin finds *some* ink, then the cut is moved to halfway
        # between the face and how dark, or light, that ink actually is.
        #
        # The second step matters more than it looks. At this resolution a stroke
        # has a soft edge a pixel or two wide, and a cut sitting a fixed distance
        # from the face colour swallows that edge and fattens every stroke.
        # Fattened strokes close the gap in a C and the tail on a Q, and both come
        # back as O.
        #
        # The ink's level is taken from the far end of what the margin caught
        # rather than its middle, because the margin also catches the tile's
        # printed outline and moulding, which sit between the face and the ink. An
        # average over that mixture puts the cut on the wrong side of the outline,
        # the outline joins the letter, and an E arrives as a filled square.
        if not (sign * (patch - face_l) > margin).any():
            continue                            # no ink of this polarity here
        # Otsu, over only those pixels lying on the ink's side of the face colour.
        # Otsu asks exactly the right question - where do two populations of
        # brightness best divide - but over the whole cell it is the wrong two: a
        # tile darker than its board splits tile from board and calls the tile
        # ink. Restricted to one side of a face colour that is already known,
        # there are only two populations left, face and ink, and the cut needs no
        # constant of its own.
        side = patch[sign * (patch - face_l) > 0.0]
        if side.size < settings.ink_otsu_min_pixels:
            continue
        level, _ = cv2.threshold(
            np.clip(side * 2.55, 0, 255).astype(np.uint8), 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        hit = sign * (patch - level / 2.55) > 0.0
        mask = cv2.morphologyEx(hit.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        local_centre = (centre[0] - px0, centre[1] - py0)
        for cand in _glyph_candidates(mask, step, polarity, local_centre):
            cx, cy = px0 + cand["cx"], py0 + cand["cy"]
            if float(np.hypot(cx - centre[0], cy - centre[1])) > limit:
                continue
            if best is not None and cand["long_side"] <= best["long_side"]:
                continue
            bx, by, bw, bh = cand["bbox"]
            cand["cx"], cand["cy"] = cx, cy
            cand["bbox"] = (px0 + bx, py0 + by, bw, bh)
            # The whole cell's ink, kept so the point value can be read off it.
            cand["patch"] = mask
            cand["origin"] = (px0, py0)
            best = cand

    if best is not None:
        bx, by, bw, bh = best["bbox"]
        ink[by:by + bh, bx:bx + bw] |= best["mask"]
    return best




def _value_marks(cand: dict, step: float) -> np.ndarray | None:
    """The tile's printed point value, as one mask, or None if nothing was found.

    Everything in the cell that is not the letter and is the size of a digit,
    unioned: a value of 10 is two components, and their combined width is a good
    part of the signal, since no single digit is that wide.

    Only marks at or below the letter's own middle count. The analysis patch is
    grown past the cell so an off-square tile keeps its whole letter, and that
    growth also admits the top of whatever is printed on the tile below - which
    would otherwise be read as this tile's value.
    """
    patch, (ox, oy) = cand["patch"], cand["origin"]
    lx, ly, lw, lh = cand["bbox"]
    lx, ly = lx - ox, ly - oy                   # letter bbox in patch coordinates

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(patch, connectivity=8)
    cell_area = step * step
    keep = np.zeros(patch.shape, np.uint8)
    found = 0
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        # Skip the letter itself: any component substantially inside its box.
        if x >= lx - 2 and x + w <= lx + lw + 2 and y >= ly - 2 and y + h <= ly + lh + 2:
            continue
        if not (settings.value_min_side_frac * step <= max(w, h)
                <= settings.value_max_side_frac * step):
            continue
        if area < settings.value_min_area_frac * cell_area:
            continue
        if centroids[i][1] < ly + 0.35 * lh:
            continue
        keep |= (labels == i).astype(np.uint8) * 255
        found += 1

    return keep if found else None


def _face_colour(
    lab: np.ndarray, ink: np.ndarray, row: int, col: int, size: int
) -> tuple[np.ndarray | None, float, float]:
    """Colour of a cell's face with its ink removed, plus lightness and roughness.

    Ink is excluded because it would otherwise drag the measured face colour
    towards the letter, by an amount that depends on how much ink the letter has
    - which would make 'I' and 'W' on identical tiles look like different tiles.

    Two things make this hold up. The patch is most of the cell, not the middle of
    it: sampling the centre puts the window exactly where the letter is, and on a
    letter-filled cell there is then almost nothing left to average, so the
    measured "face" comes back as the ink. That mistake is quiet and expensive -
    a board's tile colour then reads as its letter colour, every real tile looks
    nothing like it, and the whole board comes back empty.

    And the estimator is the most common lightness in the patch, not the average
    or the middle one. The face is whatever covers the most of a cell, so the mode
    finds it however much ink a letter carries; a median does not. On a letter that
    fills its tile - a 'Q', a 'W' - the median of the patch lands between the face
    and the ink, and a face colour halfway to the ink matches nothing: the tile is
    read as neither tile nor board, and the letter is lost.
    """
    x0, y0, x1, y1 = cell_bounds(size, row, col)
    m = settings.cell_patch_margin_frac
    mx, my = int((x1 - x0) * m), int((y1 - y0) * m)
    patch = lab[y0 + my:y1 - my, x0 + mx:x1 - mx]
    keep = ~ink[y0 + my:y1 - my, x0 + mx:x1 - mx]
    if patch.size == 0:
        return None, 0.0, 0.0
    bare = patch[keep]
    pixels = bare if len(bare) >= 0.25 * keep.size else patch.reshape(-1, 3)
    if len(pixels) < 8:
        return None, 0.0, 0.0

    lightness = pixels[:, 0]
    counts = np.bincount(
        np.clip(lightness * 2.55, 0, 255).astype(np.uint8), minlength=256
    ).astype(np.float32)
    counts = cv2.GaussianBlur(counts.reshape(-1, 1), (0, 0), 4.0).ravel()
    peak = float(np.argmax(counts)) / 2.55

    near = np.abs(lightness - peak) <= settings.face_mode_window
    face = np.median(pixels[near] if near.sum() >= 8 else pixels, axis=0)

    return face.astype(np.float32), float(lightness.mean()), float(lightness.std())


def _dominant_colour(colours: list[np.ndarray], tol: float) -> np.ndarray | None:
    """The colour that most of a set agrees on, or None if they do not agree.

    One mean-shift step seeded at the densest point, rather than k-means, because
    the number of groups is not known: a board may show one tile colour, or a
    tile colour plus a handful of stray premium-square readings. Asking for k
    clusters would invent a split that is not there, and asking for the plain
    median would be dragged off by the strays.
    """
    if len(colours) < settings.tile_cluster_min_members:
        return None
    pts = np.stack(colours).astype(np.float32)
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    counts = (d <= tol).sum(axis=1)
    seed = int(np.argmax(counts))
    if counts[seed] < settings.tile_cluster_min_members:
        return None

    centre = np.median(pts[d[seed] <= tol], axis=0)
    near = np.linalg.norm(pts - centre, axis=1) <= tol
    if near.sum() < settings.tile_cluster_min_members:
        return None
    return np.median(pts[near], axis=0).astype(np.float32)


def _is_tile(
    face: np.ndarray | None,
    tile_face: np.ndarray | None,
    board_face: np.ndarray | None,
) -> bool:
    """Does this cell's face colour belong to a tile rather than to the board?

    Nearer the tile colour than the board colour, and near the tile colour in
    absolute terms too. The comparison has to be relative: an absolute tolerance
    wide enough for shadow across one board is wider than the gap between a cream
    tile and a pale cream board, so every empty square passed the test, every
    empty square was then searched for a letter, and the printed 'DOUBLE WORD
    SCORE' on each premium square duly supplied one.

    Which of the two colours a face is closer to has no such failure. It needs no
    tolerance at all when both colours are known, and the absolute bound is left
    in only to catch a third colour that is neither - a hand, a rack, a shadow
    across the corner of the board.
    """
    if tile_face is None:
        return True                     # no colour established; shape tests only
    if face is None:
        return False
    to_tile = delta_e(face, tile_face)
    if to_tile > settings.tile_face_delta_e:
        return False
    return board_face is None or to_tile < delta_e(face, board_face)


def premium_agreement(cells: list[Cell]) -> tuple[float, int]:
    """How well the board's visible premium squares match the standard layout.

    Returns (fraction landing on a premium square, how many were visible).

    This is the only shift-sensitive measure available. Everything else the
    pipeline knows about a board is periodic - printed grid lines repeat, cells
    repeat, glyphs sit centred in cells whichever cell they are assigned to - so a
    lattice one whole cell out scores identically to the right one, and the error
    that relabels the entire board is invisible. The premium layout does not
    repeat, so a shifted reading scatters its coloured squares onto plain ones.

    Chance is about 0.27, since 61 of 225 squares are premium. A correctly placed
    board scores far above that; one shifted by a cell falls to around chance.

    Only *empty* cells can vote: a premium square under a tile is not visible, and
    late in a game most of them are covered. That is why the count is returned too
    - on a nearly full board there may be too few left to judge by.
    """
    board_face = _dominant_colour(
        [c.face for c in cells if not c.occupied and c.face is not None],
        settings.tile_face_delta_e,
    )
    if board_face is None:
        return 0.0, 0
    detected = {
        (c.row, c.col) for c in cells
        if not c.occupied and c.face is not None
        and delta_e(c.face, board_face) > settings.premium_delta_e
    }
    if len(detected) < settings.premium_min_visible:
        return 0.0, len(detected)
    return len(detected & PREMIUM) / len(detected), len(detected)


def _detect_blanks(
    cells: dict[tuple[int, int], Cell],
    tile_face: np.ndarray | None,
    board_face: np.ndarray | None,
) -> None:
    """Mark smooth cells whose face matches the tile colour as blank tiles.

    A blank tile face is featureless, so it has a low coefficient of variance -
    but so does bare board, which is the whole difficulty. The separating signal
    is colour, and now that the tile colour has been measured from the letters on
    this very board, the test is simply which of the two measured colours the
    cell's face is closer to. That replaces a hand-tuned saturation delta and hue
    window which described one board and one tile set, and quietly reported half
    a cream board's empty squares as blanks.

    If the tiles and the board are too close in colour to tell apart, no blank is
    reported and every cell says so. A wrong blank is a tile that is not there.

    Note that even a perfect detection only tells you a blank is present - which
    letter it was played as is not in the pixels at all.
    """
    if tile_face is None or board_face is None:
        return

    separation = delta_e(tile_face, board_face)
    if separation < settings.blank_min_separation:
        for c in cells.values():
            if not c.occupied:
                c.notes.append(
                    f"blank detection off: tile and board faces differ by only "
                    f"{separation:.0f} Delta-E"
                )
        return

    for c in cells.values():
        if c.occupied or c.roughness > settings.blank_max_roughness:
            continue
        if not _is_tile(c.face, tile_face, board_face):
            continue
        c.occupied = True
        c.is_blank = True
        c.notes.append(
            f"blank: face is {delta_e(c.face, tile_face):.0f} Delta-E from the "
            f"tile colour and {delta_e(c.face, board_face):.0f} from the board"
        )


def _bootstrap_ink(board: np.ndarray, step: float, size: int) -> tuple[np.ndarray, set]:
    """A first, rough pass, whose only job is to find the tile colour.

    Whole-board adaptive thresholding is good enough for that and nothing more:
    it reads most tiles, misreads the ones at the edge of the played cluster, and
    that is fine, because a colour measured from a majority of the tiles is
    correct even when a minority of the letters came out as slivers.
    """
    dark, light = binarize(board)
    assigned = _assign_to_cells(
        _glyph_candidates(dark, step, "dark") + _glyph_candidates(light, step, "light"),
        step,
    )
    ink = np.zeros((size, size), np.uint8)
    for cand in assigned.values():
        x, y, w, h = cand["bbox"]
        ink[y:y + h, x:x + w] |= cand["mask"]
    # Grown a little, so that a glyph's soft edge does not tint the face colour.
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    return ink.astype(bool), set(assigned)


def extract_cells(board: np.ndarray) -> tuple[list[Cell], np.ndarray]:
    """Return 225 Cells (row-major) and the ink mask used, for debugging."""
    size = board.shape[0]
    step = size / N
    even = flatten(board)
    lab = cielab(even)

    boot_ink, lettered_keys = _bootstrap_ink(even, step, size)

    cells: dict[tuple[int, int], Cell] = {}
    for row in range(N):
        for col in range(N):
            cell = Cell(row=row, col=col)
            cell.face, cell.mean_l, cell.roughness = _face_colour(
                lab, boot_ink, row, col, size
            )
            cells[(row, col)] = cell

    # What colour are this board's tiles, and what colour is the board? Whatever
    # the cells holding letters agree on, and whatever the rest agree on. Neither
    # is configured, because neither can be: they differ from board to board and
    # the pair of them is what every decision below turns on.
    tile_face = _dominant_colour(
        [cells[k].face for k in lettered_keys if cells[k].face is not None],
        settings.tile_face_delta_e,
    )
    board_face = _dominant_colour(
        [c.face for k, c in cells.items()
         if k not in lettered_keys and c.face is not None],
        settings.tile_face_delta_e,
    )
    if tile_face is None:
        # Too few letters to establish a colour, so no colour gate is applied.
        # Right for a nearly empty board, and it degrades to shape tests alone
        # rather than rejecting everything.
        for key in lettered_keys:
            cells[key].notes.append("tile colour not established; shape tests only")

    ink = np.zeros((size, size), np.uint8)
    for key, cell in cells.items():
        if not _is_tile(cell.face, tile_face, board_face):
            # Premium-square lettering and printed board text land here. They
            # pass every shape test - they are ink of about the right size - and
            # are only distinguishable by sitting on the board rather than a tile.
            if key in lettered_keys and cell.face is not None and tile_face is not None:
                cell.notes.append(
                    f"rejected: face is {delta_e(cell.face, tile_face):.0f} Delta-E "
                    f"from the tile colour"
                )
            continue
        cand = _cell_glyph(lab[:, :, 0], cell, size, step, ink)
        if cand is None:
            continue
        cell.bbox = cand["bbox"]
        cell.glyph = normalize_glyph(cand["mask"])
        if settings.use_point_value:
            marks = _value_marks(cand, step)
            cell.value_glyph = None if marks is None else normalize_glyph(marks)
        cell.occupied = True

    if settings.detect_blanks:
        _detect_blanks(cells, tile_face, board_face)

    return [cells[(r, c)] for r in range(N) for c in range(N)], ink
