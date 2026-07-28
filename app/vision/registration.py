"""Find the board in a photograph and rectify it to a square.

Two detectors, tried in order:
  1. red_border  - HSV segmentation of the printed red border.
  2. quad        - generic largest-4-sided-contour on Canny edges.

Both return the same thing: four corners in the original image's pixel
coordinates, ordered top-left, top-right, bottom-right, bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import settings


@dataclass
class Registration:
    corners: np.ndarray          # (4, 2) float32, original image coords
    method: str
    area_frac: float
    aspect_skew: float


class RegistrationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = (pts[:, 1] - pts[:, 0])       # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _aspect_skew(corners: np.ndarray) -> float:
    """Ratio of the longer mean side to the shorter. 1.0 is a perfect square."""
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    w = (top + bottom) / 2.0
    h = (left + right) / 2.0
    if min(w, h) < 1e-6:
        return 1e9
    return float(max(w, h) / min(w, h))


def _quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    """Reduce a contour to a convex 4-gon, or None."""
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    # Fall back to the minimum-area rectangle, which is always a quad. This
    # copes with a board whose corners are rounded or partly occluded.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(box) > 0.55 * cv2.contourArea(contour):
        return box
    return None


def _best_quad(mask: np.ndarray, frame_area: float) -> tuple[np.ndarray, float] | None:
    """Largest acceptable quad in a binary mask. Returns (corners, area_frac)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[np.ndarray, float] | None = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < settings.min_board_area_frac * frame_area:
            continue
        quad = _quad_from_contour(c)
        if quad is None:
            continue
        ordered = order_corners(quad)
        if _aspect_skew(ordered) > settings.max_board_aspect_skew:
            continue
        if area > best_area:
            best_area = area
            best = (ordered, area / frame_area)
    return best


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------


def _detect_red_border(bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Segment the printed red border and return the board's outer quad.

    Note on a tempting shortcut that does not work: the border is a ring, so it
    has a hole, and that hole looks like it should be exactly the playable grid -
    no inset constant needed. It is not. A Scrabble board's triple-word squares
    are red and sit on the outermost row, so they merge with the border under any
    morphological closing and eat into the hole unevenly. The resulting quad is
    a cell out on some edges and not others, which no later correction can undo.

    So we take the outer edge, which is clean, and let the grid-lock pass work
    out the inset from the printed grid lines.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo_s, lo_v = settings.red_sat_min, settings.red_val_min
    m1 = cv2.inRange(hsv, (0, lo_s, lo_v), (settings.red_hue_lo, 255, 255))
    m2 = cv2.inRange(hsv, (settings.red_hue_hi, lo_s, lo_v), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Fill the ring so the whole board becomes one blob to take a quad from.
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    h, w = bgr.shape[:2]
    return _best_quad(filled, float(h * w))


def _detect_quad(bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)

    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    h, w = bgr.shape[:2]
    return _best_quad(edges, float(h * w))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def find_board(bgr: np.ndarray, corners: np.ndarray | None = None) -> Registration:
    """Locate the board. If `corners` is given it is trusted and returned."""
    if corners is not None:
        ordered = order_corners(corners)
        h, w = bgr.shape[:2]
        return Registration(ordered, "supplied", 0.0, _aspect_skew(ordered))

    h, w = bgr.shape[:2]
    scale = min(1.0, settings.detect_max_edge / float(max(h, w)))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else bgr

    for name, fn in (("red_border", _detect_red_border), ("quad", _detect_quad)):
        found = fn(small)
        if found is None:
            continue
        quad, area_frac = found
        quad = quad / scale                      # back to full-res coords
        return Registration(quad, name, area_frac, _aspect_skew(quad))

    raise RegistrationError(
        "No board found. Try a less oblique angle, more even lighting, or pass "
        "explicit corners in the request."
    )


def _homography(corners: np.ndarray, size: int, inset_frac: float) -> np.ndarray:
    e = inset_frac * size
    dst = np.array(
        [[-e, -e], [size + e, -e], [size + e, size + e], [-e, size + e]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(order_corners(corners), dst)


def _apply(bgr: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpPerspective(
        bgr, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _edge_projections(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row sums of directional edge energy.

    Vertical grid lines pile up in the column projection, horizontal ones in the
    row projection. A broad baseline is removed so that a dark region of the
    board cannot outvote the lines.
    """
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    px = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)).sum(axis=0)
    py = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).sum(axis=1)

    def detrend(v: np.ndarray) -> np.ndarray:
        k = max(3, (len(v) // 15) | 1)
        base = cv2.blur(v.reshape(-1, 1), (1, k)).ravel()
        return np.maximum(v - base, 0.0)

    return detrend(px), detrend(py)


_LINES = np.arange(16)


def _comb_scores(proj: np.ndarray, phases: np.ndarray, period: float) -> np.ndarray:
    idx = np.rint(phases[:, None] + _LINES[None, :] * period).astype(np.int32)
    np.clip(idx, 0, len(proj) - 1, out=idx)
    return proj[idx].sum(axis=1)


def _comb_fit(px: np.ndarray, py: np.ndarray) -> tuple[float, float, float]:
    """Fit a 15x15 lattice to both edge projections. Returns (phase_x, phase_y, period).

    Three constraints make this reliable. It is not reliable without any of them:

    * **Shared pitch.** The board is square and already rectified, so the cell
      pitch must be the same on both axes. Fitting one pitch against both
      projections doubles the evidence for it.
    * **Shared phase, in the coarse pass.** A board's printed border is uniform,
      so if the detected quad is the board's outer edge then both axes are inset
      by the same amount. Tying the phases removes the degree of freedom that
      otherwise lets one axis lock a whole row off - the worst failure available
      here, because it silently relabels every tile on the board.
    * **A narrow pitch range.** The coarse warp is already close. Given a wide
      range, the fit finds a denser lattice among the tile edges and letters.

    A bounded per-axis refinement then recovers genuine small asymmetry, without
    enough freedom to jump a cell.
    """
    size = len(px)
    nominal = size / 15.0
    lo, hi = settings.grid_period_range

    best_period, best_phase, best_score = nominal, 0.0, -1e18
    for period in np.linspace(lo * nominal, hi * nominal, 200):
        span = 15.0 * period
        if span >= size:
            continue
        phases = np.arange(0.0, size - span + 1e-6, 0.5)
        if len(phases) == 0:
            continue
        scores = _comb_scores(px, phases, period) + _comb_scores(py, phases, period)
        i = int(np.argmax(scores))
        if scores[i] > best_score:
            best_period, best_phase, best_score = float(period), float(phases[i]), float(scores[i])

    # Bounded independent refinement, +/- 0.15 of a cell.
    slack = 0.15 * best_period
    fine = np.arange(best_phase - slack, best_phase + slack + 1e-6, 0.25)
    fine = fine[(fine >= 0.0) & (fine + 15.0 * best_period < size)]
    if len(fine):
        fx = float(fine[int(np.argmax(_comb_scores(px, fine, best_period)))])
        fy = float(fine[int(np.argmax(_comb_scores(py, fine, best_period)))])
    else:
        fx = fy = best_phase

    return fx, fy, best_period


def warp_board(bgr: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rectify the board so the 15x15 playable grid exactly fills the output.

    Two passes. The first uses the detected quad. The second locks onto the
    printed grid lines found in that image, which corrects for the quad having
    landed on the board's outer edge rather than the playable area - the usual
    case, and the thing that silently shifts every cell if left uncorrected.
    """
    size = settings.warp_size
    nominal = size / 15.0
    m = _homography(corners, size, settings.grid_inset_frac)
    coarse = _apply(bgr, m, size)

    px, py = _edge_projections(coarse)
    fx, fy, period = _comb_fit(px, py)

    # Refuse an implausible lock. Being a little misaligned costs a few
    # low-confidence cells; being a whole row out corrupts the entire board.
    inset = size - 15.0 * period
    plausible = (
        abs(period - nominal) / nominal <= 0.20
        and -1.0 <= fx <= inset + 0.5 * nominal
        and -1.0 <= fy <= inset + 0.5 * nominal
    )
    if not plausible:
        return coarse, {
            "locked": False,
            "reason": "grid fit rejected as implausible; using coarse warp",
            "grid_period": round(period, 2),
        }

    s = size / (15.0 * period)
    a = np.array([[s, 0.0, -fx * s], [0.0, s, -fy * s], [0.0, 0.0, 1.0]])
    fine = _apply(bgr, a @ m, size)

    return fine, {
        "locked": True,
        "grid_phase": (round(fx, 2), round(fy, 2)),
        "grid_period": round(period, 2),
        "grid_scale": round(s, 4),
    }
