"""Find the board in a photograph and rectify it to a square.

The board is located by *what it is*, not by what colour it happens to be painted.
Whatever its border and face colours, every Scrabble board contains one thing no
table, cloth or turntable does: a 15x15 lattice of printed grid lines filling a
square region. Detection therefore has three stages.

**Propose.** Several cheap, deliberately overlapping detectors each offer candidate
quads - regions of uniform colour at a few clustering granularities (which finds a
printed border of *any* hue, and also the board face itself), regions busier than a
bare table, and four-sided contours on Canny edges.

**Refine.** Each candidate's four edges are slid onto the strongest straight
boundaries near them. Region proposals get the board's position about right and its
shape wrong, and shape - rotation and perspective - is what the grid lock
downstream cannot repair.

**Verify.** Every candidate is trial-warped and fitted with a 15x15 comb, then
judged on two things that are not interchangeable: whether a lattice is there at
all, which rejects everything that is not a board, and how exactly its lines land,
which chooses between quads that all contain one.

Verification is what a colour-keyed detector cannot do without. A red turntable
under the board is a large, convincingly square-ish red region, and a detector that
only asks "is it red?" cannot tell it from a red border - it picked the turntable
on every photograph that had one. Asking "does a 15x15 grid fit inside this?"
separates them, and asks nothing about colour.

Everything returns four corners in the original image's pixel coordinates, ordered
top-left, top-right, bottom-right, bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import settings


@dataclass
class Registration:
    corners: np.ndarray          # (4, 2) float32, original image coords
    method: str
    area_frac: float
    aspect_skew: float
    grid_score: float = 0.0      # is a 15x15 lattice present at all
    grid_alignment: float = 0.0  # how exactly its lines land on the cut lines
    # (method, alignment) for every candidate accepted as a board, best first.
    # Diagnostic only, but the first thing worth looking at when detection picks
    # the wrong quad.
    considered: list[tuple[str, float]] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        """Does the geometry look sound enough to trust without a person checking?

        Calibrated against photographs a person judged by eye - see the settings.
        A false here is worth acting on: of the boards it doubts, all but one in
        sixty really were misregistered. A true is much weaker, because a quarter
        of the boards it accepts are wrong too, so it means "nothing looks amiss",
        never "this is correct".
        """
        return (self.grid_score >= settings.confident_grid_score
                and self.grid_alignment >= settings.confident_grid_alignment)


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
    # A board is convex, so the hull loses nothing and gains a great deal: the
    # texture and colour regions are ragged, and polygon approximation of a
    # ragged outline rarely yields four sides. The hull of the same region is
    # close to the square we are looking for.
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    # Fall back to the minimum-area rectangle, which is always a quad. This
    # copes with a board whose corners are rounded or partly occluded.
    #
    # Its weakness is real and not yet fixed: a rectangle's opposite sides are
    # parallel by construction, so it cannot describe a board photographed from a
    # low angle, whose sides converge. The warp then stays trapezoidal, the cell
    # pitch is not uniform, and no lattice fits however good the rest is.
    #
    # Offering shapes that *can* express a keystone - the hull's four longest sides
    # extended and intersected, and the hull trimmed to its four dominant corners -
    # was tried and withdrawn. It helped some photographs and made things worse
    # overall: scored against 75 boards a person had confirmed correct, 13 moved
    # more than a third of a cell and 7 by more than a whole cell. The reason is
    # that the selector cannot police it. Candidates are chosen by warping each one
    # and seeing which reads best, and that measure is *blind to whole-cell shifts*
    # - a lattice one cell out still centres every glyph perfectly, in the wrong
    # cell. More candidate shapes therefore means more chances to win with a
    # shifted lattice. Fixing the keystone needs a selector that pins the board's
    # absolute position first; see `scripts/score_corners.py` for the measurement
    # that showed this.
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)
    if cv2.contourArea(box) > 0.55 * cv2.contourArea(hull):
        return box
    return None


def _trim_hull_to_quad(hull: np.ndarray) -> np.ndarray | None:
    """Drop hull vertices, cheapest corner first, until four remain.

    Polygon approximation asks for a tolerance and returns whatever number of sides
    that happens to produce; this asks for four sides directly, which is what we
    actually know about a board. Repeatedly discarding whichever vertex encloses the
    least area converges on the four dominant corners of a ragged outline.
    """
    pts = hull.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return None
    while len(pts) > 4:
        before, after = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        u, v = pts - before, after - before
        # Twice the triangle's area, written out because numpy 2 removed the
        # two-dimensional form of np.cross.
        lost = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        pts = np.delete(pts, int(np.argmin(lost)), axis=0)
    return pts.astype(np.float32)


def _quad_from_long_edges(hull: np.ndarray) -> np.ndarray | None:
    """Extend the hull's four longest sides and intersect them.

    This is the one that can express a keystone, and the reason it is worth having:
    `minAreaRect` is a rectangle, so its opposite sides are parallel by
    construction and it cannot describe a board photographed from a low angle,
    whose sides converge. Fitting a rectangle to a trapezium leaves a warp that is
    still trapezoidal - the cell pitch is then not uniform and no lattice fits,
    however good everything downstream is.

    Treating the sides as *lines* and intersecting them recovers the true corners,
    including where a corner is rounded, occluded, or outside the frame entirely.
    """
    pts = hull.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return None
    nxt = np.roll(pts, -1, axis=0)
    lengths = np.linalg.norm(nxt - pts, axis=1)
    # Kept in cyclic order, so consecutive pairs really are adjacent sides.
    longest = sorted(int(i) for i in np.argsort(-lengths)[:4])
    if len(longest) < 4:
        return None

    corners = []
    for i in range(4):
        a, b = longest[i - 1], longest[i]
        point = _intersect((pts[a], nxt[a]), (pts[b], nxt[b]))
        if point is None:
            return None
        corners.append(point)
    quad = np.array(corners, dtype=np.float32)
    return quad if cv2.isContourConvex(order_corners(quad).reshape(-1, 1, 2)) else None


def _quads_from_contour(contour: np.ndarray) -> list[np.ndarray]:
    """Every reasonable reading of one contour as a four-sided board.

    Several rather than one, because which method is right depends on the shape and
    there is no way to tell in advance - and because the caller now judges a quad by
    warping it, reading the board, and checking the premium squares fall where the
    printed layout says they should. That last test is what makes offering more
    shapes safe: an earlier attempt at this made things worse, because the selector
    of the day could not tell a good quad from one a whole cell out.
    """
    hull = cv2.convexHull(contour)
    out: list[np.ndarray] = []
    for quad in (_quad_from_contour(contour), _quad_from_long_edges(hull),
                 _trim_hull_to_quad(hull)):
        if quad is None:
            continue
        if not any(np.abs(quad - kept).max() < 2.0 for kept in out):
            out.append(quad)
    return out


def _fills_frame(quad: np.ndarray, shape: tuple[int, int]) -> bool:
    """True if the quad is essentially the whole photograph.

    The Canny detector reliably offers one of these when the board's own edges
    are weak: the outermost closed contour it can find is the image boundary
    itself. It is never the board, and it warps the entire photograph into the
    output, so it is excluded before scoring rather than left to lose on score.
    """
    h, w = shape
    x0, y0 = quad[:, 0].min(), quad[:, 1].min()
    x1, y1 = quad[:, 0].max(), quad[:, 1].max()
    return (x1 - x0) >= 0.97 * w and (y1 - y0) >= 0.97 * h


def _quads(mask: np.ndarray, shape: tuple[int, int], limit: int) -> list[np.ndarray]:
    """Up to `limit` acceptable quads in a binary mask, largest area first.

    Every contour, not just the outermost ones. A board on a round table is
    *inside* the table's outline, so retrieving only external contours returns
    the table and discards the board - which on these photographs is a clean
    square sitting in the edge map, thrown away for being nested.
    """
    h, w = shape
    frame_area = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[float, np.ndarray]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (settings.min_board_area_frac * frame_area
                <= area <= settings.max_board_area_frac * frame_area):
            continue
        primary = _quad_from_contour(c)
        if primary is None:
            continue
        shapes = [primary]
        if settings.quad_shapes_per_contour > 1:
            shapes = _quads_from_contour(c)[: settings.quad_shapes_per_contour]
        for quad in shapes:
            ordered = order_corners(quad)
            if _aspect_skew(ordered) > settings.max_board_aspect_skew:
                continue
            if _fills_frame(ordered, shape):
                continue
            # A quad built by extending sides can reach past the contour it came
            # from, so its own area is what has to be plausible, not the contour's.
            own = cv2.contourArea(ordered.astype(np.float32))
            if not (settings.min_board_area_frac * frame_area
                    <= own <= settings.max_board_area_frac * frame_area):
                continue
            scored.append((own, ordered))
    scored.sort(key=lambda t: -t[0])
    return [q for _, q in scored[:limit]]


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Solidify a mask so a ring or a lattice becomes one blob.

    Both shapes turn up: a printed border is a ring, and a board face seen
    between the tiles on it is a lattice. Either way the blob we want a quad
    from is the filled outline, not the strokes.
    """
    out = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


# ---------------------------------------------------------------------------
# candidate proposal
# ---------------------------------------------------------------------------


def _colour_region_masks(bgr: np.ndarray) -> list[np.ndarray]:
    """One mask per dominant colour in the photograph, at several granularities.

    This replaces the old hard-coded red-border segmentation. Rather than being
    told which colour the border is, we cluster the image's colours and offer
    every cluster as a candidate region; the border, whatever its hue, is one of
    them, and so is the board face. Scoring then picks whichever actually contains
    a grid.

    Clustering runs at more than one `k` because the right number of colours is
    not knowable in advance and getting it wrong loses the board entirely. At a
    small k the board face merges with the table; at a large one it splits into
    lit and shadowed halves. On one test photograph the board was a clean cluster
    at one k and absent at the next, which made detection turn on a number chosen
    for no better reason than that it usually works.

    Clustering runs on a thumbnail. At full detection resolution k-means costs
    about a second; on a 160 px thumbnail it costs a few milliseconds and the
    cluster centres come out the same, because we are after large flat regions
    and nothing else.
    """
    h, w = bgr.shape[:2]
    edge = settings.detect_cluster_edge
    s = min(1.0, edge / float(max(h, w)))
    thumb = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    thumb = cv2.GaussianBlur(thumb, (0, 0), 1.0)

    lab = cv2.cvtColor(thumb, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    distinct = len(np.unique(lab, axis=0))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Close by roughly a cell so a border broken by triple-word squares, or a face
    # broken by the tiles on it, becomes continuous. This has to happen where the
    # board's cells are resolved. Doing it on the thumbnail is far cheaper and does
    # something else entirely - a cell is barely a pixel across there - and loses
    # whole boards; doing it at half resolution saves nothing measurable and costs
    # a little accuracy. So it happens here, at full detection resolution.
    span = max(3, int(round(max(h, w) * 0.02)) | 1)
    kernel = np.ones((span, span), np.uint8)
    tiny = np.ones((3, 3), np.uint8)

    out = []
    for k in settings.detect_colour_clusters:
        if k < 2 or k > distinct:
            continue
        # k-means seeds its centres at random, so without pinning the seed the
        # same photograph produces a different set of candidate regions on every
        # call, and therefore sometimes registers correctly and sometimes does
        # not. That is far worse than registering badly every time: it cannot be
        # reproduced from a report, and a test that passes tells you nothing
        # about the next run.
        cv2.setRNGSeed(settings.detect_rng_seed)
        _, labels, _ = cv2.kmeans(lab, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(thumb.shape[:2])
        for i in range(k):
            m = (labels == i).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, tiny)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            out.append(_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)))
    return out


def _texture_masks(bgr: np.ndarray) -> list[np.ndarray]:
    """Where the photograph is busy - the colour-free backstop, two ways.

    The playable area is wall-to-wall detail: grid lines, tile edges, letters,
    point values. Tables, cloths and turntables are smooth. Neither reading of
    "busy" works on its own, so both are offered:

    * **Otsu on blurred gradient energy** splits the frame into its busier and
      quieter halves. Decisive when the board's detail is strong, and too decisive
      when it is not: on a board whose grid lines are pale grey on a pale grey
      face, the only thing above the threshold is the tiles, and the region
      collapses to the shape of the words played so far.
    * **Local variation against the smoothest part of the frame** asks the gentler
      question - is this quieter than a bare table? - and so keeps the board's
      empty squares. Being a lower bar, it is the more easily fooled of the two by
      a patterned cloth.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    sigma = max(3.0, min(bgr.shape[:2]) * settings.texture_sigma_frac)
    span = max(3, int(round(sigma)) | 1)
    kernel = np.ones((span, span), np.uint8)

    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
                              (0, 0), sigma)
    scaled = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, strict = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Local standard deviation, thresholded relative to the quietest part of the
    # frame rather than to the frame's own average. A bare table is smooth in
    # absolute terms - a few grey levels of sensor noise - so anything a few times
    # rougher than the quietest region is an object sitting on it.
    flat = gray.astype(np.float32)
    mean = cv2.boxFilter(flat, -1, (span, span))
    variance = np.maximum(cv2.boxFilter(flat * flat, -1, (span, span)) - mean * mean, 0.0)
    deviation = np.sqrt(variance)
    floor = float(np.quantile(deviation, settings.texture_floor_quantile))
    loose = (deviation > max(floor * settings.texture_floor_multiple, 1.0))
    loose = (loose.astype(np.uint8) * 255)

    return [_fill_holes(cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel))
            for m in (strict, loose)]


def _edge_mask(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 60, 60)

    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def _outward_normals(quad: np.ndarray) -> np.ndarray:
    """Unit outward normal of each of the four edges, in quad order."""
    nxt = np.roll(quad, -1, axis=0)
    d = nxt - quad
    length = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
    # order_corners returns TL, TR, BR, BL, which is clockwise with y downwards,
    # so rotating each edge direction this way points away from the interior.
    return np.stack([d[:, 1] / length, -d[:, 0] / length], axis=1)


def _already_have(
    quad: np.ndarray, kept: list[tuple[str, np.ndarray]], shape: tuple[int, int]
) -> bool:
    """True if a quad this close to `quad` has already been collected."""
    tolerance = settings.candidate_dedupe_frac * max(shape)
    return any(np.abs(quad - other).max() < tolerance for _, other in kept)


def _line_energy(
    gx: np.ndarray, gy: np.ndarray, normal: np.ndarray,
    a: np.ndarray, b: np.ndarray, samples: int,
) -> np.ndarray:
    """Mean across-edge gradient along each of a batch of line segments.

    `a` and `b` are (K, 2) arrays of endpoints; the result is (K,).

    The gradient is projected onto the normal at the sampled points, not over the
    whole frame. Building the projected image first is the obvious way round and
    costs about eighty times more: there are a couple of hundred thousand sampled
    points in a whole snap, against a million pixels multiplied and added twelve
    times over for every candidate quad.
    """
    t = np.linspace(0.0, 1.0, samples)[None, :]
    xs = a[:, 0:1] + (b[:, 0:1] - a[:, 0:1]) * t
    ys = a[:, 1:2] + (b[:, 1:2] - a[:, 1:2]) * t
    h, w = gx.shape
    ix = np.rint(xs).astype(np.int32)
    iy = np.rint(ys).astype(np.int32)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    np.clip(ix, 0, w - 1, out=ix)
    np.clip(iy, 0, h - 1, out=iy)
    across = np.abs(gx[iy, ix] * normal[0] + gy[iy, ix] * normal[1])
    return np.where(inside, across, 0.0).mean(axis=1)


def _snap_edge(
    gx: np.ndarray, gy: np.ndarray, p: np.ndarray, q: np.ndarray,
    normal: np.ndarray, radius: float, keep_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide an edge's two endpoints along the normal onto the strongest line.

    Both endpoints move independently, so the edge can rotate as well as
    translate. That matters: a region-derived proposal is often tilted by several
    degrees relative to the board, and rotation is the one error the grid lock
    downstream cannot repair.

    Only the component of the gradient across the edge is counted, so a strong
    edge running the wrong way contributes nothing.
    """
    offsets = np.linspace(-radius, radius, settings.snap_offsets)
    d1, d2 = (v.ravel() for v in np.meshgrid(offsets, offsets, indexing="ij"))

    a = p[None, :] + d1[:, None] * normal[None, :]
    b = q[None, :] + d2[:, None] * normal[None, :]
    score = _line_energy(gx, gy, normal, a, b, settings.snap_samples)

    # Among lines within `keep_frac` of the best, take the outermost. A board's
    # outer edge and its first printed grid line are both strong and parallel, one
    # cell apart, and picking between them by strength alone is a coin toss - which
    # is a whole row of the board. Preferring the outer one is at least consistent,
    # and the grid lock then works out the inset from the printed grid anyway.
    #
    # Consistent, but not always right, which is why the caller asks twice: at
    # keep_frac 1.0 this is simply the strongest line. Neither rule wins on every
    # board, so both quads are offered and scored.
    good = np.nonzero(score >= keep_frac * score.max())[0]
    pick = good[np.argmax(d1[good] + d2[good])]
    return a[pick], b[pick]


def _gradients(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal and vertical image gradient, computed once per photograph."""
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return (cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def _snap_quad(
    bgr: np.ndarray, quad: np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | None = None,
    keep_frac: float | None = None,
) -> np.ndarray | None:
    """Fit the quad's four edges to the board's actual straight boundaries.

    Region proposals get the board's rough position right and its shape wrong -
    a texture region shrinks towards wherever the tiles are densest, a colour
    region merges the board with anything of a similar hue. Neither produces
    edges that are straight or square, and the grid lock cannot rescue a quad
    whose rotation or perspective is wrong, only its offset and scale.

    Snapping is coarse-to-fine so that a badly tilted edge can travel a long way
    on the first pass and settle precisely on the last.
    """
    gx, gy = gradients if gradients is not None else _gradients(bgr)
    keep = settings.snap_keep_frac if keep_frac is None else keep_frac

    quad = order_corners(quad)
    side = float(np.sqrt(max(cv2.contourArea(quad.astype(np.float32)), 1.0)))

    for frac in settings.snap_radius_fracs:
        normals = _outward_normals(quad)
        lines = []
        for i in range(4):
            p, q = quad[i], quad[(i + 1) % 4]
            lines.append(_snap_edge(gx, gy, p, q, normals[i], frac * side, keep))

        corners = []
        for i in range(4):
            point = _intersect(lines[i - 1], lines[i])
            if point is None:
                return None
            corners.append(point)
        quad = order_corners(np.array(corners, dtype=np.float32))

    if _aspect_skew(quad) > settings.max_board_aspect_skew:
        return None
    if _fills_frame(quad, bgr.shape[:2]):
        return None
    area = cv2.contourArea(quad.astype(np.float32))
    frame_area = float(bgr.shape[0] * bgr.shape[1])
    if not (settings.min_board_area_frac * frame_area
            <= area <= settings.max_board_area_frac * frame_area):
        return None
    return quad


def _intersect(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> np.ndarray | None:
    """Where two lines, each given as a pair of points, cross."""
    (a, b), (c, d) = first, second
    m = np.array([b - a, c - d], dtype=np.float64).T
    det = np.linalg.det(m)
    if abs(det) < 1e-9:
        return None
    t = np.linalg.solve(m, (c - a).astype(np.float64))
    return (a + t[0] * (b - a)).astype(np.float32)


def _candidates(bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Every quad worth scoring, tagged with the detector that proposed it."""
    shape = bgr.shape[:2]
    out: list[tuple[str, np.ndarray]] = []

    for i, mask in enumerate(_colour_region_masks(bgr)):
        for quad in _quads(mask, shape, limit=2):
            out.append((f"colour[{i}]", quad))
    for tag, mask in zip(("texture", "rough"), _texture_masks(bgr)):
        for quad in _quads(mask, shape, limit=2):
            out.append((tag, quad))
    for quad in _quads(_edge_mask(bgr), shape, limit=3):
        out.append(("edges", quad))

    # Two detectors often agree to within a pixel or two. Scoring duplicates is
    # harmless but wasteful, and it clutters the diagnostic list.
    unique: list[tuple[str, np.ndarray]] = []
    for name, quad in out:
        if not _already_have(quad, unique, shape):
            unique.append((name, quad))
    return unique


# ---------------------------------------------------------------------------
# grid fitting - used both to score candidates and to lock the final warp
# ---------------------------------------------------------------------------


@dataclass
class GridFit:
    phase_x: float
    phase_y: float
    period: float
    nominal: float      # cell pitch the candidate quad implies, same units
    score: float        # is a lattice present at all; ~1.0 means no
    alignment: float    # how exactly it is aligned; below 1.0 means half a cell out
    plausible: bool

    @property
    def period_frac(self) -> float:
        """Playable width as a fraction of the candidate quad's width.

        The only scale-free way to report the fit, and the number worth reading:
        below 1.0 the quad included a border, above 1.0 it cropped into the
        playable area.
        """
        return self.period / self.nominal if self.nominal > 1e-6 else 0.0


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
        v = np.maximum(v - base, 0.0)
        # Clip so that no single edge can outweigh the grid. The ceiling is a
        # quantile of the positions that survived detrending - typically a few
        # hundred out of several hundred - which puts it at about the strength of
        # an ordinary grid line: every real line saturates it, and the fit is
        # then counting lines found rather than adding up contrast. Unclipped,
        # the boundary between a bright turntable and a dark cloth outweighs the
        # entire printed grid and drags the comb onto itself.
        #
        # The quantile has to be taken over the non-zero positions. Detrending
        # zeroes most of a projection made of sharp lines, so a quantile over all
        # of them can itself be zero, and clipping at zero erases the grid.
        live = v[v > 0.0]
        if live.size:
            v = np.minimum(v, np.quantile(live, settings.grid_line_clip_quantile))
        return v

    return detrend(px), detrend(py)


_LINES = np.arange(16)


def _comb_scores(proj: np.ndarray, phases: np.ndarray, period: float) -> np.ndarray:
    idx = np.rint(phases[:, None] + _LINES[None, :] * period).astype(np.int32)
    np.clip(idx, 0, len(proj) - 1, out=idx)
    return proj[idx].sum(axis=1)


def _periodicity(proj: np.ndarray, period: float) -> float:
    """How much a 15-cell comb at this pitch prefers one phase over any other.

    The comb's best response divided by its typical response. Dimensionless, so
    it compares across candidate quads of different sizes and photographs of
    different contrast: a region with no lattice in it has no phase preference
    and scores about 1.0, while a printed grid scores well above that.

    Two wrong measures were tried first, and both fail in ways worth recording,
    because each looks perfectly reasonable until it is put in front of a real
    photograph:

    * **Total energy landing on the comb, over chance.** A quad holding one small
      busy object in a sea of tablecloth puts nearly all of its edge energy into
      a handful of rows, so a comb line crossing that object collects most of the
      frame's energy and the candidate outscores the real board. Turntable and
      cloth beat board.
    * **Requiring all sixteen lines to be individually strong.** True of a board
      with visible gutters, false of a board late in a game: where tiles abut,
      the printed line between them is simply not in the photograph. Boards with
      the most tiles on them - the ones most worth reading - scored zero.

    A phase preference survives both. Missing lines weaken it without destroying
    it, and energy concentrated in one place cannot create it, because a comb has
    sixteen teeth and most phase offsets put one of them on the bright spot.

    "Typical" is the median over phases, floored at a small fraction of the mean.
    Both parts are needed and each was arrived at by getting it wrong:

    * The plain median is zero for the sharpest grids. Grid lines are two pixels
      wide in a pitch of thirty, so most phase offsets land the whole comb between
      lines and score nothing - and dividing by that scored the crispest boards
      lowest of all, which is how a correct quad came to be rejected outright.
    * The plain mean does not discriminate. Sixteen teeth two pixels wide span more
      than one pitch between them, so almost every phase offset catches something
      and the mean sits close to the maximum. Good and wrong quads then score
      within a few percent of each other and the threshold has nothing to bite on.

    The floor keeps the median's separation - boards score tens, non-boards score
    about one - while bounding the ratio instead of letting it divide by zero.
    """
    span = 15.0 * period
    if span >= len(proj):
        return 0.0
    phases = np.arange(0.0, len(proj) - span + 1e-6, 0.5)
    if len(phases) < 8:
        return 0.0
    scores = _comb_scores(proj, phases, period)
    typical = max(float(np.median(scores)),
                  settings.grid_typical_floor * float(scores.mean()))
    if typical <= 1e-6:
        return 0.0
    return float(scores.max() / typical)


def _lattice_score(px: np.ndarray, py: np.ndarray, period: float) -> float:
    """Lattice evidence on the weaker axis.

    The weaker axis decides, because a quad can sit square-on to one family of
    lines and be a cell out on the other, and averaging would hide exactly the
    failure that relabels every tile on the board.
    """
    return min(_periodicity(px, period), _periodicity(py, period))


def _sample(proj: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Strongest value within a pixel of each position, for sub-pixel tolerance."""
    idx = np.rint(positions).astype(np.int32)
    lo = np.clip(idx - 1, 0, len(proj) - 1)
    hi = np.clip(idx + 2, 1, len(proj))
    return np.array([proj[a:b].max() for a, b in zip(lo, hi)])


def _alignment(px: np.ndarray, py: np.ndarray, fx: float, fy: float, period: float) -> float:
    """Edge energy on the fitted grid lines, against energy mid-cell.

    Used to *rank* candidates that have already been accepted as boards, which is
    a different question from whether a board is present and needs a different
    measure. Periodicity answers "is there a 15-fold lattice here", and it
    saturates: two quads that both contain the board score in the hundreds, and
    which of them scores higher has more to do with how much phase range each
    one leaves than with which is better aligned. It picked a quad a cell out over
    one that was exact.

    This ratio does not saturate. It falls as the cut lines drift off the printed
    grid towards the middles of the cells, and half a cell out it inverts. So it
    ranks well and gates badly - a featureless quad puts nothing on either set of
    positions and can score anything at all. Hence both.
    """
    worst = 1e9
    for proj, phase in ((px, fx), (py, fy)):
        lines = _sample(proj, phase + _LINES * period)
        mids = _sample(proj, phase + (np.arange(15) + 0.5) * period)
        # The floor keeps a board with immaculate gutters from dividing by nothing.
        floor = 0.05 * float(proj.mean())
        worst = min(worst, float(lines.mean() / max(float(mids.mean()), floor, 1e-6)))
    return worst


def _comb_fit(
    px: np.ndarray, py: np.ndarray, nominal: float
) -> tuple[float, float, float]:
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
      The range is asymmetric for that reason: the playable area may be a good
      deal *narrower* than the candidate quad, since the quad often includes a
      printed border, but a lattice much *denser* than nominal is always
      spurious.

    A bounded per-axis refinement then recovers genuine small asymmetry, without
    enough freedom to jump a cell.
    """
    size = len(px)
    lo, hi = settings.grid_period_range

    def search(periods: np.ndarray, phase_step: float) -> tuple[float, float, float]:
        best = (nominal, 0.0, -1e18)
        for period in periods:
            span = 15.0 * period
            if span >= size:
                continue
            phases = np.arange(0.0, size - span + 1e-6, phase_step)
            if len(phases) == 0:
                continue
            scores = _comb_scores(px, phases, period) + _comb_scores(py, phases, period)
            i = int(np.argmax(scores))
            if scores[i] > best[2]:
                best = (float(period), float(phases[i]), float(scores[i]))
        return best

    # Coarse then fine, over both pitch and phase. A flat search fine enough to
    # place the pitch to a fraction of a pixel costs several times as much, and
    # every candidate quad pays for it; the response is smooth in pitch, so a
    # coarse pass locates the peak and a narrow fine pass around it lands on the
    # same answer.
    steps = settings.grid_period_steps
    coarse = np.linspace(lo * nominal, hi * nominal, steps)
    best_period, best_phase, _ = search(coarse, 1.0)

    reach = 2.0 * (hi - lo) * nominal / max(steps - 1, 1)
    best_period, best_phase, _ = search(
        np.linspace(best_period - reach, best_period + reach, 9), 0.5
    )

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


def _fit_grid(board: np.ndarray, nominal: float) -> GridFit:
    """Locate the 15x15 lattice in a warped board and say how sure we are.

    `nominal` is the cell pitch the candidate quad implies. It is passed in
    rather than derived from the image size because the image deliberately
    extends past the quad - see `_margined_homography`.
    """
    px, py = _edge_projections(board)
    fx, fy, period = _comb_fit(px, py, nominal)

    size = len(px)
    lo, hi = settings.grid_period_range
    # The lattice must sit near the middle of the frame. Without this it can
    # slide off into the margin, locking onto whatever lies beside the board.
    span = 15.0 * period
    centre_error = max(abs(fx + span / 2 - size / 2), abs(fy + span / 2 - size / 2))
    plausible = (
        lo - 1e-6 <= period / nominal <= hi + 1e-6
        and centre_error <= settings.grid_margin_frac * 15.0 * nominal
    )
    return GridFit(fx, fy, period, nominal,
                   _lattice_score(px, py, period),
                   _alignment(px, py, fx, fy, period), plausible)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def _homography(corners: np.ndarray, size: int, inset_frac: float) -> np.ndarray:
    e = inset_frac * size
    dst = np.array(
        [[-e, -e], [size + e, -e], [size + e, size + e], [-e, size + e]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(order_corners(corners), dst)


def _margined_homography(
    corners: np.ndarray, size: int, margin: float
) -> tuple[np.ndarray, float]:
    """Map a quad to a centred square inside a larger frame.

    Returns the transform and the quad's side length in the output, which is the
    nominal cell pitch times 15.

    The margin is what lets the grid lock work in both directions. A proposal
    that includes the printed border has to be shrunk, which fitting inside the
    quad alone can do; a proposal that has cropped a third of a cell off one
    side has to be *grown*, and there is no way to fit what was never warped in.
    Every detector here can be a little tight - a texture region shrinks towards
    wherever the tiles are densest - and a third of a cell is enough to push
    glyphs past the tolerance for being assigned to their own cell.
    """
    q = size / (1.0 + 2.0 * margin)
    e = margin * q
    dst = np.array([[e, e], [e + q, e], [e + q, e + q], [e, e + q]], dtype=np.float32)
    return cv2.getPerspectiveTransform(order_corners(corners), dst), q


def _apply(bgr: np.ndarray, m: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpPerspective(
        bgr, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _score_candidate(small: np.ndarray, quad: np.ndarray) -> tuple[float, float]:
    """(is-it-a-board, how-well-aligned) for one candidate quad.

    An implausible fit means the comb found something periodic that is not a 15x15
    grid at a credible pitch and position. Such a candidate is not a board,
    whatever else it scored.
    """
    size = settings.score_warp_size
    m, q = _margined_homography(quad, size, settings.grid_margin_frac)
    fit = _fit_grid(_apply(small, m, size), q / 15.0)
    if not fit.plausible:
        return 0.0, 0.0
    return fit.score, fit.alignment


def find_board(bgr: np.ndarray, corners: np.ndarray | None = None) -> Registration:
    """The single best guess at where the board is.

    Kept for callers that want one answer. `shortlist` is the better entry point
    when the caller can judge a candidate by reading it, because the score here is
    a weak predictor of how well a board will actually read - measured on 163
    photographs, quads scoring 3 sometimes read better than quads scoring 30.
    """
    return shortlist(bgr, corners, limit=1)[0]


def shortlist(
    bgr: np.ndarray, corners: np.ndarray | None = None, limit: int = 1,
) -> list[Registration]:
    """Up to `limit` plausible board positions, best guess first.

    Returning several is the point. The lattice score answers "is there a grid
    here", which is what rejects a turntable, and it is genuinely bad at the finer
    question of *which* near-miss quad is right: on the test corpus a quad scoring
    3.2 read 96% of its words correctly while the average quad scoring over 12 read
    84%. Anything that decides from this score alone will therefore discard boards
    it could have read.

    So the shortlist is deliberately generous and the caller is expected to choose
    by warping each one and seeing which yields a sensible board - a judgement made
    on the tiles themselves rather than on edge projections. See
    `pipeline._best_reading`.
    """
    if corners is not None:
        ordered = order_corners(corners)
        return [Registration(ordered, "supplied", 0.0, _aspect_skew(ordered))]

    h, w = bgr.shape[:2]
    scale = min(1.0, settings.detect_max_edge / float(max(h, w)))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else bgr
    sh, sw = small.shape[:2]
    frame_area = float(sh * sw)

    gradients = _gradients(small)

    # Snapping is cheap and scoring is not, and many proposals snap to the same
    # place - a dozen colour regions around one board converge on its edges. So
    # every variant is generated first and the duplicates dropped before any of
    # them is fitted.
    shapes: list[tuple[str, np.ndarray]] = []
    for name, quad in _candidates(small):
        for tag, shape in (
            (name, quad),
            (name + "+snap", _snap_quad(small, quad, gradients)),
            (name + "+snap!", _snap_quad(small, quad, gradients, keep_frac=1.0)),
        ):
            if shape is None:
                continue
            if _already_have(shape, shapes, small.shape[:2]):
                continue
            shapes.append((tag, shape))

    scored: list[tuple[float, float, str, np.ndarray]] = []
    for tag, shape in shapes:
        score, alignment = _score_candidate(small, shape)
        scored.append((score, alignment, tag, shape))

    # Two stages, because the two questions need different measures. Anything that
    # does not look like a board at all is dropped on periodicity; the survivors
    # are then ordered by how exactly their grid lines land, which is the finer
    # judgement and the one periodicity is too blunt to make.
    #
    # The second cut is relative as well as absolute. Alignment is only meaningful
    # between quads that both contain the board, and a quad containing nothing in
    # particular can post a high alignment by accident - so only candidates within
    # a fraction of the best periodicity are allowed into the ranking at all.
    boards = [c for c in scored if c[0] >= settings.min_grid_score]
    if boards:
        cut = settings.grid_score_relative * max(c[0] for c in boards)
        boards = [c for c in boards if c[0] >= cut]
    boards.sort(key=lambda c: -c[1])
    considered = [(tag, round(align, 3)) for _, align, tag, _ in boards]

    # Near-duplicates waste a caller's reading budget on the same board twice.
    picked: list[Registration] = []
    for score, alignment, name, quad in boards:
        if _already_have(quad, [(n.method, n.corners * scale) for n in picked],
                         small.shape[:2]):
            continue
        area_frac = float(cv2.contourArea(quad.astype(np.float32)) / frame_area)
        full = quad / scale                      # back to full-res coords
        picked.append(Registration(full, name, area_frac, _aspect_skew(full),
                                   score, alignment, considered))
        if len(picked) >= max(1, limit):
            break
    if picked:
        return picked

    best = ""
    if scored:
        top = max(scored, key=lambda c: c[0])
        best = f" Best candidate scored {top[0]:.2f} ({top[2]})."
    raise RegistrationError(
        "No board found: nothing in the frame contains a 15x15 grid." + best
        + " Try a less oblique angle, more even lighting, or pass explicit "
          "corners in the request."
    )


def warp_board(bgr: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, dict]:
    """Rectify the board so the 15x15 playable grid exactly fills the output.

    Two passes. The first warps the detected quad plus a margin of surrounding
    context. The second locks onto the printed grid lines found in that image,
    which is what turns an approximately-right quad into an exactly-right one -
    and the thing that silently shifts every cell if left uncorrected.
    """
    size = settings.warp_size
    margin = settings.grid_margin_frac
    m, q = _margined_homography(corners, size, margin)
    fit = _fit_grid(_apply(bgr, m, size), q / 15.0)

    # Refuse an implausible lock. Being a little misaligned costs a few
    # low-confidence cells; being a whole row out corrupts the entire board.
    if not fit.plausible:
        plain = _apply(bgr, _homography(corners, size, settings.grid_inset_frac), size)
        return plain, {
            "locked": False,
            "reason": "grid fit rejected as implausible; using the quad as given",
            "period_frac": round(fit.period_frac, 4),
            "grid_score": round(fit.score, 3),
        }

    s = size / (15.0 * fit.period)
    a = np.array([[s, 0.0, -fit.phase_x * s], [0.0, s, -fit.phase_y * s], [0.0, 0.0, 1.0]])
    fine = _apply(bgr, a @ m, size)

    return fine, {
        "locked": True,
        # The playable area as a fraction of the detected quad. Around 0.9 means
        # the quad was the board's outer edge; above 1.0 means it cropped in.
        "period_frac": round(fit.period_frac, 4),
        # How far inside the quad's edge the playable area started, in pixels of
        # the warp. Negative means the quad had cropped into the playable area.
        "grid_phase": (round(fit.phase_x - margin * q, 2),
                       round(fit.phase_y - margin * q, 2)),
        "grid_score": round(fit.score, 3),
        "grid_alignment": round(fit.alignment, 3),
    }
