"""image bytes -> board state. One entry point: `detect`."""

from __future__ import annotations

import os
import time
import uuid

import cv2
import numpy as np

from ..config import settings
from ..schemas import (
    BLANK,
    EMPTY,
    UNKNOWN,
    CellResult,
    DetectResponse,
    RegistrationInfo,
)
from . import classify, grid, registration

N = grid.N


class DecodeError(ValueError):
    pass


def decode(data: bytes) -> np.ndarray:
    if not data:
        raise DecodeError("Empty request body.")
    if len(data) > settings.max_upload_bytes:
        raise DecodeError(
            f"Image is {len(data)} bytes; limit is {settings.max_upload_bytes}."
        )
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError("Could not decode image. Send JPEG or PNG bytes.")
    return img


def _reading_quality(cells: list[grid.Cell], bank) -> float:
    """How much a warp looks like a board once its cells have been read.

    Counts tiles whose glyph matched a letter well, so it rewards a warp that puts
    whole glyphs inside cells and punishes one that slices them across boundaries.

    This exists because the geometric score cannot make this judgement. Measured
    over 163 photographs, a quad scoring 3.2 on lattice strength read 96% of its
    words correctly while quads scoring above 12 averaged 84% - and a
    misregistered board gives itself away far more clearly here, losing more than
    half its tiles (a median of 40 against 94) because the glyphs fall between
    cells and are discarded.

    Scored on how well each glyph overlaps its best template, *not* on the
    classifier's confidence. Confidence is a margin between letters, so rewarding
    it favours a warp that reads wrongly but decisively over one that reads
    correctly but hedges - which is the one trade this pipeline must never make.
    Overlap instead measures whether a whole, undamaged glyph landed inside a
    cell, which is exactly what a correct warp produces.
    """
    found = 0.0
    for cell in cells:
        if not cell.occupied:
            continue
        if cell.is_blank or cell.glyph is None:
            found += 0.5                    # a tile, but nothing to corroborate it
            continue
        m = bank.match(cell.glyph)
        if m.letter is not None:
            # One per readable tile, with overlap only as a tiebreak. Both parts
            # were arrived at by measurement. Dropping the constant and scoring
            # overlap alone moved five boards away from answers a person had
            # confirmed, because it let a warp with fewer but crisper glyphs beat
            # one that simply read more of the board. Using the classifier's
            # *confidence* as the tiebreak instead was worse in a different way: it
            # rewards reading decisively over reading correctly, and a test caught
            # it preferring a warp that misread a Q as an L with total conviction.
            found += 1.0 + m.best_iou
    return found


def _wins(quality: float, off_pitch: float, best: float, best_off: float) -> bool:
    """Whether a candidate beats the best so far, on reading quality then pitch.

    Reading quality decides it, except when the two are close enough that the
    difference is noise. Measured on a photograph a person had confirmed: the right
    quad read 173.5 and a quad two cells out read 173.7 - a tenth of a percent apart,
    and the wrong one won. Everything else about them differed clearly, so a
    tiebreak was all that was missing.

    The tiebreak is how far the fitted cell pitch is from the one the quad implies,
    because that answers a question reading quality structurally cannot: a lattice
    is periodic, so a quad cropped a little way into the playable area still puts
    whole glyphs in cells, just the wrong cells. The loser above was 7% in; the
    winner was within a tenth of a percent.
    """
    if quality > best * (1.0 + settings.reading_tie_frac):
        return True
    if quality < best * (1.0 - settings.reading_tie_frac):
        return False
    return off_pitch < best_off


def _best_reading(
    image: np.ndarray, options: list[registration.Registration], bank,
) -> tuple[registration.Registration, np.ndarray, dict, list[grid.Cell], np.ndarray]:
    """Warp and read every shortlisted board position, and keep the best one.

    Judged on two things that answer different questions. Reading quality says
    whether glyphs land whole and centred, which settles alignment to within a
    fraction of a cell but is *completely blind* to being a whole cell out - a
    shifted lattice centres every glyph just as well, in the wrong cell. Measured
    on confirmed boards, shifting by one cell costs reading quality about 3%.

    The premium-square layout answers the question reading quality cannot, because
    it is the one thing on a board that does not repeat. The same one-cell shift
    takes it from 0.70 to 0.16 - below the 0.27 that random cells would score,
    since a displaced layout is actively anti-correlated with the true one.

    They are multiplied rather than added so that neither can be traded away: a
    candidate must both read cleanly *and* sit where the printed board says it
    does.
    """
    best = None
    for reg in options:
        board, info = registration.warp_board(image, reg.corners)
        cells, mask = grid.extract_cells(board)
        quality = _reading_quality(cells, bank)
        agreement, visible = grid.premium_agreement(cells)
        # A veto, not a weight. Using the agreement as a multiplier was tried and
        # measured worse: it re-ranks every candidate, so on the boards where few
        # premium squares are visible its noise overrode choices a person had
        # already confirmed correct - eleven of seventy-five moved. Demoting only
        # the candidates that are *clearly* anti-correlated leaves every other
        # ranking exactly as it was, and those are the shifted ones: a board a whole
        # cell out scores around 0.15 where chance alone is 0.27.
        shifted = (visible >= settings.premium_min_visible
                   and agreement < settings.premium_reject_below)
        combined = quality * (settings.premium_floor if shifted else 1.0)
        # How far the fitted cell pitch is from the pitch this quad implies. Only
        # consulted to separate candidates that read equally well, where it says
        # which one actually bounds the playable area rather than cutting into it.
        off_pitch = abs(info.get("period_frac", 1.0) - 1.0)
        if best is None or _wins(combined, off_pitch, best[0], best[1]):
            best = (combined, off_pitch, quality, agreement, visible,
                    reg, board, info, cells, mask)

    _, _, quality, agreement, visible, reg, board, info, cells, mask = best
    info = {**info, "reading_quality": round(quality, 1),
            "premium_match": round(agreement, 2), "premium_visible": visible,
            "candidates_read": len(options)}
    return reg, board, info, cells, mask


def detect(
    image: np.ndarray,
    corners: np.ndarray | None = None,
    want_debug: bool = False,
) -> DetectResponse:
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    bank = classify.bank()
    options = registration.shortlist(
        image, corners, limit=settings.registration_candidates
    )
    timing["registration"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    reg, board, grid_info, cells, mask = _best_reading(image, options, bank)
    # One figure, because choosing between board positions means warping and
    # segmenting each of them; they are no longer separable stages.
    timing["warp_and_read"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    value_bank = classify.values()
    rows = [[EMPTY] * N for _ in range(N)]
    tiles: list[CellResult] = []
    low: list[CellResult] = []

    for cell in cells:
        if not cell.occupied:
            continue

        if cell.is_blank or cell.glyph is None:
            result = CellResult(
                row=cell.row, col=cell.col, letter=None, blank=True,
                confidence=1.0 if cell.is_blank else 0.0, notes=list(cell.notes),
            )
            rows[cell.row][cell.col] = BLANK
        else:
            # The tile's own printed point value, as a nudge towards the letters
            # that carry it. Weak by construction: it is added to the overlap
            # scores, so it settles a pair the shapes cannot and cannot outvote a
            # confident read.
            read = classify.read_value(value_bank, cell.value_glyph)
            bonus = classify.value_bonus(bank.letters, read.letter, read.best_iou)
            m = bank.match(cell.glyph, bonus)

            notes = list(cell.notes)
            if bonus is not None:
                notes.append(f"point value read as {read.letter}"
                             f" (overlap {read.best_iou:.2f})")
            result = CellResult(
                row=cell.row, col=cell.col, letter=m.letter, blank=False,
                confidence=round(m.confidence, 4), best_iou=round(m.best_iou, 4),
                candidates=[(ch, round(s, 4)) for ch, s in m.ranked],
                point_value=read.letter if bonus is not None else None,
                notes=notes,
            )
            rows[cell.row][cell.col] = m.letter or UNKNOWN

        tiles.append(result)
        if result.letter is None or result.confidence < settings.low_conf_threshold:
            low.append(result)

    timing["classify"] = (time.perf_counter() - t) * 1000

    debug_name = None
    if want_debug:
        t = time.perf_counter()
        debug_name = _write_debug(image, reg, board, mask, rows)
        timing["debug"] = (time.perf_counter() - t) * 1000

    timing["total"] = (time.perf_counter() - t0) * 1000

    return DetectResponse(
        ok=True,
        board=["".join(r) for r in rows],
        tiles=tiles,
        low_confidence=low,
        tile_count=len(tiles),
        registration=RegistrationInfo(
            method=reg.method,
            corners=[(float(x), float(y)) for x, y in reg.corners],
            area_frac=round(reg.area_frac, 4),
            aspect_skew=round(reg.aspect_skew, 4),
            grid_score=round(reg.grid_score, 3),
            grid_alignment=round(reg.grid_alignment, 3),
            confident=reg.confident,
            considered=reg.considered,
            grid=grid_info,
        ),
        timing_ms={k: round(v, 2) for k, v in timing.items()},
        debug_image=f"/debug/{debug_name}" if debug_name else None,
    )


# ---------------------------------------------------------------------------
# debug rendering
# ---------------------------------------------------------------------------


def _annotate_source(image: np.ndarray, reg: registration.Registration) -> np.ndarray:
    out = image.copy()
    pts = reg.corners.astype(np.int32)
    cv2.polylines(out, [pts], True, (0, 255, 0), max(2, image.shape[1] // 400))
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), max(4, image.shape[1] // 180), (0, 0, 255), -1)
        cv2.putText(out, "TL TR BR BL".split()[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, image.shape[1] / 1600, (0, 0, 255), 2)
    cv2.putText(out, f"method={reg.method}", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return out


def _annotate_board(board: np.ndarray, rows: list[list[str]]) -> np.ndarray:
    out = board.copy()
    size = out.shape[0]
    step = size / N
    for i in range(N + 1):
        p = int(round(i * step))
        cv2.line(out, (p, 0), (p, size), (60, 60, 60), 1)
        cv2.line(out, (0, p), (size, p), (60, 60, 60), 1)
    for r in range(N):
        for c in range(N):
            ch = rows[r][c]
            if ch == EMPTY:
                continue
            colour = (0, 200, 0) if ch.isalpha() else (0, 140, 255)
            org = (int(c * step + step * 0.22), int((r + 1) * step - step * 0.22))
            cv2.putText(out, ch, org, cv2.FONT_HERSHEY_SIMPLEX,
                        step / 44.0, colour, 2, cv2.LINE_AA)
    return out


def _write_debug(image, reg, board, mask, rows) -> str:
    os.makedirs(settings.debug_dir, exist_ok=True)
    _prune_debug()

    panels = [
        _annotate_source(image, reg),
        _annotate_board(board, rows),
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
    ]
    h = 720
    resized = []
    for p in panels:
        scale = h / p.shape[0]
        resized.append(cv2.resize(p, (int(p.shape[1] * scale), h)))
    strip = np.hstack(resized)

    name = f"{uuid.uuid4().hex[:12]}.jpg"
    cv2.imwrite(os.path.join(settings.debug_dir, name), strip,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return name


def _prune_debug() -> None:
    try:
        entries = [
            os.path.join(settings.debug_dir, f)
            for f in os.listdir(settings.debug_dir)
            if f.endswith(".jpg")
        ]
    except OSError:
        return
    if len(entries) <= settings.keep_debug_images:
        return
    entries.sort(key=lambda p: os.path.getmtime(p))
    for path in entries[: len(entries) - settings.keep_debug_images]:
        try:
            os.remove(path)
        except OSError:
            pass
