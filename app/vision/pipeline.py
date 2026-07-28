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


def detect(
    image: np.ndarray,
    corners: np.ndarray | None = None,
    want_debug: bool = False,
) -> DetectResponse:
    timing: dict[str, float] = {}
    t0 = time.perf_counter()

    reg = registration.find_board(image, corners)
    timing["registration"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    board, grid_info = registration.warp_board(image, reg.corners)
    timing["warp"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    cells, mask = grid.extract_cells(board)
    timing["segment"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    bank = classify.bank()
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
            m = bank.match(cell.glyph)
            result = CellResult(
                row=cell.row, col=cell.col, letter=m.letter, blank=False,
                confidence=round(m.confidence, 4), best_iou=round(m.best_iou, 4),
                candidates=[(ch, round(s, 4)) for ch, s in m.ranked],
                notes=list(cell.notes),
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
