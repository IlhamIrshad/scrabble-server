from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EMPTY = "."
BLANK = "?"       # a blank tile: occupied, letter not determinable from pixels
UNKNOWN = "*"     # a tile is there, but no template matched well enough


class CellResult(BaseModel):
    row: int = Field(ge=0, le=14)
    col: int = Field(ge=0, le=14)
    letter: str | None = Field(None, description="A-Z, or null for blank/unknown")
    blank: bool = False
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    best_iou: float = 0.0
    candidates: list[tuple[str, float]] = Field(
        default_factory=list,
        description="Top template matches, descending. Feed these to the operator UI.",
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def coord(self) -> str:
        """Standard Scrabble notation, e.g. H8."""
        return f"{chr(ord('A') + self.row)}{self.col + 1}"


class RegistrationInfo(BaseModel):
    method: Literal["red_border", "quad", "supplied"]
    corners: list[tuple[float, float]]
    area_frac: float
    aspect_skew: float
    grid: dict = Field(
        default_factory=dict,
        description="Grid-lock result: phase, period and scale correction per axis.",
    )


class DetectResponse(BaseModel):
    ok: bool = True
    board: list[str] = Field(
        description=(
            "15 rows of 15 characters. '.' empty, 'A'-'Z' a letter tile, "
            "'?' a blank tile, '*' a tile whose letter could not be read."
        )
    )
    tiles: list[CellResult] = Field(description="Occupied cells only.")
    low_confidence: list[CellResult] = Field(
        description="Cells needing operator confirmation."
    )
    tile_count: int = 0
    registration: RegistrationInfo
    timing_ms: dict[str, float] = Field(default_factory=dict)
    debug_image: str | None = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    stage: str
