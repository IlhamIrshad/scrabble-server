from __future__ import annotations

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
    point_value: str | None = Field(
        None,
        description=(
            "The point value read off the tile, when it was legible enough to be "
            "believed. Null means the letter was decided on its shape alone."
        ),
    )
    notes: list[str] = Field(default_factory=list)

    @property
    def coord(self) -> str:
        """Standard Scrabble notation, e.g. H8."""
        return f"{chr(ord('A') + self.row)}{self.col + 1}"


class RegistrationInfo(BaseModel):
    method: str = Field(
        description=(
            "Which detector proposed the winning quad, with '+snap' if its edges "
            "were then fitted to the board's boundary: 'colour[n]' a region of "
            "uniform colour, 'texture' a region of high local detail, 'edges' a "
            "contour on Canny edges, or 'supplied' for caller-given corners."
        )
    )
    corners: list[tuple[float, float]]
    area_frac: float
    aspect_skew: float
    grid_score: float = Field(
        0.0,
        description=(
            "Whether a 15x15 lattice is present in the chosen quad at all, as a "
            "multiple of chance. Around 1.0 means none was found; boards score "
            "tens to thousands."
        ),
    )
    grid_alignment: float = Field(
        0.0,
        description=(
            "Edge energy on the grid lines over energy mid-cell. Large is good; "
            "below 1.0 means the cut lines are nearer the middles of the cells "
            "than the printed grid, so every cell is likely half a cell out."
        ),
    )
    considered: list[tuple[str, float]] = Field(
        default_factory=list,
        description=(
            "Every candidate quad accepted as a board, as (detector, alignment), "
            "best first. The first thing to look at if the wrong quad was picked."
        ),
    )
    grid: dict = Field(
        default_factory=dict,
        description="Grid-lock result: where the playable area sat inside the quad.",
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
