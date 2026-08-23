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
        """This square in Scrabble notation, e.g. H8 for the centre.

        Columns are lettered A-O left to right and rows numbered 1-15 top to
        bottom, so the letter comes from the *column* and the number from the row.

        This was the other way round until it was checked against a board-UI
        specification. The two conventions agree on exactly one square - the centre,
        H8, which is symmetric - so the error survived every casual glance at it, and
        would have transposed every coordinate written into a game record.
        """
        return f"{chr(ord('A') + self.col)}{self.row + 1}"


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
    confident: bool = Field(
        True,
        description=(
            "Whether the board's geometry looks sound. Calibrated against 160 "
            "photographs judged by eye: when this is false the board really was "
            "misregistered in all but one case in sixty, so treat it as a reason "
            "to re-take the photograph. When true it means only that nothing "
            "looks wrong - about a quarter of boards marked confident are still "
            "misregistered - so it is not a guarantee that the board is correct."
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
