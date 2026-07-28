"""All tunables in one place.

Every field can be overridden by an environment variable of the same name
(upper or lower case). On Dokploy, set them in the app's Environment tab.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ---- server ----------------------------------------------------------
    debug_dir: str = "/tmp/scrabble-debug"
    keep_debug_images: int = 50
    max_upload_bytes: int = 12 * 1024 * 1024

    # ---- board registration ---------------------------------------------
    # Longest edge the input is resized to before board detection. Detection
    # runs on the small image; the warp runs on the full-resolution original.
    detect_max_edge: int = 1000

    # Output size of the rectified board, in pixels. 720 => 48 px per cell.
    warp_size: int = 720

    # Fraction of the board width to crop inward after the *first* warp. Usually
    # leave at 0: the second pass locks onto the printed grid lines and works out
    # the true inset for itself. Only set this if grid-line detection fails.
    grid_inset_frac: float = 0.0

    # Pitch search range for the grid-lock pass, as a multiple of warp_size/15.
    # Keep it narrow. Widening it lets the fit lock onto tile edges and letters
    # instead of the printed grid, which shifts the whole board by a row.
    grid_period_range: tuple[float, float] = (0.86, 1.04)

    # A candidate quad must occupy at least this fraction of the frame.
    min_board_area_frac: float = 0.06
    # ...and be roughly square once rectified.
    max_board_aspect_skew: float = 1.35

    # Red-border detection (many tournament boards have one). Two hue bands
    # because red wraps around the HSV hue axis.
    red_hue_lo: int = 10
    red_hue_hi: int = 170
    red_sat_min: int = 70
    red_val_min: int = 50

    # ---- tile occupancy and glyph extraction ----------------------------
    # Divide out low-frequency brightness variation before analysis. Makes both
    # thresholding and blank detection immune to a lighting gradient across the
    # board, which is the normal condition on a real table.
    flatten_illumination: bool = True
    flatten_sigma_frac: float = 0.20     # blur sigma, as a fraction of board size
    flatten_grid: int = 64               # size the background estimate is computed at

    # Adaptive threshold used to isolate dark glyphs on light tiles.
    thresh_block: int = 25          # must be odd
    thresh_c: int = 10

    # A glyph contour's bounding box, as a fraction of one cell's side.
    # The lower bound is the main thing rejecting the small point-value digit
    # printed in the tile corner: letters are ~0.4-0.6 of a cell, digits ~0.3.
    glyph_min_side_frac: float = 0.30
    glyph_max_side_frac: float = 0.85
    # ...and its area as a fraction of the cell area.
    glyph_min_area_frac: float = 0.025
    glyph_max_area_frac: float = 0.400
    # A blob wider or taller than this has probably merged with the tile's
    # point-value digit, so it gets an erode-and-split pass.
    glyph_split_side_frac: float = 0.55
    # In that pass, a fragment is kept if it is at least this fraction of the
    # largest fragment's area...
    glyph_keep_frac: float = 0.15
    # ...and its centre is within this many cell-widths of the largest
    # fragment's centre. This second test is the one that actually works: a
    # letter's strokes cluster near its own centre, the corner digit does not.
    glyph_merge_radius_frac: float = 0.30

    # ink area / bounding-box area. Rejects thin premium-square lettering.
    glyph_min_extent: float = 0.20
    # h/w or w/h, whichever is larger. Must stay above ~3 to admit 'I'.
    glyph_max_aspect: float = 4.0

    # How far a glyph centroid may sit from its cell centre, as a fraction of
    # the cell side. Tolerates tiles placed slightly askew.
    max_glyph_shift_frac: float = 0.34

    # Tile faces are unsaturated cream; uncovered premium squares are not.
    # Set to 255 to disable this gate.
    max_tile_saturation: int = 105

    # ---- blank tile detection -------------------------------------------
    # A blank tile carries no ink, so there is nothing to classify; it can only
    # be told from bare board by the colour of its face. Measured on test
    # boards, brightness does not separate them at all (blank 242 vs board
    # 238 +/- 6). Saturation does - a cream face reads ~50 against ~10 for plain
    # board - but premium squares reach 65, so saturation needs a hue gate with
    # it. That combination is clean on test boards, but the constants below
    # describe one board and one tile set. Re-measure on yours.
    detect_blanks: bool = True

    # A blank tile face is smooth: low coefficient of variance.
    blank_max_cov: float = 0.055
    # Saturation must exceed the local plain-board level by this much.
    blank_min_sat_delta: float = 20.0
    # ...and the hue must look like a tile face rather than a premium square.
    # OpenCV hue is 0-180. Cream sits around 15-25; premium red is 0-10 or
    # 170-180 and premium blue is 100-115.
    blank_hue_range: tuple[float, float] = (8.0, 34.0)
    # Cells at or below this saturation form the plain-board reference set.
    blank_plain_max_sat: float = 30.0
    blank_neighbourhood: int = 4         # cells, Chebyshev radius
    blank_min_neighbours: int = 6

    # ---- classifier ------------------------------------------------------
    glyph_size: int = 32            # normalised glyph is glyph_size square
    glyph_pad: int = 3
    # A preference, not a requirement. If it cannot be opened, a per-platform
    # candidate list is tried (see classify._FONT_CANDIDATES), so the server
    # starts on Windows and macOS as well as in the container.
    template_font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # An unwritable path here is not fatal; templates are rebuilt on cold start.
    template_cache: str = "/tmp/scrabble-templates.npz"
    match_shift: int = 2            # search +/- this many px when matching
    # Below this best-IoU the cell is reported as an unknown occupied tile
    # rather than a guessed letter.
    min_match_iou: float = 0.35
    # Softmax temperature over template scores. Lower = more confident.
    conf_temperature: float = 0.05
    # Cells below this confidence are listed in `low_confidence` for the
    # operator UI to resolve.
    low_conf_threshold: float = 0.60


settings = Settings()
