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

    # Output size of the rectified board, in pixels. 1080 => 72 px per cell.
    #
    # Raised from 720 once real photographs arrived: they are about 145 px per cell
    # at the source, so warping to 48 discarded two thirds of the detail that had
    # been captured. Measured over 60 of them, words read correctly went from 79.4%
    # to 86.5% for 19% more time, and impossible letter counts fell from 55 boards
    # to 48. Going on to 96 px per cell gained 0.4 of a point for another 14% - so
    # this is where the curve flattens, not a cap.
    warp_size: int = 1080

    # Size candidates are trial-warped to when being scored. Only needs to be big
    # enough to resolve grid lines, and every candidate pays for it. 360 gives 24
    # pixels per cell against a line width of two or three; measured against 480
    # and 300, accuracy is the same across all three.
    score_warp_size: int = 360

    # How much context around the detected quad the grid-lock pass gets to see,
    # as a fraction of the quad's width. This is what lets the lock correct a
    # quad that cropped *into* the playable area, not just one that included the
    # border. It also bounds how far the lattice may drift from the quad.
    grid_margin_frac: float = 0.10

    # Fraction of the board width to crop inward, used only when the grid lock
    # is rejected and the quad has to be trusted as given.
    grid_inset_frac: float = 0.0

    # Pitch search range for the grid-lock pass, as a multiple of the pitch the
    # detected quad implies. Asymmetric on purpose: the playable area is often a
    # good deal narrower than the quad, because the quad tends to include the
    # printed border, but a lattice much denser than nominal is always spurious
    # - it means the fit found tile edges or letters instead of the grid, which
    # shifts the whole board by a row.
    grid_period_range: tuple[float, float] = (0.86, 1.10)

    # Pitches tried in the coarse pass of the grid fit, before a narrow fine pass
    # around the best of them. Every candidate quad is fitted, so this is the
    # single biggest cost in detection.
    grid_period_steps: int = 40

    # A candidate quad must occupy this fraction of the frame...
    min_board_area_frac: float = 0.06
    max_board_area_frac: float = 0.95
    # ...and be roughly square once rectified.
    max_board_aspect_skew: float = 1.35

    # How strongly a candidate must show a 15x15 lattice to be accepted, as a
    # multiple of chance. This is the test that rejects a red turntable, a wooden
    # table or the photograph's own border, none of which a shape or colour test
    # can rule out.
    #
    # Measured on the test photographs and the synthetic set, a quad that holds
    # the board scores 25 to several thousand and a quad that does not scores
    # under 10, so anything in between separates them with a wide margin either
    # way. Setting it low does not degrade gracefully: a near-miss quad is
    # accepted, its grid locks onto the wrong lattice, and every cell is reported
    # one row out - which is worse than reporting no board at all.
    # Lowered from 12 to a permissive floor, because measurement on 163 real
    # photographs showed this score is a weak predictor of whether a board will
    # read: quads scoring 3 sometimes read better than quads scoring 30, and a
    # gate at 12 threw away 13 boards that read at 75% of their words or better.
    # It is now a sanity check that rejects a turntable, not the decision - the
    # decision is made by reading each shortlisted quad. See
    # `registration_candidates` and `pipeline._best_reading`.
    min_grid_score: float = 1.5

    # How many shortlisted board positions to warp and actually read before
    # choosing. Each one costs a warp, a segmentation and a classification pass,
    # so this trades time for the thing the geometric score cannot judge. Measured
    # over 163 photographs: 3 candidates give 107 boards read with 70+ tiles at
    # about 1.8s each, 6 give 111 at about 2.7s. Turn it down for the WebSocket
    # path, where frames are cheap and latency is not.
    registration_candidates: int = 6
    # Probe for a whole-cell shift when the chosen board agrees with the printed
    # premium layout less well than this. Set between the two populations it has to
    # separate, measured over 161 photographs a person judged: boards registered
    # correctly agree at 0.62 to 0.97, and the two that were a cell out sat at 0.06
    # and 0.16. Nothing correct comes near this bar, so the probe is free on them.
    shift_probe_below: float = 0.50
    # A shifted board is only adopted if it agrees at least this well and improves on
    # the original by at least this much. Both, because agreeing merely less badly
    # means the board has been nudged rather than found, and the original at least
    # came from a detector. The two recovered boards go to 0.94 and 0.78.
    shift_adopt_above: float = 0.50
    shift_adopt_margin: float = 0.25

    # Two candidates whose reading qualities are within this fraction of each other
    # are treated as reading equally well, and separated on cell pitch instead. One
    # percent, because that is comfortably below the difference a real misregistration
    # makes - a board a whole cell out loses about 3% - and comfortably above the
    # tenth of a percent that let a quad two cells out win on a confirmed board.
    reading_tie_frac: float = 0.01

    # Candidates below this fraction of the best periodicity are not ranked
    # against it on alignment, however well they happen to align.
    grid_score_relative: float = 0.25

    # Above both of these, a registration is reported as confident. Calibrated
    # against 160 photographs whose registration a person judged by eye: this bar
    # marks 60 of them doubtful, and only one of those 60 was in fact correct,
    # while catching 59 of the 85 that were wrong.
    #
    # It is advisory and deliberately not a rejection. Twenty-six wrong boards
    # still clear it, so "confident" means "nothing looks wrong", never "this is
    # right"; and being over-eager to reject costs a board that would have read
    # perfectly. Both numbers describe the *geometry* only - tile count separates
    # these boards better still, and is not used, because every photograph in that
    # corpus was an end-of-game board and a rule keyed on tile count would call a
    # legitimate opening position misregistered.
    confident_grid_score: float = 3.2
    confident_grid_alignment: float = 6.8

    # Floor on the reference level in the periodicity measure, as a fraction of
    # the mean comb response. A perfectly sharp grid has a zero median response,
    # and this is what stops that dividing by nothing. It bounds the score at
    # roughly 1/this, so it must stay well below the good/bad separation.
    grid_typical_floor: float = 0.05

    # Edge projections are clipped at this quantile before the grid is fitted,
    # so that one very strong edge cannot outweigh sixteen ordinary grid lines.
    grid_line_clip_quantile: float = 0.90

    # How many alternative four-sided readings of one contour to offer. At 1 only
    # polygon-approximation-or-bounding-rectangle is used, which cannot express a
    # keystone; at 3 the two shapes that can are added as well.
    #
    # Left at 1, because the keystone shapes still measure worse even with the
    # premium-layout veto in place: scored against 75 boards a person confirmed,
    # 3 gives 57 within a third of a cell against 71 at 1. The veto stops a shifted
    # lattice winning, but these extra shapes are evidently wrong in ways that are
    # not shifts - most likely genuinely poor quads that happen to read well. Worth
    # revisiting only with a way to tell a good keystone from a bad one.
    quad_shapes_per_contour: int = 1

    # Line proposals: quads built by intersecting the board's four boundary lines
    # instead of tracing a region's outline. A segment must be this long, relative
    # to the frame's longest edge, to be considered a board edge at all - long
    # enough to exclude tile and cell edges, which is what makes the two families
    # findable, and short enough to survive an edge broken by a hand or a rack.
    line_min_length_frac: float = 0.22
    # How wide a break in an edge to step over, again relative to the longest edge.
    # This one matters more than it looks. The far edge of a tilted board is the
    # hardest of the four - foreshortened, lowest contrast, and interrupted by tiles
    # and racks sitting against it - and at 2% it was missed entirely on a board at
    # 28 degrees, leaving three sides and no quad. At 5% it is found.
    line_max_gap_frac: float = 0.05
    # Hough votes a line needs. Low, because the gate that matters is length, and
    # requiring both is how the far edge gets lost.
    line_votes: int = 40
    # How far from the dominant orientation a segment may lie and still belong to
    # its family, in degrees. Wide on purpose, and it has to be: a keystone means
    # opposite edges are genuinely not parallel, and on a board photographed at 28
    # degrees the two side edges sit 22 degrees apart. At 12 degrees they landed in
    # different families and the board was rejected for having only one side.
    # Anything above 45 would be ambiguous between the two families, so this is
    # nearly as loose as the idea allows; clutter it lets in is dealt with by
    # grouping and by ranking on how much of each edge was actually seen.
    line_angle_tolerance: float = 35.0
    # Segments within this fraction of the frame's longest edge of each other, along
    # their shared normal, are treated as fragments of one physical edge. About a
    # third of a cell, so the two sides of a board's printed border stay separate.
    line_group_gap_frac: float = 0.012
    # How many candidate edges to offer per side, outermost first. Four sides means
    # this to the fourth power combinations, so it is deliberately small.
    line_groups_per_side: int = 3
    # Quads offered per photograph, best-supported first.
    line_quad_limit: int = 3

    # Colour-region proposals: the frame is clustered into this many colours, once
    # per value, and the thumbnail size clustering runs at. Several values because
    # the right number is not knowable in advance - too few and the board face
    # merges with the table, too many and it splits into lit and shadowed halves,
    # and a board that is a clean cluster at one k can be absent at the next. More
    # values means more candidates to score, so more time.
    detect_colour_clusters: tuple[int, ...] = (4, 6, 9)
    detect_cluster_edge: int = 160
    # Fixed so that the same photograph always yields the same candidates. Change
    # it only to check that a result is not an artefact of this particular value.
    detect_rng_seed: int = 0

    # Two candidate quads within this fraction of the frame's longest edge of each
    # other count as the same one. Fitting is much dearer than proposing, and a
    # dozen colour regions around one board snap to the same four edges.
    candidate_dedupe_frac: float = 0.006

    # Texture proposals: gradient energy is blurred by this fraction of the
    # frame's short side, which should be around half a cell so that the
    # playable area fuses into one region.
    texture_sigma_frac: float = 0.025
    # The second texture reading calls a region busy when its local variation is
    # this many times that of the frame's quietest areas, taken at this quantile.
    # A bare table varies by a couple of grey levels, so a small multiple of that
    # still admits a board's empty squares.
    texture_floor_quantile: float = 0.10
    texture_floor_multiple: float = 3.0

    # Edge snapping. Each proposal's four edges are slid onto the strongest
    # straight boundary near them, coarse to fine, as a fraction of the board's
    # side. The first radius must be large enough to cover how wrong a region
    # proposal's shape can be - a texture region can be a tenth of the board out
    # on the side where fewest tiles have been played.
    snap_radius_fracs: tuple[float, ...] = (0.12, 0.06, 0.03)
    snap_offsets: int = 9            # positions tried per endpoint, per pass
    snap_samples: int = 256          # points sampled along a candidate line
    # Among lines within this fraction of the best score, the outermost wins.
    snap_keep_frac: float = 0.85

    # ---- tile occupancy and glyph extraction ----------------------------
    # Divide out low-frequency brightness variation before analysis. Makes both
    # thresholding and blank detection immune to a lighting gradient across the
    # board, which is the normal condition on a real table.
    flatten_illumination: bool = True
    flatten_sigma_frac: float = 0.20     # blur sigma, as a fraction of board size
    flatten_grid: int = 64               # size the background estimate is computed at

    # Adaptive threshold used to isolate ink from its tile. Applied twice, once
    # per polarity, so it finds dark letters on cream tiles and light letters on
    # dark tiles without being told which to expect.
    # Window size for that threshold, as a fraction of a cell. Was 25 pixels flat,
    # which meant the same number described half a letter at one warp size and a
    # whole one at another; 0.52 reproduces the old 25px at the default warp.
    thresh_block_frac: float = 0.52
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
    # point-value digit, so it gets an erode-and-split pass...
    glyph_split_side_frac: float = 0.55
    # ...eroding up to this many times, since how thick the bridge between letter
    # and digit is depends on the photograph and one pass often leaves it intact.
    glyph_split_rounds: int = 2
    # In that pass, a fragment is kept if it is at least this fraction of the
    # largest fragment's area...
    glyph_keep_frac: float = 0.15
    # ...and its centre is within this many cell-widths of the cell's centre. This
    # second test is the one that actually works: a letter is centred on its tile,
    # the point-value digit sits out in a corner. Measured over the synthetic set
    # anything from 0.30 to 0.42 scores the same, so this sits inside the flat
    # region and towards the cautious end of it - a letter's furthest stroke is
    # about 0.3 of a cell from the centre and the digit about 0.4.
    glyph_merge_radius_frac: float = 0.32

    # ---- the tile's printed point value ---------------------------------
    # A tile states its letter twice, and the number separates exactly the pairs
    # overlap struggles with: Q is 10 against O's 1, D is 2, C is 3, S is 1 against
    # B's 3. Read where legible and added as weak evidence to the letters that carry
    # that value - added, never used to filter, so a misread digit can only overturn
    # a near-tie.
    #
    # Off, because at the resolution these photographs give it is a coin flip, and a
    # coin flip is worse than nothing. Built, measured, and left in place because it
    # becomes right with more pixels on the board - it is not a dead end, it is
    # waiting on the camera. Measured over the synthetic set, digit references taken
    # from the same tile artwork as the letters:
    #
    #   a digit-sized mark found on    37% of tiles
    #   best precision reading it      54% (at value_min_iou 0.70, 13% of tiles);
    #                                  nothing at all qualifies above 0.70
    #   words read on real photos      91.0% off, 90.6% on, 89.6% at triple the
    #                                  bonus - it gets worse the more it is trusted
    #
    # The cause is size, not method: the value is about eight pixels tall here, and
    # every digit at eight pixels is a blob that matches '8' best because '8' has the
    # most ink. At roughly three times the pixels on the board - a 2500-pixel-wide
    # photograph rather than 960 - the digit is around 28 pixels and this becomes a
    # strong signal rather than a weak one. Turn it on then, and re-measure with
    # scripts against the synthetic set before trusting it.
    use_point_value: bool = False
    # Size of a digit-sized mark, as a fraction of a cell's side and area.
    value_min_side_frac: float = 0.08
    value_max_side_frac: float = 0.30
    value_min_area_frac: float = 0.002
    # How well the marks must match a reference digit before they are believed...
    value_min_iou: float = 0.55
    # ...and how much overlap agreeing with the value is worth. Letters are told
    # apart by a few percent of overlap, so this is the scale that settles a tie
    # without being able to override a clear read.
    value_agreement_bonus: float = 0.05

    # ink area / bounding-box area. Rejects thin premium-square lettering.
    glyph_min_extent: float = 0.20
    # h/w or w/h, whichever is larger. A bold sans 'I' is a bare stroke about six
    # times taller than it is wide, so anything under that silently drops every
    # 'I' on the board - which looks like a tile that was never placed rather
    # than a misread, and so is easy to miss. What keeps a stray grid-line
    # fragment out is not this bound but the distance from the cell centre: a
    # line lies along a cell edge, half a cell from where a letter would be.
    glyph_max_aspect: float = 7.0

    # How far a glyph centroid may sit from its cell centre, as a fraction of
    # the cell side. Tolerates tiles placed slightly askew.
    max_glyph_shift_frac: float = 0.34

    # ---- tile face colour -----------------------------------------------
    # The tile face colour is not configured: it is measured from the cells that
    # hold letters, because every tile in a set shares one face colour whatever
    # colour that is. These two govern that measurement.
    #
    # How far a cell's face may sit from the measured tile colour and still count
    # as a tile, in approximate CIELAB Delta-E. About 2-3 is just visible, 10 is
    # obvious, so 22 is loose enough for shadow and white balance across one board
    # and tight enough to separate any two colours a manufacturer would choose for
    # tiles and board. Raise it if tiles in shadow are being missed; lower it if
    # premium-square lettering is being read as tiles.
    tile_face_delta_e: float = 22.0
    # Cells that must agree before a measured colour is trusted at all. Below
    # this, no colour gate is applied and glyphs are accepted on shape alone -
    # which is the right answer for a nearly empty board.
    tile_cluster_min_members: int = 6
    # Fraction of a cell trimmed from each side before its face is measured. Big
    # enough to keep the neighbouring cell's grid line and the tile's bevel out,
    # small enough that a letter does not fill what is left - at 0.22 the window
    # was the middle half of the cell, which on a letter-filled tile is almost all
    # ink, and the measured face colour came back as the ink colour.
    cell_patch_margin_frac: float = 0.14
    # ...and the fraction added to each side when a cell's glyph is cut out, so
    # that a tile sitting slightly askew still has its whole letter inside.
    cell_patch_grow_frac: float = 0.20
    # How far a pixel's lightness must sit from its tile's face, in CIELAB L
    # units (0-100), to count as ink. Low enough for a worn or shadowed letter,
    # high enough to ignore the tile's own bevel and moulding.
    ink_contrast_min: float = 10.0
    # Fewest pixels on the ink side of the face colour for Otsu to be asked where
    # the cut goes. Below this there is not enough of a histogram to split.
    ink_otsu_min_pixels: int = 24
    # Lightness window, in CIELAB L units, around the peak of a cell's histogram
    # whose pixels define that cell's face colour.
    face_mode_window: float = 8.0

    # ---- pinning the board's absolute position ---------------------------
    # An empty cell whose face differs from the plain board by this much in
    # Delta-E is taken to be a premium square. Premium squares are printed to
    # stand out, so the gap is large; this only has to clear shadow and white
    # balance across one board.
    premium_delta_e: float = 15.0
    # Fewest visible premium squares before the layout match is trusted. Late in a
    # game most of them are under tiles, and a handful of cells is not evidence.
    premium_min_visible: int = 6
    # Below this the layout is taken to be actively misplaced rather than merely
    # unconvincing. Chance is 0.27 (61 premium squares of 225) and a board shifted
    # by a whole cell measures around 0.15, because a displaced layout is
    # anti-correlated with the true one rather than independent of it. Sitting
    # under chance means only clearly-shifted candidates are demoted.
    premium_reject_below: float = 0.20

    # A floor, so that a badly placed board is heavily penalised but never scored
    # at exactly zero - which would make every candidate tie.
    premium_floor: float = 0.02

    # ---- blank tile detection -------------------------------------------
    # A blank tile carries no ink, so there is nothing to classify; it can only
    # be told from bare board by the colour of its face. Brightness does not
    # separate them (measured: blank 242 against board 238 +/- 6). Colour does,
    # and since both the tile colour and the board colour are measured from the
    # photograph, the test is which of the two a cell's face is nearer.
    detect_blanks: bool = True

    # A blank tile face is smooth. Measured as the spread of lightness in CIELAB L
    # units (0-100), not as a fraction of the mean: a coefficient of variance calls
    # a near-black tile seven times rougher than a cream one at identical noise, and
    # every blank on a dark-tile board was missed for it.
    # Measured over the synthetic set, the usable window is one notch wide:
    #
    #   3.0  23/24 blanks, 0 invented      3.3  24/24, 1 phantom tile
    #   3.1  24/24 blanks, 0 invented      3.5  24/24, 6 phantoms, 1 false blank
    #                                      4.0  24/24, 44 phantoms, 9 false blanks
    #
    # A missed blank is a cell an operator fills in; an invented one is a tile on
    # the board that was never played, so the low side is the safe side. That the
    # window is this narrow is the honest measure of how far colour and smoothness
    # can separate a blank tile from bare board - it is the weakest test in the
    # pipeline and the first thing to distrust on a board it has not seen.
    blank_max_roughness: float = 3.1
    # Tiles and board must differ by at least this much Delta-E before any blank
    # is reported. A cream tile on a cream board is genuinely not separable, and
    # guessing there invents tiles that are not on the board.
    blank_min_separation: float = 12.0

    # ---- classifier ------------------------------------------------------
    # Normalised glyph is glyph_size square. Letters that differ only in a small
    # feature need the resolution to see it: at 32 a 'Q' is an 'O' plus a two-pixel
    # tail, it beats 'O' on overlap by about five percent, and which of them wins
    # comes down to the tilt of the photograph. Measured over the synthetic set,
    # 48 halves the error rate and 64 is not better again.
    glyph_size: int = 48
    glyph_pad: int = 4
    # Folder of pictures of real tiles, one per letter, named A.png ... Z.png. Used
    # in preference to a rendered font when present, because a reference glyph only
    # has to be the right *shape* and no font is the shape of a tile: rendered
    # references confuse D with O, C with O and S with B, which is where most of
    # the remaining letter errors on real photographs came from.
    #
    # A tile set's own artwork is the best reference there is. If your tiles came
    # with an app that draws them, its image assets will do; otherwise photograph
    # one of each letter square-on and crop to the tile.
    # Several folders may be listed, comma separated; a letter then scores as its
    # best-matching shape across all of them.
    template_tiles: str = ""
    # Fewest letters a tile folder must supply before it is used at all.
    template_tiles_min_letters: int = 20
    # Keep the rendered font as an extra shape per letter alongside the tile art.
    # On, because measurement says it is strictly the better of the two and not a
    # compromise between them. Tile art alone reads real photographs about ten
    # points better than a font alone, and reads a board set in some *other*
    # typeface far worse - 90 errors against 2 over the synthetic set. Offering both
    # shapes scores the best of each, since a letter is matched by whichever of its
    # variants fits and the loser costs nothing.
    template_font_fallback: bool = True

    # A preference, not a requirement. If it cannot be opened, a per-platform
    # candidate list is tried (see classify._FONT_CANDIDATES), so the server
    # starts on Windows and macOS as well as in the container.
    template_font: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # A prebuilt bank, taken as given and used instead of building one. This is how
    # tile art reaches a container: build it where the tile images are with
    # scripts/build_templates.py, ship the file, and point this at it.
    template_bank: str = ""
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
