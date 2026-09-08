# Backlog

Work deliberately set aside, with the measurements that justify picking it up again.

Current priority is one board and one tile set - the brown-rimmed, green-faced board
with cream Protiles - because that is the set being played on while the hardware is
built. Generalising to other boards is the largest block of work here and is all
deferred until that path works end to end.

Numbers below come from 55 boards checked cell by cell (`scripts/review_letters.py`)
and 161 boards judged for registration (`scripts/review_registration.py`). Re-measure
with `scripts/score_letters.py` and `scripts/score_corners.py` before acting on any of
them; several are already stale by design.

---

## 1. Generalising beyond the current tile set

**Where it stands.** 98.996% of cells correct on the target board, 98.343% across all
55 checked boards. Most of the gap is other tile sets.

### 1.1 WESPAC 2025 round revolving boards

The single largest block of failures. 37 photographs - 23% of the corpus - producing
48% of all registration failures.

| camera angle | WESPAC wrong | other boards wrong |
|---|---|---|
| square on (<8 deg) | 100% (n=5) | 24% (n=88) |
| tilted (8-15 deg) | 80% (n=15) | 37% (n=27) |
| steep (>15 deg) | 94% (n=18) | 80% (n=5) |

They fail at *every* angle, so the angle is incidental. The cause is shape: a square
playable area inset in a large round navy turntable. Every detector in
`registration.py` looks for a square region or a square outline, and the dominant
shape here is a circle, so the quad comes out offset and mis-scaled - taking in table
on one side while clipping the first column on the other. Their fitted pitch varies
three times as widely as other boards (0.907-1.080 against 0.955-1.019), phase error
is 2.3x larger, and 2 of 37 fail to lock at all where other boards never do.

**Proposed.** Find the circle first (Hough circles, or the enclosing ellipse of the
turntable region), then look for the square *inside* it. Nothing currently proposes a
quad from a circular parent shape.

They also read badly *when registration succeeds* - the four that register correctly
read only 51-73% of words. Pale yellow cells with dark navy tiles and white letters is
the inverse polarity of the current set, and their premium squares are a non-standard
palette. Treat as a separate problem from the registration one, and measure it only on
WESPAC boards that registered correctly, or the two confound each other.

### 1.2 Non-standard premium palettes weaken two safety checks

Six of 89 confirmed boards score 0.34-0.53 on premium agreement while being registered
perfectly, purely because their premium squares are not the usual red and blue - the
corpus has a red and yellow board, a grey and purple one, and a white one with black
tiles. Premium agreement is matched partly on colour, so it is weaker evidence on those
boards than its number suggests.

This already forced `shift_probe_below` down from 0.50 to 0.30 to stop the shift
correction firing on correctly-registered boards. A colour-agnostic premium test -
comparing relative groupings rather than absolute hues - would restore the margin.

---

## 2. Reading the remaining letters

Errors left on the target board, 70 over 6975 cells:

| | count | tractable? |
|---|---|---|
| thin letters (I, L, T) read as blank | 14 | **not by tuning** - see below |
| O and Q confused, both directions | 9 | partly; needs better reference glyphs |
| empty squares read as blank | 9 | lives in the 4-8 Delta-E band |
| long tail (P->R, F->E, I->M, ...) | ~38 | mostly better reference glyphs |

### 2.1 Thin letters - a closed door, recorded so it is not reopened blindly

I, L and T are frequently reported as blanks. `_cell_glyph` finds no usable ink on
those cells at all, so the shape filters are not the bottleneck. Five separate
relaxations were measured and every one was net-negative on the target board (76
errors baseline):

| relaxation | errors |
|---|---|
| `ink_contrast_min` 10 -> 7 | 80 |
| `ink_contrast_min` 10 -> 5 | 84 |
| `glyph_min_side_frac` 0.30 -> 0.22 | 76 (no change) |
| `glyph_min_area_frac` 0.025 -> 0.012 | 81 |
| `glyph_min_extent` 0.20 -> 0.12 | 78 |
| `max_glyph_shift_frac` x1.5 | 82 |

Each admits more noise than signal. This needs a different extraction for faint,
thin strokes, not a threshold.

### 2.2 The Protiles font does not help - measured, August 2026

A TrueType file of the Protiles face was tried as a source of reference glyphs. Its
letters come with the point value already attached as a subscript, so each was rendered
as a tile face and put through `build_from_tiles`, which strips the value the same way
it does for photographs of real tiles. All 26 isolated cleanly.

Measured over the 31 checked boards of the target set:

| templates | errors |
|---|---|
| Protiles artwork + Arial fallback (current) | **70** |
| Protiles font alone | 87 |
| artwork + Protiles font | 95 |
| artwork + Protiles font + Arial | 69 |

It does not help, and the reason is worth keeping. The bank *already* uses genuine
Protiles artwork from the Zyzzyva install, so the font is the same typeface arriving a
second time. A letter scores as its best-matching variant, so a near-duplicate variant
gives every letter another way to be matched without telling the matcher anything new -
and the one confusion that matters got worse, 'O' read where 'Q' belonged going from 7
to 10, against 'O' for 'C' improving from 2 to 0. The best configuration is one error
better than the current one out of 6975 cells, which is noise.

The general lesson, already written into `classify._build` and now measured: more
reference shapes are not automatically better. A second set earns its place only by
describing the tiles *differently and correctly*.

This would still be worth trying for a tile set whose artwork is **not** available, where
a font would be the only reference there is.

### 2.3 Reference glyphs from real tile artwork

**Needs the user, not the code.** ~27 square-on photographs per tile set, one per
letter, feeding `settings.template_tiles` via `scripts/build_templates.py`.

Reference glyphs taken from the Protiles artwork in the local NASPA Zyzzyva install
were worth +9 points when introduced. Photographs of the actual tiles should beat
that, and the test suite already records that the font-rendered 'Q' - the largest
single confusion remaining - reads correctly when given real tile artwork. Artwork
read from the Zyzzyva install is never committed or redistributed.

### 2.3 Blank tiles on sets whose blanks are not plain

The current rule is "tile colour, no ink, therefore blank", which is exactly right for
Protiles - they carry no marking whatsoever. Other sets print a star or similar on the
blank, which will be found as a glyph and misread as a letter. Needs a per-set notion
of what a blank looks like before those sets can be read.

---

## 3. Registration

### 3.1 Line detector second pass

`_line_quads` proposes nothing at all on about 40% of photographs, because it needs
four sides and frequently finds three - the far edge of a tilted board is
foreshortened, low in contrast, and interrupted by racks. A retry at a shorter minimum
length when the first pass comes up empty is cheap and aimed at the boards still
failing.

Caveat measured afterwards: when the line detector *does* win it is correct only 24%
of the time against a 56% base rate. It never made a board worse, and it fixed four
outright, but "make it propose more" is not obviously good. Measure before extending.

### 3.2 Corner ground truth goes stale

`fixtures/corners.json` is harvested from confirmed registrations, so it is anchored to
whatever the code did at harvest time. Re-harvest after any registration change, or
`scripts/score_corners.py` measures agreement with a configuration that no longer
exists. It has already contained a board that was confirmed but wrong.

Score by **cell centres**, which `score_corners.py` now reports, not by corner
distance. The grid lock re-fits pitch and phase after the quad is chosen, so corner
distance both absorbs and manufactures disagreement: one confirmed pair is 1.67 cells
apart at the corners and 0.15 apart cell for cell, and another is 0.37 apart at the
corners and 1.66 apart at the cells.

### 3.3 Sub-cell drift

15 of 49 photographs of the target board were judged wrongly registered, and 14 of
those read 79-98% of their words correctly. They are half-a-cell imperfections that
look wrong and transcribe fine. Chasing them buys about a point; it is on the list only
so that a future measurement is not surprised by it.

---

## 4. Orientation, and why it matters more than it used to

**The standard premium-square layout is perfectly 4-fold rotationally symmetric** -
verified, all three rotations reproduce the printed layout exactly. Nothing in this
pipeline can therefore detect a rotated board. A board turned 90 degrees registers
cleanly, reads every tile, passes every check, and yields a transcript with every
coordinate wrong.

This was harmless while the output was a board state and becomes serious the moment the
output is a move log with coordinates in it.

Two options, in preference order:

1. **Resolve rotation against the previous turn's board.** Try all four rotations and
   keep the one whose existing tiles match what is already known. Needs no hardware,
   cannot fall off, and is unambiguous once a handful of tiles are down. Only the
   opening play is ambiguous, and there it does not matter.
2. **A corner marker (ArUco or a sticker).** Raised earlier and deferred. Still the
   more robust answer for an empty board, and worth having if option 1 proves fiddly.

---

## 5. Live capture and the game loop

Not started. Design notes only, so the shape is not re-derived from scratch.

**Latency.** 4.5 s for a 4K frame today: 1.2 s registration, 3.25 s warping and reading
six candidate positions, 0.08 s classifying. A camera that does not move should reuse
its corners - `pipeline.detect(image, corners=...)` already accepts them - which
removes the registration pass and five of the six reads. Roughly half a second a frame,
which is ample for a clock-triggered scan and not enough for continuous video. A `/ws`
WebSocket endpoint already exists.

**Resolution.** Accuracy here was measured on 4K photographs, and 72 pixels per cell
measurably beats 48. The board must therefore span about 1080 pixels in the source
image. An ESP32-CAM at 1600x1200 clears that only if the board fills the frame - worth
checking early, because falling short costs several percent silently.

**Diffing is what makes it practical.** Scanning on the clock press gives a stable,
complete board and a natural diff point. Per turn the operator confirms about 7 new
tiles rather than 95; a tile confirmed once is known for the rest of the game; and a
tile played early is re-observed 20+ times, so a modal reading per cell across the game
should beat any single scan by a wide margin. That temporal vote is likely worth more
than further work on the vision.

**Store board states, generate GCG.** GCG is a move log; this pipeline produces board
states, and converting one to the other is where the ambiguity lives. Keep the per-turn
scans as the source of truth so a mis-derived move can be re-derived without
re-photographing a finished game.

---

## 6. Rack camera - planned, not started

The intended second camera, and the reason the GCG output can eventually be complete
rather than partial. From the board alone:

| GCG field | from the board camera |
|---|---|
| coordinate, word, direction | yes, from the diff |
| score and running total | yes, computable - the premium layout is already known |
| blank shown lowercase | partly; the blank is detected but not which letter it stood for, though the lexicon usually determines it |
| the player's rack | **no** - the leave is never visible |
| exchanges against passes | **no** - both are a clock press with no board change |

A rack camera closes the last two, which are exactly the fields Quackle and Macondo
need for analysis. Until then a camera-only GCG is a good game record and a weak
analysis input.

Worth designing the turn log so a rack observation can be attached to a turn later
without rewriting what is already stored.
