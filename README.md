# Scrabble Capture — detection server

Upload a photograph of a Scrabble board, get back the board state as JSON.

Stage one of the clock-mounted capture system: software first, on a server, with
image upload. The ESP32 talks to the same code later over the WebSocket endpoint.

---

## What it does

```
photo ──▶ find board ──▶ rectify ──▶ lock to grid ──▶ 15×15 cells
                                                          │
                          board state JSON ◀── classify ◀──┘
```

```bash
curl -F "image=@board.jpg" https://your-host/detect?debug=true
```

```json
{
  "ok": true,
  "board": ["...............", "...........F...", "....RETAINS...."],
  "tiles": [{"row": 7, "col": 4, "letter": "R", "confidence": 0.94, "best_iou": 0.71}],
  "low_confidence": [],
  "tile_count": 25,
  "registration": {
    "method": "colour[4]+snap", "grid_score": 97.1, "grid_alignment": 112.4,
    "grid": {"locked": true, "period_frac": 0.9954}
  },
  "timing_ms": {"total": 116.5},
  "debug_image": "/debug/0138ca9a44b8.jpg"
}
```

Board characters: `.` empty, `A`–`Z` a letter tile, `?` a blank tile, `*` a tile
whose letter could not be read.

### Endpoints

| | |
|---|---|
| `POST /detect` | multipart image upload. `?debug=true` also renders a diagnostic strip |
| `POST /detect/raw` | raw image bytes in the body — what the ESP32 will use |
| `WS /ws` | send a JPEG frame, get board state back, one per frame |
| `GET /debug/{name}` | the annotated diagnostic image |
| `GET /health` | liveness, and which template set is loaded |
| `GET /docs` | interactive OpenAPI — the quickest way to try it |

Both detect endpoints accept `?corners=x1,y1,...,x4,y4` to override detection,
which is useful when you want to test the rest of the pipeline in isolation.

---

## On the model: why there is no model

You asked for the best OCR model for this case. The honest answer is that this
is not an OCR problem and a general OCR model is the wrong tool.

OCR engines — Tesseract, PaddleOCR, TrOCR — are built to find lines of text of
unknown length, in unknown fonts, at unknown positions, and read them. Every one
of those unknowns costs accuracy and time. Here, none of them apply:

- The grid is fixed and known. 15 × 15, always.
- One glyph per cell, never a word.
- Always upright, never rotated or italic.
- 27 outcomes, not an open vocabulary.

So the job is 27-class classification of a small canonical glyph, which is the
easiest problem in the family. What it needs is **one reference picture of each
letter**, normalised the same way the extracted glyphs are, compared by overlap.
That is what `app/vision/classify.py` does, and it scores **1492/1500 letters** on
the test set.

One unknown does survive, and it turned out to matter more than any of the others:
the *typeface* is not known, because it differs between tile sets. That is not an
argument for a general OCR engine — it is an argument for a reference glyph of the
right shape, which is far cheaper. See **Reference glyphs** below; it was worth nine
points of accuracy on real photographs.

This is also the answer to "why do we need training data?" — you don't. There is
no dataset and no training step. Reference glyphs come from a rendered font, from
pictures of real tiles (`TEMPLATE_TILES`), or from crops of your own board:

```python
from app.vision.classify import build_from_samples, TemplateBank
bank = TemplateBank(build_from_samples({"A": [crop1, crop2], ...}), "real-tiles")
bank.save("/tmp/scrabble-templates.npz")
```

All three are reference sheets, not training sets — one picture per letter, no
labels beyond the letter itself, no training loop.

**When a learned model would be worth it.** If real photographs turn out to be
much harder than expected — heavy motion blur, deep shadow across the board,
tiles at 25 px instead of 45 — then a small CNN on 32 × 32 crops would beat
template matching, and by then you will have the labelled crops to train it from
the server's own logs. Not before. A general OCR engine would still be the wrong
choice at that point.

---

## Accuracy, measured

60 synthetic boards: **four colour schemes** × five perspective tilts (0–28°) ×
three layouts, with uneven lighting, sensor noise and tiles jittered off-square.
The schemes vary the board face, the border, the tile face and the ink
independently — cream tiles on a pale board, dark red tiles with white letters on
cream, cream tiles on green, near-black tiles with white letters on grey — because
none of those colours is known to the code in advance.

| | |
|---|---|
| Letters read correctly | **1492 / 1500 — 99.47%** |
| Cells correct (all 225 per board) | **13492 / 13500 — 99.94%** |
| Silent errors | **0** — all eight misreads arrived flagged low-confidence |
| Tiles missed, tiles invented | **0** and **0** |
| Blank tiles found | 24 / 24, no false positives |
| Board found | 60 / 60, grid within **0.25 of a cell** of truth at worst |
| Time per board | ~0.95 s |

The zero-silent-errors result is the one that matters. Every cell carries a
confidence and a ranked candidate list, and anything below `LOW_CONF_THRESHOLD`
arrives in `low_confidence` for an operator to confirm. A wrong letter reported
confidently is a corrupted game record; a wrong letter reported doubtfully is one
tap.

Every remaining misread is one of two pairs — `Q`/`O` and `O`/`C` — that differ
only in the shape of one small feature. These boards are drawn with a *font*, and a
font's `Q` is not the shape of a tile's `Q`; given reference glyphs taken from real
tile artwork the same letters read correctly. See **Reference glyphs** below.

### On real photographs

Synthetic accuracy is an upper bound: it says the pipeline is sound and the
thresholds are in sensible places, not what it will do on your board. The five
photographs in this repository cover four boards and three tile colours, on a
wooden table, a red turntable, a maroon turntable and a metal plate.

There is no hand-transcribed ground truth for them, but there is a usable
substitute: every maximal run of two or more letters on a real board is a real
word, and a misread letter breaks both the word it sits in across and the one it
sits in down. Scored against CSW24:

| | |
|---|---|
| Board found and squared | **5 / 5** |
| Tiles located | 94–97 per board |
| Runs that are real words | **193 / 212 — 91.0%**, best board 95% |

Word validity cannot see a misread that still happens to spell something, so treat
it as directional rather than exact.

### Reference glyphs: the single biggest lever on letter accuracy

A reference glyph only has to be the right **shape**, and no font is the shape of
the letters on a tile. That sounds like a detail and it was worth nine points:

| reference glyphs | words read correctly |
|---|---|
| rendered font (Arial Bold) | 173 / 209 — 82.8% |
| real tile artwork + the font | **193 / 212 — 92.3%** |

The failures a font produces are not random. They are `D`→`O`, `C`→`O`, `S`→`B` —
pairs separated by the shape of exactly one feature, which is the thing that varies
between typefaces. `EROICLY` came back `EROIOIY`, `AIDE` as `AIOE`, `RERENTS` as
`RERENTB`. With tile artwork those largely go away and one board reads every word
correctly.

Point `TEMPLATE_TILES` at a folder of images named `A.png` … `Z.png`, cropped to
the tile; the letter is isolated from its point value automatically. Several folders
may be listed and a letter then scores as its best-matching shape across all of
them, which is how one deployment reads more than one tile set. The rendered font
stays in as one more shape per letter — measured, that is better than either source
alone, because tile art alone reads a board in some *other* typeface far worse (90
errors against 2 over the synthetic set).

The container has no tile artwork on it, so build the bank where the images are and
ship the file:

```bash
python scripts/build_templates.py templates/protiles.npz "/path/to/tiles/white-pro"
```

Then set `TEMPLATE_BANK=/app/templates/protiles.npz`, which is loaded as given.

### What is not worth trying

Each of these looked right, was measured, and lost:

- **Reading the printed point value** to separate `Q`(10) from `O`(1), `D`(2) from
  `O`, `S`(1) from `B`(3). This one is built and switched off rather than removed —
  see `USE_POINT_VALUE` — because it is waiting on the camera, not on a better
  method. Digit references are taken from the same tile artwork as the letters, so
  they need no labelling: `Q.png` *is* a reference for "10". Measured:

  | | |
  |---|---|
  | a digit-sized mark found on | 37% of tiles |
  | best precision reading it | **54%**, at 13% of tiles; nothing qualifies higher |
  | words read on real photos | 91.0% off, 90.6% on, 89.6% at triple the weight |

  It gets worse the more it is trusted, which is what a below-chance signal looks
  like. The cause is size: the value is ~8 px tall here, and every digit at 8 px is
  a blob that matches `8` best because `8` has the most ink. At ~28 px it should be
  a strong signal — it is the main reason to photograph the board bigger.
- **Softening the templates** to match a photographed glyph's blur: doubled the
  misreads, 3 → 6.
- **Cheaper colour proposals** — clustering morphology on the thumbnail (lost whole
  boards), at half resolution (no measurable saving, slightly worse), or fewer
  cluster counts (lost boards).

---

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

# generate a test board and try it
python scripts/make_fake_board.py /tmp/fake --tilt 15 --seed 1
curl -F "image=@/tmp/fake/board.jpg" "localhost:8000/detect?debug=true"

# generate one of the other colour schemes
python scripts/make_fake_board.py /tmp/dark --scheme dark_tiles --tilt 22

pytest tests -q          # 79 tests, ~95s, no image fixtures needed
```

### Deploying on Dokploy

Compose application, pointed at this repo — `docker-compose.yml` is picked up as
is, and Traefik handles TLS and routing to port 8000. Or a Dockerfile
application, port 8000. Nothing is published on the host by default; uncomment
the `ports` block for local testing.

The container is stateless. Debug images and the template cache live in `/tmp`
and are disposable — no volumes needed, and it scales horizontally as is.

Every setting in `app/config.py` is an environment variable of the same name.
The ones worth knowing:

| | |
|---|---|
| `WARP_SIZE` | rectified board size in px. 720 gives 48 px cells |
| `LOW_CONF_THRESHOLD` | below this, a cell is sent for confirmation |
| `TEMPLATE_TILES` | folders of real tile images, `A.png`…`Z.png`. Worth ~9 points of letter accuracy over a font — see Reference glyphs |
| `TEMPLATE_BANK` | a prebuilt bank from `scripts/build_templates.py`, used as given. How tile art reaches a container |
| `TEMPLATE_FONT` | path to the font templates are rendered from when no tile art is given |
| `DETECT_BLANKS` | see the warning below |
| `MIN_GRID_SCORE` | how much of a grid a candidate must show to be called a board. Lower it if a board is being missed, raise it if something board-shaped but blank is being picked |
| `TILE_FACE_DELTA_E` | how far a cell's colour may sit from the measured tile colour and still be a tile, in CIELAB Delta-E. Raise it if tiles in shadow are missed |
| `DETECT_COLOUR_CLUSTERS` | how many colours the frame is clustered into, once per value. Fewer is faster and finds fewer boards |
| `GRID_PERIOD_RANGE` | leave alone unless grid lock fails |

---

## Things that were harder than they looked

Worth knowing before you tune anything, because each of these was found by
measurement and each cost real accuracy.

**The tile's point-value digit is the single biggest source of errors.** At board
resolution its ink often touches the letter's, so both arrive as one connected
component, and an `O` with a `1` stuck to its lower right is an excellent `Q`.
Left unhandled this was 95% letter accuracy instead of 99.9%. It is fixed by
eroding a pixel to part the bridge, then keeping only fragments near the main
one's centre — a letter's strokes cluster around its own centre, a corner digit
does not. Relative area alone does not work: it cannot tell a digit from the arm
of a `K`.

**Finding the board is not the same as finding the grid.** The detected quad is
roughly the board; the playable area may be inset by a printed border or, if the
detector was a little tight, slightly larger than the quad. Get that wrong and
every cell shifts, which reads as random letter errors far from the board centre.
So a second pass locks onto the printed grid lines directly, warping a margin of
context around the quad so it can correct in either direction. It fits one pitch
and one phase against both axes at once — independent phases let one axis lock a
whole row off, which silently relabels the entire board, and that is the worst
failure mode available here.

**Do not key detection to a colour.** A red border is not a reliable signal: a
board may have no border, and a red *turntable* under the board is a large,
convincingly square-ish red region that a colour test cannot tell from one. Ask
instead what every board has and no table has — a 15 × 15 lattice of printed grid
lines filling a square. Several cheap detectors propose candidate quads (regions
of uniform colour at several clustering granularities, regions busier than a bare
table, contours on Canny edges), each candidate's edges are then fitted to the
nearest strong straight lines, and the winner is chosen by how well a 15 × 15 comb
fits inside it. Two measures are needed there and they are not interchangeable:
whether a lattice is present at all, which is what rejects the turntable, and how
exactly its lines land, which is what chooses between two quads that both contain
the board. See the notes in `app/vision/registration.py`.

**Do not assume dark letters on pale tiles either.** Both threshold polarities are
searched, so white letters on dark red tiles read as well as black on cream, and
each cell's ink is cut against *that cell's own face colour* rather than a window
around it — a tile at the edge of the played cluster has half a board in that
window, which loses its letter to slivers. The tile face colour is not configured;
it is measured from the cells that hold letters, on the one assumption that holds
for every tile set: they all share one face colour, whatever it is.

**Blank tiles are the weakest part.** Brightness does not separate a blank tile
from bare board at all — measured at 242 against 238 ± 6. Colour does, and since
both the tile colour and the board colour are measured from the photograph, the
test is simply which of the two a smooth empty cell's face is nearer. If the two
are too close to call, no blank is reported and every empty cell says so in its
notes, because a wrongly reported blank is a tile that is not on the board.

And note the harder truth: even perfect detection only tells you a blank is
*there*. Which letter it was played as is not in the pixels. That needs the game
record — or the operator.

---

## Layout

```
app/
  main.py              FastAPI app, endpoints, WebSocket
  config.py            every tunable, one place, all env-overridable
  schemas.py           response models and the board-character convention
  vision/
    registration.py    find the board, rectify it, lock onto the grid
    grid.py            illumination flattening, cells, glyphs, blanks
    classify.py        template bank and matching
    pipeline.py        orchestration and debug rendering
scripts/
  make_fake_board.py   synthetic board photos plus ground truth
tests/
  test_pipeline.py     end to end, parametrised over tilt and seed
```

---

## Next

- Point it at real photographs and see where the thresholds drift. Use
  `?debug=true` on every failure — the diagnostic strip shows the detected quad,
  the rectified grid, and the binary mask side by side, which is usually enough
  to see the cause immediately.
- Build the UI against `/detect`. The `low_confidence` list is the operator
  queue, and `candidates` gives the pick-list for each entry.
- Then the ESP32, against `/ws`.

Two changes are already measured to be worth making, in this order:

**Add tile artwork for each tile set you actually use.** Protiles artwork took one
board from 85% to every word correct; the board it did *not* describe — dark red
tiles, white letters — is still the worst of the five at 85%, and its errors are the
signature of a wrong reference shape. One photograph per letter, listed in
`TEMPLATE_TILES`, is the whole job.

**Photograph the board bigger.** At ~46 px per cell several things are at their
limit at once: thin white-on-dark strokes thin out, and the printed point value —
which would settle `Q`/`O`, `D`/`O`, `C`/`O` and `S`/`B` outright — is only ~8 px
tall and unreadable (measured: recoverable on 27% of tiles, and the signal
indistinguishable from stray ink). At ~2500 px across the board, the digit is ~28 px
and can simply be classified.

Two more things that will matter for the device and are worth designing for now:

**Board history changes what is possible.** Right now each request is
independent. Once the server tracks a game, two things get much better almost
free: a letter can be voted across many frames instead of decided from one, and a
blank becomes reliably detectable as "this cell changed and has no glyph".

**A fiducial marker would pay for registration.** Registration is about two thirds
of the ~0.95 s and is redone every frame. The board turns about its centre as play
passes between players, so the corners cannot simply be cached — but a sticker or
ArUco marker at one corner gives orientation directly, which turns the expensive
part into a cheap one: find the marker, and the four corners follow from the board's
known geometry. The grid-lock pass stays as the check that the result is right, and
the existing detector stays as the fallback when no marker is visible.

**The clock press is the turn boundary.** When the ESP32 sends frames, have it
send the press event too, and capture from a short ring buffer rather than at the
instant of the press — hands are often still over the board.
