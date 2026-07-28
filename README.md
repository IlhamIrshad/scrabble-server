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
  "registration": {"method": "red_border", "grid": {"locked": true, "grid_period": 43.9}},
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
- One known font, always upright, never rotated or italic.
- 27 outcomes, not an open vocabulary.

So the job is 27-class classification of a small canonical glyph, which is the
easiest problem in the family. What it needs is **one reference picture of each
letter**, normalised the same way the extracted glyphs are, compared by overlap.
That is what `app/vision/classify.py` does, and it scores **874/875 letters** on
the test set.

This is also the answer to "why do we need training data?" — you don't. There is
no dataset and no training step. The 26 reference glyphs are rendered from a font
at startup and cached. To match your real tiles more closely, either point
`TEMPLATE_FONT` at a closer font, or build templates by averaging real crops:

```python
from app.vision.classify import build_from_samples, TemplateBank
bank = TemplateBank(build_from_samples({"A": [crop1, crop2], ...}), "real-tiles")
bank.save("/tmp/scrabble-templates.npz")
```

That is still a reference sheet, not a training set — a handful of crops per
letter, no labels beyond the letter itself, no training loop.

**When a learned model would be worth it.** If real photographs turn out to be
much harder than expected — heavy motion blur, deep shadow across the board,
tiles at 25 px instead of 45 — then a small CNN on 32 × 32 crops would beat
template matching, and by then you will have the labelled crops to train it from
the server's own logs. Not before. A general OCR engine would still be the wrong
choice at that point.

---

## Accuracy, measured

35 synthetic boards, perspective tilt 0–30°, uneven lighting, sensor noise,
tiles jittered off-square:

| | |
|---|---|
| Letters read correctly | **874 / 875 — 99.89%** |
| Cells correct (all 225 per board) | **7874 / 7875 — 99.99%** |
| Silent errors | **0** — the one mistake was flagged at 0.15 confidence |
| Blank tiles found | 24 / 24, no false positives |
| Grid lock | correct on all 35, pitch within 0.2% of truth |
| Time per board | ~115 ms |

The zero-silent-errors result is the one that matters. Every cell carries a
confidence and a ranked candidate list, and anything below `LOW_CONF_THRESHOLD`
arrives in `low_confidence` for an operator to confirm. A wrong letter reported
confidently is a corrupted game record; a wrong letter reported doubtfully is one
tap.

Synthetic accuracy is an upper bound. It says the pipeline is sound and the
thresholds are in sensible places, not that it will hit 99.9% on your board.

---

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

# generate a test board and try it
python scripts/make_fake_board.py /tmp/fake --tilt 15 --seed 1
curl -F "image=@/tmp/fake/board.jpg" "localhost:8000/detect?debug=true"

pytest tests -q          # 52 tests, ~18s, no image fixtures needed
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
| `TEMPLATE_FONT` | path to the font templates are rendered from |
| `DETECT_BLANKS` | see the warning below |
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
the board's outer edge; the playable area is inset by the printed border. Get
that inset wrong and every cell shifts progressively, which reads as random
letter errors far from the board centre. So a second pass locks onto the printed
grid lines directly. It fits one pitch and one phase against both axes at once —
independent phases let one axis lock a whole row off, which silently relabels the
entire board, and that is the worst failure mode available here.

**Do not use the red border's inner hole as the grid**, tempting as it is. A
board's triple-word squares are red and sit on the outermost row, so they merge
with the border under any closing operation and eat into the hole unevenly. The
result is a quad that is a cell out on some edges and not others, which no later
correction can undo.

**Blank tiles are the weakest part.** Brightness does not separate a blank tile
from bare board at all — measured at 242 against 238 ± 6. Saturation does, since
a cream tile face reads about 50 against about 10 for plain board, but premium
squares reach 65, so it needs a hue gate with it. That combination is clean on
test boards, but those constants describe one board and one tile set. Re-measure
`BLANK_*` on yours, or set `DETECT_BLANKS=false`.

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

Two things that will matter for the device and are worth designing for now:

**Board history changes what is possible.** Right now each request is
independent. Once the server tracks a game, two things get much better almost
free: a letter can be voted across many frames instead of decided from one, and a
blank becomes reliably detectable as "this cell changed and has no glyph".

**The clock press is the turn boundary.** When the ESP32 sends frames, have it
send the press event too, and capture from a short ring buffer rather than at the
instant of the press — hands are often still over the board.
