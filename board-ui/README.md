# Virtual board

A Scrabble board you can look at, type into, and step through turn by turn. It reads
boards from photographs through the vision service, and falls back to typing them in by
hand when a photograph will not do. Plain HTML, CSS and modules - no build step.

## Running it

Two servers: the page itself, and the board reader it scans with.

```
cd board-ui
python -m http.server 8200        # the page
```

```
python -m uvicorn app.main:app --port 8000    # the reader, from the repository root
```

Then open <http://localhost:8200>. The page works without the reader - you just have to
type boards in instead of scanning them.

**It will not work if you open `index.html` directly.** The code is split into ES
modules, and browsers refuse to load those over `file://`. If the page comes up blank,
that is almost always why.

Press **load sample** to get a board to play with.

## What it is for

Three things at once, which is why it is worth more than it looks:

- **A test harness.** It draws what the vision pipeline read, so you can see whether
  the pipeline got it right without a camera, a board, or a game in front of you.
- **A manual fallback.** When a photograph is unreadable - bad light, an unfamiliar
  tile set, a knocked camera - a person can type the board in instead. A system that
  degrades to manual entry is far more use in a hall than one that stops.
- **The start of the operator console.** The board is built as a reusable piece rather
  than a page, so it can move into the live console without a rewrite.

Stepping through turns is a **stand-in for the clock**. A person pressing "new turn +
scan" is doing, for now, the job a clock press will do later. When the trigger becomes
real hardware it should call `runScan(true)` in `app.js` and nothing else, so that
swapping a button for a wire is one change.

## Using it

Click a square and type. That is the whole thing.

| | |
|---|---|
| click a square | put the cursor there |
| type a letter | place a tile and move on, stepping over tiles already down |
| **Shift** + letter | place a **blank** designated as that letter |
| click the same square again, or **Tab** | switch between typing across and down |
| **Backspace** | clear this square, or step back and clear that one |
| **Delete** | clear this square without moving |
| arrow keys | move without editing |
| **Escape** | deselect |
| **Ctrl/Cmd + Z**, **Ctrl/Cmd + Shift + Z** | undo, redo |
| **Home** / **End** | first turn, last turn |

Typing over a tile that is already there steps past it rather than replacing it, the
way a crossword does. To change a tile, clear it first.

Past turns are **read-only** until you press *edit this turn*. Amending history can
invalidate everything after it, so it is deliberately a decision rather than an
accident. Nothing is propagated forward - if an edit makes later turns inconsistent,
that is for you to sort out.

## Scanning a photograph

The board reader runs as a small HTTP service. Start it from the repository root:

```
python -m uvicorn app.main:app --port 8000
```

The **Scan** panel says whether it can see it. Then:

- **new turn + scan** - the ordinary move. Copies the board forward into a new turn,
  asks for a photograph, and fills the turn in from it.
- **rescan this turn** - read the same turn again, after moving the camera or the light.

On a phone the file picker offers the camera directly; on a desktop it is an ordinary
file chooser. Either way you get a still, which is all the reader wants.

**Where the picture comes from is the only thing that varies.** A photograph, a webcam
frame and a camera on another machine all end at the same three steps: post the bytes,
map the answer to cells, fill the turn. `src/scan.js` keeps that as a seam -
`fromFile()` is wired to the buttons and `fromCamera(video)` is written and waiting for
one.

### The board gets turned

A board rotated between turns reads perfectly and puts every coordinate in the wrong
place, because the premium squares are 4-fold symmetric and cannot say which way round
anything is. So a scan is matched against the tiles already known from the previous
turn, and turned to fit. It says so when it does. On an opening scan there is nothing
to match against, and nothing yet to be inconsistent with.

### When a scan is wrong

It often will be, and the panels are there to say so. A reading that is impossible -
more blanks than exist, two Zs - fails the Distribution check outright, with no
confidence score needed. Cells the reader doubts are ringed and marked. A registration
the reader is unsure of is reported after the scan, and is usually worth re-taking the
photograph for.

Nothing about a scan is final: it fills the turn in, and then you correct it by hand.
The whole point of scanning into a *new* turn is that only the handful of new tiles
need checking, rather than the whole board.

## What it will not do

Scoring, word checking, racks, and clocks. Those are left out because the things they
depend on do not exist yet, and guessing at their interfaces now would mean rewriting
against the real ones later.

## The JSON

The whole point of the file format is that the Python side writes it and reads it, so
it is worth knowing two things about it.

**Every turn holds the entire board, not just what changed.** Deltas would be smaller
and more elegant. Snapshots cannot cascade a failure, can be hand-edited one turn at a
time, and let any turn be drawn without replaying the game. The "new this turn"
highlight is computed by diffing at render time.

**A letter may be `null`.** That means a tile is on the square but its letter is not
known - which is the ordinary state of a blank straight from a camera, since a blank's
designation is not in the pixels at all, and of any tile the reader could not name.
Over 55 checked boards that is about 1.7 tiles per board. `isBlank` is a separate flag,
and letters are always uppercase; the lowercase-means-blank convention belongs to the
GCG format and should stay in whatever writes GCG.

`confidence: null` means *not from OCR*, and renders as certain. A tile a person typed
is the most trustworthy one on the board.

## Layout

```
index.html      markup
style.css       all styling; every colour is a custom property
src/
  constants.js  premium grid, tile distribution, values, thresholds
  state.js      the document, the edit log, undo - and no DOM at all
  board.js      draws the 15x15 grid and owns the cell elements
  editor.js     cursor, keyboard, writes
  turns.js      the turn strip and the read-only lock
  pool.js       what is left in the bag
  validate.js   the structural checks
  io.js         import, export, autosave, query string
  scan.js       frame sources, the call to the reader, orientation matching
  app.js        wiring; the only file that knows about all the others
fixtures/
  sample-game.json
```

`state.js` never touches the DOM and the renderers never write state directly. An
interaction calls a state function, state emits, renderers redraw. Keeping that one
direction is what will let the board drop into the operator console later.

## The sample game

`fixtures/sample-game.json` is deliberately not a clean game. It contains a blank
played as an A, two tiles read doubtfully, one tile typed in by hand, a passed turn,
and - in the last turn - **a second Z**, which is impossible. A fixture where
everything is right only ever proves the happy path, and the error states are the ones
that get looked at.
