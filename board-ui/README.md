# Virtual board

A standalone Scrabble board you can look at, type into, and step through turn by turn.
It loads and saves JSON and talks to nothing else - no server, no network, no build
step.

## Running it

```
cd board-ui
python -m http.server 8000
```

Then open <http://localhost:8000>.

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

Stepping through turns is a **stand-in for the clock**. A person clicking "new turn" is
doing, for now, the job a clock press will do later; `state.addTurn()` is the single
function that a real trigger should call.

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

## What it will not do

Scoring, word checking, racks, clocks, and anything over a network. Those are left out
because the things they depend on do not exist yet, and guessing at their interfaces
now would mean rewriting against the real ones later.

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
