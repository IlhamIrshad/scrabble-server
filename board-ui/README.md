# Virtual board

A Scrabble board you can look at, type into, and step through turn by turn. It reads
boards from photographs through the vision service, and falls back to typing them in by
hand when a photograph will not do. Plain HTML, CSS and modules - no build step.

## Running it

Two servers: the page itself, and the board reader it scans with.

```
cd board-ui
python serve.py                   # the page, on 8200
```

```
python -m uvicorn app.main:app --port 8000    # the reader, from the repository root
```

Then open <http://localhost:8200>. You land on **Games**, the list of everything saved;
opening one goes to the board. The reader is not optional any more - games live on the
server, so the page needs it to sign in and to load anything.

Sign-in is set in `.env` as `UI_USERNAME` and `UI_PASSWORD`, which is not tracked. With
no password set the server refuses every attempt and the page says so, rather than
looking like a mistyped password.

**It is a doorbell, not a lock.** The token lives in the server's memory, the traffic is
plain HTTP, and the whole thing assumes one person on one machine. Do not put it on a
network and believe the games are protected.

**It will not work if you open `index.html` directly.** The code is split into ES
modules, and browsers refuse to load those over `file://`. If the page comes up blank,
that is almost always why.

`serve.py` is `python -m http.server` with caching switched off. Any static server will
do in principle, but the plain one lets the browser decide how long a file stays fresh,
and it decides generously - an edited module keeps being served from cache and the page
goes on running the old version, which looks exactly like a change that did not work.

Press **More** (the dots in the ribbon) and then **Load the sample game** to get a
board to play with.

## Where games live

One folder per game under `games/`, alongside the photographs each turn was read from:

```
games/<id>/game.json
games/<id>/frames/t003-1758240000.jpg
```

Files rather than a database, because a game *is* a document, the format was designed
to be hand-editable, and a folder you can open beats a row you have to query. The board
saves about half a second after you stop typing and says so in the header - **saved**,
**saving...**, or the reason it could not. A board that has quietly stopped saving looks
exactly like one that is saving, right up until the browser closes, so it never stays
silent.

`games/` is gitignored, for the same reason `photos/` is.

Each row on the games list answers one question before you open it: **is this game
finished being checked?** Turn and tile counts say how big a game is, which is rarely
why anyone opens one; the count of tiles still waiting on a person is why, so it is the
only thing on the page rendered in a colour. `GET /games` computes it from the last turn
of each game, and the bars beside the name are the tile count at each turn - the cheapest
honest picture of a game's shape without opening it.

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

The same table is in the page, behind **Keys** in the top bar, so it does not have to
be remembered from here.

**New game**, on the games list, starts an empty document under a new name - use it
before scanning a board from a different game, or the boards pile up as turns of the
same one. **Delete the last turn** is in the **More** sheet; nothing is deleted from the
middle, because it would renumber every turn after it.

The ribbon along the bottom holds four moves: step back, step forward, **More**, and
**new turn + scan**. Import, export, undo, redo, delete-last, the sample game and the
reader's address all live in the More sheet. They were a row of eleven buttons, which
put `load sample` next to `export` and `delete last` next to the turn navigation - one
slip from a game that has no second copy.

Typing over a tile that is already there steps past it rather than replacing it, the
way a crossword does. To change a tile, clear it first.

Past turns are **read-only** until you press *edit it anyway*. Amending history can
invalidate everything after it, so it is deliberately a decision rather than an
accident. Nothing is propagated forward - if an edit makes later turns inconsistent,
that is for you to sort out.

## Confirming a turn

The **New this turn** card at the top of the rail is what the whole diffing argument is
for: per turn a person checks about seven new tiles rather than ninety-five, which is
what makes recording a game during a timed round possible at all.

It lists the squares this turn added as chips - coordinate, then letter - with the ones
the reader is unsure of marked. Clicking a chip goes to that square on the board, because
the correction happens there and not in the rail.

**Confirm** records that a person looked. A confirmed tile is stored exactly as a typed
one, `confidence: null`, because a tile somebody checked is as trustworthy as one
somebody typed, and a "confirmed but 0.62" state would make every reader of the file
decide what to do with it. Nothing is lost: the reading it arrived with is in the edit
log, which is also what undo walks back, so one press restores the whole turn.

Two things it will not pretend to fix. A tile with **no letter** - unreadable, or a blank,
whose designation is not in the photograph at all - needs somebody to type one, so
confirming does not count it. And tiles **carried forward** from earlier turns that were
never confirmed are listed separately: every turn holds the whole board, so a doubtful
reading from turn 2 is still doubtful in turn 9, and it is exactly what the games list
counts.

## Scanning a photograph

The board reader runs as a small HTTP service. Start it from the repository root:

```
python -m uvicorn app.main:app --port 8000
```

Its address, and whether the page can see it, are in the **More** sheet. Then:

- **new turn + scan** - the ordinary move, and the only primary button in the ribbon.
  Copies the board forward into a new turn,
  asks for a photograph, and fills the turn in from it.
- **Rescan**, in the New this turn card - read the same turn again, after moving the
  camera or the light.

On a phone the file picker offers the camera directly; on a desktop it is an ordinary
file chooser. Either way you get a still, which is all the reader wants.

**Where the picture comes from is the only thing that varies.** A photograph, a webcam
frame and a camera on another machine all end at the same three steps: post the bytes,
map the answer to cells, fill the turn. `src/scan.js` keeps that as a seam -
`fromFile()` is wired to the buttons and `fromCamera(video)` is written and waiting for
one.

### The board gets turned

A photograph of a board rotated between plays reads perfectly and puts every coordinate
in the wrong place, because the premium squares are 4-fold symmetric and cannot say
which way round anything is.

That is a property of the *photograph*, so it is corrected on the way in: a scan is
matched against the tiles already known from the previous turn and turned to fit before
its cells are stored. It says so when it does. The virtual board is then always the
right way up, and is always drawn straight.

The one case that cannot be corrected automatically is an opening scan, where there are
no earlier tiles to match against. **turn scan** handles that by hand: it turns this
turn's tiles a quarter turn, as an ordinary edit that undoes in one step.

There was, briefly, a control that turned the *display* instead. It was a mistake and is
recorded as one in `board.js`: turning the drawing moves the squares while leaving the
letters upright, so words run backwards and the board becomes unreadable. Nothing about
the virtual board ever needs turning.

### Comparing against the photograph

**Photograph** opens the picture *beside* the board, not instead of it. Checking a
reading means looking at a square and the tile it was read from at the same time, and a
switch between the two makes you hold one in your head while you look at the other. The
board narrows to make room rather than being covered, so every square stays clickable
and the correction can be made while the picture is still up.

It was a tab first, and that was wrong twice over: the board did not actually go away
when the tab changed - `.board` sets `display: grid`, which beats the `hidden`
attribute - so the photograph was appended *below* a full-height board, on a page that
deliberately does not scroll. On a short window it could not be reached at all.

Two views:

- **as taken** - the picture the reader was given.
- **as the reader saw it** - the reader's own view, in three panels: where it decided
  the board's edges were, the straightened board with what it read written on it, and
  the ink it worked from. When a reading looks wrong, that middle panel usually says
  why, and the original never will.

Click the picture to swap between fitted and full size, with the well scrolling under
it. Fitted is right for a photograph of a whole board; it is wrong for the reader's own
view, which is three panels side by side and ends up a third as tall as it is wide -
the panel that explains the mistake is exactly the one that gets too small to read.

Below 900px there is no room for two columns, so the photograph goes under the board
instead - which is fine there, because at that width the page scrolls again.

Photographs are held in memory, not in the document. A picture is a megabyte and a game
is twenty turns, so writing them into the JSON would make a file no editor will open,
for something whose purpose is to be glanced at while a reading is checked. They last
as long as the page is open, which is as long as the checking takes.

### When a scan is wrong

It often will be, and the panels are there to say so. A reading that is impossible -
more blanks than exist, two Zs - fails the Distribution check outright, with no
confidence score needed. Cells the reader doubts are ringed and marked. A registration
the reader is unsure of is reported after the scan, and is usually worth re-taking the
photograph for.

Nothing about a scan is final: it fills the turn in, and then you correct it by hand.
The whole point of scanning into a *new* turn is that only the handful of new tiles
need checking, rather than the whole board.

## On a phone

The camera is on a phone, so the page that takes the photograph has to work on the
device taking it. Below 900px the rail moves under the board instead of beside it;
below 720px the turn strip and the keyboard sheet go away - neither means anything
without a keyboard or a wide window - and every target grows to 40px. The sign-in
page drops its picture and becomes the form.

Before this there were no width rules at all: one `@media` block, for dark mode.

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
style.css       all styling; every colour, face and measurement is a custom property
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
