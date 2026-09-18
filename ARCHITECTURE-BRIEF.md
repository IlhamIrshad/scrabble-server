# Scrabble board reader - architecture brief

A short brief for someone giving architectural advice. It describes what exists, what
is decided, and the questions that are actually open. Detail on deferred work is in
`BACKLOG.md`.

## What it does

Turns a photograph of a Scrabble board into the board state: which squares hold tiles,
which letter is on each, and which are blanks. About 6,000 lines of Python -
FastAPI, OpenCV and NumPy, no machine-learning model and no training step. Everything
is classical computer vision plus template matching against reference glyphs.

**Current accuracy**, measured against boards a person checked cell by cell:

| | |
|---|---|
| target board (31 photographs, 6,975 cells) | 98.996% of cells correct |
| all board types (55 photographs, 12,375 cells) | 98.343% |
| registration judged correct by eye | 89 of 161 photographs |
| latency, 4K photograph | 4.5 s (1.2 s find board, 3.25 s read, 0.08 s classify) |

## Where it is going

The point is not board states. It is **GCG files** - the standard Scrabble game-record
format that Quackle and Macondo analyse. The intended flow:

1. A camera watches the board.
2. A **clock press** triggers a scan - the one moment the board is stable and complete.
3. The new board is **diffed** against the previous turn to derive the play.
4. That play is appended to a game record.

A **second camera on the rack** is planned. A board camera alone can supply the
coordinate, the word, the direction, the score and the running total, but never the
player's rack, and it cannot tell an exchange from a pass - both are a clock press with
no board change. Those are exactly the fields analysis tools need.

## How it is built now

```
photo -> registration -> warp + grid lock -> per-cell colour -> glyph -> template match
         (find the         (rectify to        (tile or bare      (extract    (letter)
          board quad)       15x15, lock        board?)            the ink)
                            onto printed
                            grid lines)
```

- `app/main.py` - FastAPI. `POST /detect` takes an image and returns board state;
  `/ws` takes a JPEG frame and returns the same, intended for an embedded camera.
- `app/vision/registration.py` - proposes candidate board quads several independent
  ways (colour clustering, texture, edges, boundary lines), scores them, and warps.
- `app/vision/grid.py` - splits the rectified board into cells, measures the tile and
  board face colours *from the photograph itself*, and extracts each letter.
- `app/vision/classify.py` - matches each glyph against reference shapes taken from
  real tile artwork, with additional evidence from enclosed holes and the number of
  digits in the printed point value.
- `app/config.py` - 99 tunables, each carrying the measurement that set it.

Two properties worth knowing, because they shaped everything:

- **Nothing is hard-coded about a particular board.** Tile colour, board colour, cell
  pitch and even how many holes each letter should enclose are measured per photograph
  or derived from the loaded templates. This was hard-won; the pipeline originally only
  worked on one board and one tile set.
- **Confidence is honest.** Roughly 17% of tiles are flagged for review, and a misread
  that arrives *flagged* is treated as far cheaper than one that arrives confident. An
  invisible error is considered worse than a visible one throughout.

Deployment target is a container (Dokploy), hence `opencv-python-headless`.

## The architectural crux

**The service today is stateless and per-frame. Everything ahead is stateful and
per-game.** That is the change worth advice on. Specifically:

- Deriving a play requires the *previous* board, so there must be a game session and a
  turn log. Neither exists.
- **Board orientation depends on it too.** The standard premium-square layout is
  perfectly 4-fold rotationally symmetric, so a board turned 90 degrees registers
  cleanly, reads every tile, passes every check, and produces a transcript with every
  coordinate wrong. The board is rotated between turns in real play. The intended fix
  is to try all four rotations and keep the one matching the previously known tiles -
  which again requires prior state.
- **Diffing collapses the human review burden**, which is what makes the whole thing
  practical: per turn an operator confirms ~7 new tiles rather than 95. It also means a
  tile confirmed once is known for the rest of the game, and a tile played early is
  re-observed 20+ times, so a modal reading per cell across a game should beat any
  single scan comfortably.

## Open questions

1. **Where should the game state live?** Current thinking is to store per-turn board
   states as the source of truth and generate GCG as an export, since converting states
   to moves is where the ambiguity lives, and a mis-derived move could then be
   re-derived without re-photographing a finished game. Is that the right split?
2. **What owns the turn logic** - rotation resolution, diffing, scoring? The vision
   service, a separate game service, or the client?
3. **Where does the operator sit?** Confirming ~7 tiles per turn needs a UI with low
   latency, during a timed game.
4. **How much belongs on the edge?** An embedded camera streaming frames to a server is
   simple, but a network drop mid-game is unacceptable in a tournament.
5. **How should a second camera be integrated later** so that rack observations attach
   to turns already recorded, without rewriting the log?
6. **Scoring and rules** - implement, or lean on an existing engine? The premium-square
   layout is already known to the code, so scoring a play is largely free; validating
   one against a lexicon is a separate matter.

## Constraints

- Photographs of real games are **never** committed: they contain people and score
  sheets with handwritten names. `photos/` and `fixtures/` are gitignored, and all
  ground truth lives there.
- Reference tile artwork is read from a local NASPA Zyzzyva install and never
  redistributed; a build step bakes shippable templates.
- Accuracy figures assume roughly 72 pixels per board cell, so the board must span
  about 1,080 pixels in the source image. That is a real constraint on camera choice.
