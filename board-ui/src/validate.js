// Cheap structural checks, run on every change.
//
// These are the same constraints the vision pipeline should eventually apply to its
// own output. They cost almost nothing and catch a large share of both misreads and
// typos, because several of them are *provable* - there is exactly one Z, so a board
// showing two is wrong with no confidence score required.

import { SIZE, CENTRE, TOTAL_TILES, FLAG_BELOW } from './constants.js';
import * as state from './state.js';
import { pool } from './pool.js';

export function checks() {
  const turn = state.currentTurn();
  const grid = state.grid(turn);
  const out = [];
  const tiles = turn.cells.length;

  const centre = grid[CENTRE.row][CENTRE.col];
  out.push({
    label: 'Centre covered',
    ok: tiles === 0 || !!centre,
    severity: 'fail',
    detail: tiles === 0 ? 'board empty' : (centre ? '' : 'H8 is empty'),
  });

  const loose = disconnected(grid, tiles);
  out.push({
    label: 'Connectivity',
    ok: loose === 0,
    // A warning rather than a failure, against the specification, and for a reason
    // from the photographs: players leave unplayed tiles sitting on the board after a
    // game, unconnected to anything. The vision pipeline's own connectivity check was
    // removed for exactly this - it fired constantly on boards that were read
    // perfectly. A real disconnection is worth surfacing; it is not worth crying wolf.
    severity: 'warn',
    detail: loose ? loose + ' tile(s) not joined to the centre group' : '',
  });

  const p = pool(turn);
  out.push({
    label: 'Distribution',
    ok: p.errors.length === 0,
    severity: 'fail',
    detail: p.errors.map(e => e.used + ' x ' + e.letter + ', max ' + e.total).join('; '),
  });

  out.push({
    label: 'Tile total',
    ok: tiles <= TOTAL_TILES,
    severity: 'fail',
    detail: tiles > TOTAL_TILES ? tiles + ' tiles placed' : tiles + ' of 100 placed',
  });

  const prev = state.previousTurn();
  const removed = prev ? prev.cells.filter(c => !state.cellAt(c.row, c.col, turn)).length : 0;
  out.push({
    label: 'Turn monotonic',
    ok: removed === 0,
    // Warn, never fail: a challenged phony comes back off the board, and that is a
    // legal thing for a game to do.
    severity: 'warn',
    detail: removed ? removed + ' tile(s) removed since the previous turn' : '',
  });

  const added = state.newThisTurn().size;
  const expected = turn.annotation === 'play';
  out.push({
    label: 'Turn size',
    ok: !expected || (added >= 1 && added <= 7) || state.get().currentTurn === 0 && added === 0,
    severity: 'warn',
    detail: added + ' new tile(s) this turn',
  });

  const flagged = turn.cells.filter(c => c.confidence != null && c.confidence < FLAG_BELOW).length;
  const nameless = turn.cells.filter(c => c.letter == null && !c.isBlank).length;
  out.push({
    label: 'Low confidence',
    ok: true,
    severity: 'info',
    detail: flagged + ' cell(s) below ' + FLAG_BELOW
      + (nameless ? ', ' + nameless + ' unreadable' : ''),
  });

  return out;
}

/** Tiles not orthogonally joined to whichever group holds the centre. */
function disconnected(grid, tiles) {
  if (!tiles) return 0;
  const start = grid[CENTRE.row][CENTRE.col] ? CENTRE : firstTile(grid);
  if (!start) return 0;
  const seen = new Set();
  const queue = [start];
  while (queue.length) {
    const { row, col } = queue.pop();
    const k = row + ',' + col;
    if (seen.has(k)) continue;
    if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) continue;
    if (!grid[row][col]) continue;
    seen.add(k);
    queue.push({ row: row + 1, col }, { row: row - 1, col },
               { row, col: col + 1 }, { row, col: col - 1 });
  }
  return tiles - seen.size;
}

function firstTile(grid) {
  for (let r = 0; r < SIZE; r++) {
    for (let c = 0; c < SIZE; c++) if (grid[r][c]) return { row: r, col: c };
  }
  return null;
}
