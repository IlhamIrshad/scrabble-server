// What is left in the bag and on racks, derived from the board alone.

import { TILES, BLANK_COUNT, TOTAL_TILES } from './constants.js';
import * as state from './state.js';

/**
 * Count what the displayed turn has on the board, and subtract it from a full set.
 *
 * Deliberately not split into "bag" and "racks". Without rack data there is no way to
 * know which is which, and a guess would read as information.
 */
export function pool(turn = state.currentTurn()) {
  const placed = {};
  let blanks = 0;
  let unknown = 0;

  for (const c of turn.cells) {
    if (c.isBlank) { blanks++; continue; }
    // A tile whose letter was not readable still counts against the hundred - the
    // tile is demonstrably on the board - but it cannot be charged to any letter.
    if (c.letter == null) { unknown++; continue; }
    placed[c.letter] = (placed[c.letter] || 0) + 1;
  }

  const letters = Object.keys(TILES).map(ch => {
    const total = TILES[ch][0];
    const used = placed[ch] || 0;
    return { letter: ch, total, used, left: total - used, over: used > total };
  });

  const onBoard = turn.cells.length;
  return {
    letters,
    blanks: { total: BLANK_COUNT, used: blanks, left: BLANK_COUNT - blanks, over: blanks > BLANK_COUNT },
    unknown,
    onBoard,
    left: TOTAL_TILES - onBoard,
    errors: letters.filter(l => l.over).concat(blanks > BLANK_COUNT ? [{ letter: 'blank', total: BLANK_COUNT, used: blanks }] : []),
  };
}
