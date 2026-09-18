// The edit cursor and keyboard handling. Writes go through state.js only.
//
// The interaction is the crossword one, because it is well proven and because the
// point of manual entry is typing a whole board quickly rather than correcting one
// square. Click the first square of a word, type the word.

import { SIZE } from './constants.js';
import * as state from './state.js';

export const view = {
  selected: null,           // canonical {row, col}, or null
  direction: 'horizontal',
  blankMode: false,         // the touch equivalent of holding shift
};

let onChange = () => {};
// Injected so this module needs to know nothing about turns; app.js wires it up.
let canEdit = () => true;

export function init(notify, editable) {
  onChange = notify;
  if (editable) canEdit = editable;
  document.addEventListener('keydown', onKey);
}

/** One square along, or null at the edge. The board is never drawn turned. */
function step(sel, direction, back = false) {
  const delta = back ? -1 : 1;
  const row = direction === 'vertical' ? sel.row + delta : sel.row;
  const col = direction === 'horizontal' ? sel.col + delta : sel.col;
  if (row < 0 || row >= SIZE || col < 0 || col >= SIZE) return null;
  return { row, col };
}

/** The next square in this direction that has no tile on it. */
function nextFree(sel, direction) {
  let at = sel;
  while (at && state.cellAt(at.row, at.col)) at = step(at, direction);
  return at;
}

export function select(row, col) {
  if (view.selected && view.selected.row === row && view.selected.col === col) {
    // Clicking the square you are already on flips the direction, as crosswords do.
    view.direction = view.direction === 'horizontal' ? 'vertical' : 'horizontal';
  } else {
    view.selected = { row, col };
  }
  onChange();
}

export function deselect() {
  view.selected = null;
  onChange();
}

export function setBlankMode(on) {
  view.blankMode = on;
  onChange();
}

function place(letter, isBlank) {
  if (!canEdit()) return;
  // Land on a free square first. Typing across a word that is already on the board
  // steps over its letters rather than overwriting them - correcting a tile is a
  // deliberate act, done with Delete or Backspace.
  const target = nextFree(view.selected, view.direction);
  if (!target) return;
  state.setCell(target.row, target.col, letter, isBlank);
  view.selected = nextFree(step(target, view.direction), view.direction) || target;
  onChange();
}

function onKey(e) {
  const typing = e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName);
  const meta = e.ctrlKey || e.metaKey;

  if (meta && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    if (e.shiftKey) state.redo(); else state.undo();
    return;
  }
  if (typing) return;

  if (e.key === 'Escape') { deselect(); return; }
  if (!view.selected) return;
  const sel = view.selected;

  if (e.key === 'Tab') {
    e.preventDefault();
    view.direction = view.direction === 'horizontal' ? 'vertical' : 'horizontal';
    onChange();
    return;
  }

  if (/^Arrow(Up|Down|Left|Right)$/.test(e.key)) {
    e.preventDefault();
    const dir = (e.key === 'ArrowUp' || e.key === 'ArrowDown') ? 'vertical' : 'horizontal';
    const back = e.key === 'ArrowUp' || e.key === 'ArrowLeft';
    const next = step(sel, dir, back);
    if (next) { view.selected = next; onChange(); }
    return;
  }

  if (e.key === 'Backspace') {
    e.preventDefault();
    if (!canEdit()) return;
    // Clear where we are if there is something here, otherwise step back and clear
    // that - which is what makes holding backspace walk a mistyped word off the board.
    if (state.cellAt(sel.row, sel.col)) {
      state.clearCell(sel.row, sel.col);
    } else {
      const back = step(sel, view.direction, true);
      if (back) { view.selected = back; state.clearCell(back.row, back.col); }
    }
    onChange();
    return;
  }

  if (e.key === 'Delete') {
    e.preventDefault();
    if (!canEdit()) return;
    state.clearCell(sel.row, sel.col);
    onChange();
    return;
  }

  if (/^[a-zA-Z]$/.test(e.key)) {
    e.preventDefault();
    // Shift places a blank designated as that letter. It needs its own gesture
    // because a blank is a letter *and* a flag, and retrofitting that later would
    // change the shape of every write.
    place(e.key.toUpperCase(), e.shiftKey || view.blankMode);
    return;
  }
}
