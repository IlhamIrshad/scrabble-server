// Renders the 15x15 grid and owns the cell DOM. Reads state; never writes it.

import { PREMIUM, SIZE, TILES, FLAG_BELOW } from './constants.js';
import * as state from './state.js';

const cells = [];           // [row][col] of display-position elements
let host = null;

/**
 * Where a stored square appears once the board has been turned, and back again.
 *
 * Rotation is a property of the *view*: the cells keep their true coordinates and only
 * the drawing is turned. That is what makes turning the board four times return it
 * exactly to where it started, and it keeps the document meaningful to anything that
 * reads it without knowing which way the camera was pointing.
 *
 * The premium squares are deliberately not rotated. That layout is perfectly 4-fold
 * symmetric, so rotating it changes nothing - and code that appears to rotate it
 * invites the reader to believe orientation could be recovered from it, which is
 * exactly the mistake this whole rotation control exists to work around.
 */
export function toDisplay(row, col, rotation) {
  let r = row, c = col;
  for (let i = 0; i < ((rotation / 90) | 0) % 4; i++) {
    const nr = c, nc = SIZE - 1 - r;
    r = nr; c = nc;
  }
  return { row: r, col: c };
}

export function toCanonical(row, col, rotation) {
  let r = row, c = col;
  for (let i = 0; i < ((rotation / 90) | 0) % 4; i++) {
    const nr = SIZE - 1 - c, nc = r;
    r = nr; c = nc;
  }
  return { row: r, col: c };
}

export function build(container, onCellClick) {
  host = container;
  host.innerHTML = '';
  host.style.setProperty('--size', SIZE);

  // Column letters across the top, row numbers down the side. Both are labels of the
  // *displayed* board, so they follow the rotation like everything else on screen.
  const corner = document.createElement('div');
  corner.className = 'gutter';
  host.appendChild(corner);
  for (let c = 0; c < SIZE; c++) {
    const el = document.createElement('div');
    el.className = 'gutter col-label';
    el.textContent = String.fromCharCode(65 + c);
    host.appendChild(el);
  }

  for (let r = 0; r < SIZE; r++) {
    cells[r] = [];
    const num = document.createElement('div');
    num.className = 'gutter row-label';
    num.textContent = r + 1;
    host.appendChild(num);
    for (let c = 0; c < SIZE; c++) {
      const el = document.createElement('div');
      el.className = 'cell';
      el.tabIndex = -1;
      el.dataset.row = r;
      el.dataset.col = c;
      el.addEventListener('click', () => onCellClick(r, c));
      cells[r][c] = el;
      host.appendChild(el);
    }
  }
}

export function render(doc, view) {
  const turn = state.currentTurn();
  const fresh = state.newThisTurn();
  const byPos = state.grid(turn);
  const rot = doc.rotation;

  for (let dr = 0; dr < SIZE; dr++) {
    for (let dc = 0; dc < SIZE; dc++) {
      const { row, col } = toCanonical(dr, dc, rot);
      draw(cells[dr][dc], byPos[row][col], PREMIUM[dr][dc],
           fresh.has(row + ',' + col),
           view.selected && view.selected.row === row && view.selected.col === col,
           view.direction);
    }
  }
}

function draw(el, cell, premium, isNew, isSelected, direction) {
  el.className = 'cell';
  el.innerHTML = '';
  el.removeAttribute('title');

  if (!cell) {
    if (premium !== '..') {
      el.classList.add('premium', 'p-' + (premium === 'ST' ? 'DW' : premium));
      const label = document.createElement('span');
      label.className = 'premium-label';
      label.textContent = premium === 'ST' ? '★' : premium;
      el.appendChild(label);
    }
  } else {
    el.classList.add('tile');
    el.classList.add(isNew ? 'is-new' : 'is-established');
    if (cell.isBlank) el.classList.add('is-blank');
    if (cell.source === 'manual') el.classList.add('is-manual');
    // Null confidence means "not from OCR", which is certain, so only a real number
    // below the bar counts as doubtful.
    if (cell.confidence != null && cell.confidence < FLAG_BELOW) el.classList.add('is-flagged');

    const face = document.createElement('span');
    face.className = 'letter';
    // A tile whose letter is not known still has to be visible as a tile. This is the
    // ordinary state of a blank straight off the camera, and of any tile the reader
    // could not name - never a reason to draw an empty square.
    face.textContent = cell.letter == null ? '·' : cell.letter;
    if (cell.letter == null) face.classList.add('unknown');
    el.appendChild(face);

    const value = document.createElement('span');
    value.className = 'value';
    value.textContent = cell.isBlank ? '0' : (cell.letter && TILES[cell.letter] ? TILES[cell.letter][1] : '');
    el.appendChild(value);

    if (cell.source === 'manual') {
      const mark = document.createElement('span');
      mark.className = 'manual-mark';
      el.appendChild(mark);
    }
    if (cell.confidence != null && cell.confidence < FLAG_BELOW) {
      const warn = document.createElement('span');
      warn.className = 'flag-mark';
      warn.textContent = '!';
      el.appendChild(warn);
      el.title = 'read with confidence ' + cell.confidence.toFixed(2);
    }
  }

  if (isSelected) {
    el.classList.add('is-selected', direction === 'vertical' ? 'dir-v' : 'dir-h');
  }
}

export function focusCell(row, col, rotation) {
  const d = toDisplay(row, col, rotation);
  const el = cells[d.row] && cells[d.row][d.col];
  if (el) el.focus({ preventScroll: true });
}
