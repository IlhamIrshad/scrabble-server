// Renders the 15x15 grid and owns the cell DOM. Reads state; never writes it.

import { PREMIUM, SIZE, TILES, FLAG_BELOW } from './constants.js';
import * as state from './state.js';

const cells = [];           // [row][col] of display-position elements
let host = null;

/**
 * The virtual board is always drawn straight, and there is no view rotation.
 *
 * There was, and it was a mistake worth recording. Turning the *display* moves the
 * squares while leaving every letter upright, so words run backwards and upwards and
 * the board becomes unreadable - it looks turned without being usable, which is the
 * worst of both.
 *
 * Rotation is a property of the *photograph*, not of the board. A camera may see the
 * board any way round, so a scan is turned to match what is already known before its
 * cells are ever stored (see `scan.js`). By the time anything is drawn, the question
 * has been settled, and the board on screen is the board as it will be recorded.
 */

export function build(container, onCellClick) {
  host = container;
  host.innerHTML = '';
  host.style.setProperty('--size', SIZE);

  // Column letters across the top, row numbers down the side.
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

  for (let row = 0; row < SIZE; row++) {
    for (let col = 0; col < SIZE; col++) {
      draw(cells[row][col], byPos[row][col], PREMIUM[row][col],
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
    }
    el.title = describe(cell);
  }

  if (isSelected) {
    el.classList.add('is-selected', direction === 'vertical' ? 'dir-v' : 'dir-h');
  }
}

/**
 * What the reader made of this square, in words, for the hover.
 *
 * Worth spelling out rather than showing a bare number. "0.62" means nothing on its
 * own; whether that is below the bar this board is being judged by is the thing an
 * operator actually needs, and so is the difference between a tile the reader named
 * doubtfully and one it could not name at all.
 */
function describe(cell) {
  const where = String.fromCharCode(65 + cell.col) + (cell.row + 1);
  const parts = [where];

  if (cell.isBlank) parts.push(cell.letter ? 'blank played as ' + cell.letter : 'blank, letter unknown');
  else if (cell.letter == null) parts.push('a tile, but no letter could be read');
  else parts.push('read as ' + cell.letter + ' (' + (TILES[cell.letter] ? TILES[cell.letter][1] : '?') + ' points)');

  if (cell.confidence == null) {
    parts.push(cell.source === 'manual' ? 'typed in by hand' : 'not from the reader');
  } else if (cell.confidence < FLAG_BELOW) {
    parts.push('confidence ' + cell.confidence.toFixed(2) + ' - below ' + FLAG_BELOW + ', worth checking');
  } else {
    parts.push('confidence ' + cell.confidence.toFixed(2));
  }
  return parts.join(' · ');
}

export function focusCell(row, col) {
  const el = cells[row] && cells[row][col];
  if (el) el.focus({ preventScroll: true });
}
