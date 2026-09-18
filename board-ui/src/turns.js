// The turn strip, navigation, and the lock that keeps history read-only.

import * as state from './state.js';

// Whether the operator has explicitly unlocked a past turn. Reset whenever the
// displayed turn changes, so unlocking is never sticky.
let unlocked = false;

export function isLast() {
  const d = state.get();
  return d.currentTurn === d.turns.length - 1;
}

/**
 * Past turns are read-only until deliberately unlocked.
 *
 * Amending history is a real hazard once turns are chained - a change to an early turn
 * can silently invalidate everything after it - and the habit is much cheaper to
 * establish now than to unlearn later. One button is the whole cost.
 */
export function canEdit() {
  return isLast() || unlocked;
}

export function unlock() {
  unlocked = true;
}

export function goTo(index) {
  unlocked = false;
  state.goToTurn(index);
}

export function next() {
  goTo(state.get().currentTurn + 1);
}

export function prev() {
  goTo(state.get().currentTurn - 1);
}

export function first() {
  goTo(0);
}

export function last() {
  goTo(state.get().turns.length - 1);
}

export function add() {
  unlocked = false;
  state.addTurn();
}

export function removeLast() {
  unlocked = false;
  state.deleteLastTurn();
}

export function render(strip, onSelect) {
  const d = state.get();
  strip.innerHTML = '';
  d.turns.forEach((t, i) => {
    const chip = document.createElement('button');
    chip.className = 'chip' + (i === d.currentTurn ? ' current' : '');
    chip.textContent = i + 1;
    chip.title = (d.players[t.player] || 'Player ' + t.player) + ' · ' + t.annotation;
    chip.addEventListener('click', () => onSelect(i));
    strip.appendChild(chip);
  });
}
