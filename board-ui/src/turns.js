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
    // A turn still holding tiles nobody has confirmed carries a mark, so the strip says
    // where the work is left without opening every turn to find out.
    const waiting = state.doubtfulIn(t);
    chip.className = 'chip'
      + (i === d.currentTurn ? ' current' : '')
      + (waiting ? ' doubt' : '');
    chip.textContent = i + 1;
    chip.title = (d.players[t.player] || 'Player ' + t.player) + ' · ' + t.annotation
      + (waiting ? ' · ' + waiting + ' tile' + (waiting === 1 ? '' : 's') + ' still to confirm' : '');
    chip.addEventListener('click', () => onSelect(i));
    strip.appendChild(chip);
  });

  // Twenty turns overflow the strip, and the chip you are on is the one that has to be
  // visible. Nearest rather than centred, so it only scrolls when it has to.
  const current = strip.children[d.currentTurn];
  if (current) current.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}
