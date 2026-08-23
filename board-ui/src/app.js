// Wiring. The only module that knows about all the others.

import * as state from './state.js';
import * as board from './board.js';
import * as editor from './editor.js';
import * as turns from './turns.js';
import * as io from './io.js';
import { pool } from './pool.js';
import { checks } from './validate.js';
import { ROTATIONS, TILES } from './constants.js';

const $ = id => document.getElementById(id);

// -- start up ----------------------------------------------------------------

board.build($('board'), (dr, dc) => {
  // The board hands back the square that was clicked *on screen*; turn it into the
  // square that square really is before anything else sees it.
  const { row, col } = board.toCanonical(dr, dc, state.get().rotation);
  editor.select(row, col);
});

editor.init(render, turns.canEdit);

// Read what was saved *before* anything is loaded, and before the autosave
// subscription exists. Loading a document triggers a save, so reading afterwards
// returns the blank document that has just overwritten the real one - which meant
// every page load quietly destroyed the previous session and then offered to
// restore the wreckage.
const previous = io.saved();

state.subscribe(() => { io.save(); render(); });

const fromLink = io.fromQueryString();
if (fromLink) {
  state.load(fromLink);
} else {
  state.load(state.createDocument('untitled'));
  if (previous) {
    $('restore').hidden = false;
    $('restore-yes').onclick = () => { state.load(previous); $('restore').hidden = true; };
    $('restore-no').onclick = () => { io.clearSaved(); $('restore').hidden = true; };
  }
}

// -- render ------------------------------------------------------------------

function render() {
  const doc = state.get();
  const turn = state.currentTurn();

  board.render(doc, editor.view);
  turns.render($('turn-strip'), i => turns.goTo(i));

  $('turn-n').textContent = doc.currentTurn + 1;
  $('turn-total').textContent = doc.turns.length;
  $('turn-who').textContent = (doc.players[turn.player] || 'Player ' + turn.player)
    + ' · ' + turn.annotation;
  const added = state.newThisTurn().size;
  $('turn-delta').textContent = added + ' new tile' + (added === 1 ? '' : 's') + ' this turn';
  $('annotation').value = turn.annotation;
  if ($('note').value !== turn.note) $('note').value = turn.note;

  $('rotation').textContent = doc.rotation + '°';
  $('direction').textContent = editor.view.direction === 'horizontal' ? 'across' : 'down';
  $('blank-mode').setAttribute('aria-pressed', String(editor.view.blankMode));
  $('blank-mode').classList.toggle('on', editor.view.blankMode);
  $('cursor-label').textContent = editor.view.selected
    ? state.cellLabel(editor.view.selected.row, editor.view.selected.col)
    : 'no square selected';

  const locked = !turns.canEdit();
  $('lock-note').hidden = !locked;
  document.body.classList.toggle('locked', locked);

  $('undo').disabled = !state.canUndo();
  $('redo').disabled = !state.canRedo();
  $('del-turn').disabled = doc.turns.length <= 1;

  renderPool();
  renderChecks();
}

function renderPool() {
  const p = pool();
  $('pool-left').textContent = p.left + ' left';
  const host = $('pool');
  host.innerHTML = '';
  for (const l of p.letters) {
    const el = document.createElement('span');
    el.className = 'pool-letter' + (l.over ? ' over' : '') + (l.left === 0 ? ' empty' : '');
    // Dots read faster than numbers at a glance, and the count is there for when the
    // glance is not enough.
    el.innerHTML = '<b>' + l.letter + '</b>' + '·'.repeat(Math.max(0, l.left));
    el.title = l.used + ' of ' + l.total + ' placed';
    host.appendChild(el);
  }
  const blanks = document.createElement('span');
  blanks.className = 'pool-letter' + (p.blanks.over ? ' over' : '');
  blanks.innerHTML = '<b>?</b>' + '·'.repeat(Math.max(0, p.blanks.left));
  blanks.title = p.blanks.used + ' of ' + p.blanks.total + ' blanks placed';
  host.appendChild(blanks);

  const errors = p.errors.map(e => e.used + ' × ' + e.letter + ' placed, max ' + e.total);
  if (p.unknown) errors.push(p.unknown + ' tile(s) of unknown letter, not charged to any letter');
  $('pool-errors').hidden = errors.length === 0;
  $('pool-errors').textContent = errors.join(' · ');
}

function renderChecks() {
  const host = $('checks');
  host.innerHTML = '';
  for (const c of checks()) {
    const li = document.createElement('li');
    const mark = c.severity === 'info' ? 'i' : (c.ok ? '✓' : (c.severity === 'fail' ? '✗' : '!'));
    li.className = c.severity === 'info' ? 'info' : (c.ok ? 'pass' : c.severity);
    li.innerHTML = '<span class="mark">' + mark + '</span> ' + c.label
      + (c.detail ? ' <span class="muted">' + c.detail + '</span>' : '');
    host.appendChild(li);
  }
}

// -- controls ----------------------------------------------------------------

$('rotate').onclick = () => {
  const at = ROTATIONS.indexOf(state.get().rotation);
  state.setRotation(ROTATIONS[(at + 1) % ROTATIONS.length]);
};
$('flip').onclick = () => {
  editor.view.direction = editor.view.direction === 'horizontal' ? 'vertical' : 'horizontal';
  render();
};
$('blank-mode').onclick = () => editor.setBlankMode(!editor.view.blankMode);
$('clear-cell').onclick = () => {
  const s = editor.view.selected;
  if (s && turns.canEdit()) state.clearCell(s.row, s.col);
};
$('unlock').onclick = () => { turns.unlock(); render(); };

$('annotation').onchange = e => state.setAnnotation(e.target.value);
$('note').oninput = e => state.setNote(e.target.value);

$('first').onclick = turns.first;
$('prev').onclick = turns.prev;
$('next').onclick = turns.next;
$('last').onclick = turns.last;
$('add-turn').onclick = turns.add;
$('del-turn').onclick = () => {
  if (confirm('Delete the last turn? Its tiles cannot be recovered from the board.')) turns.removeLast();
};
$('undo').onclick = () => state.undo();
$('redo').onclick = () => state.redo();
$('export').onclick = () => io.download();

$('load-sample').onclick = async () => {
  try {
    state.load(await io.loadFixture('fixtures/sample-game.json'));
  } catch (err) {
    alert(err.message);
  }
};

document.addEventListener('keydown', e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === 'Home') { e.preventDefault(); turns.first(); }
  if (e.key === 'End') { e.preventDefault(); turns.last(); }
});

// -- import ------------------------------------------------------------------

const dialog = $('import-dialog');
$('import').onclick = () => {
  $('import-problems').hidden = true;
  $('import-text').value = '';
  dialog.showModal();
};

function tryImport(text) {
  const result = io.importJSON(text);
  if (result.ok) {
    dialog.close();
  } else {
    // Say what was wrong. A file that fails silently, or half-loads, looks exactly
    // like a bug in the vision pipeline.
    $('import-problems').hidden = false;
    $('import-problems').textContent = result.problems.slice(0, 6).join(' · ');
  }
}

$('import-go').onclick = () => tryImport($('import-text').value);
$('import-file').onchange = e => {
  const file = e.target.files[0];
  if (file) file.text().then(tryImport);
};

const drop = $('drop');
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) file.text().then(tryImport);
});

render();
