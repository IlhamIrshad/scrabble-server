// Wiring. The only module that knows about all the others.

import * as state from './state.js';
import * as board from './board.js';
import * as editor from './editor.js';
import * as turns from './turns.js';
import * as io from './io.js';
import { pool } from './pool.js';
import { checks } from './validate.js';
import { TILES } from './constants.js';
import * as scan from './scan.js';

const $ = id => document.getElementById(id);

// -- the photograph behind a turn ---------------------------------------------
//
// Held in memory rather than in the document. A photograph is a megabyte and a game is
// twenty turns, so writing them into the JSON would turn a readable file into
// something no editor will open, for a picture whose whole purpose is to be glanced at
// while the reading is being checked. They last as long as the page does, which is as
// long as the checking does.

const frames = new Map();          // turn index -> { photoUrl, debugUrl, name }
let tab = 'board';

function rememberFrame(index, result) {
  const old = frames.get(index);
  if (old && old.photoUrl) URL.revokeObjectURL(old.photoUrl);
  frames.set(index, {
    photoUrl: result.photo ? URL.createObjectURL(result.photo) : null,
    debugUrl: result.debugUrl || null,
    name: (result.photo && result.photo.name) || 'frame',
  });
}

function showTab(which) {
  tab = which;
  $('tab-board').classList.toggle('on', which === 'board');
  $('tab-photo').classList.toggle('on', which === 'photo');
  $('tab-board').setAttribute('aria-selected', String(which === 'board'));
  $('tab-photo').setAttribute('aria-selected', String(which === 'photo'));
  $('board').hidden = which !== 'board';
  $('photo-pane').hidden = which !== 'photo';
  renderPhoto();
}

function renderPhoto() {
  if (tab !== 'photo') return;
  const frame = frames.get(state.get().currentTurn);
  const kind = document.querySelector('input[name="photo-kind"]:checked').value;
  const url = frame && (kind === 'debug' ? frame.debugUrl : frame.photoUrl);
  $('photo').hidden = !url;
  $('photo-none').hidden = !!url;
  if (url) $('photo').src = url;
  if (!frame) {
    $('photo-caption').textContent = '';
  } else if (kind === 'debug' && !frame.debugUrl) {
    $('photo-none').textContent = 'The reader did not return its own view for this scan.';
    $('photo-caption').textContent = '';
  } else {
    $('photo-none').textContent = 'No photograph for this turn. Scans keep theirs for as '
      + 'long as the page is open; typed-in turns never had one.';
    $('photo-caption').textContent = kind === 'debug'
      ? frame.name + ' — left: where the board was found · middle: straightened, with what was read · right: the ink it worked from'
      : frame.name + ' — the photograph as taken';
  }
}

$('tab-board').onclick = () => showTab('board');
$('tab-photo').onclick = () => showTab('photo');
for (const radio of document.querySelectorAll('input[name="photo-kind"]')) {
  radio.onchange = renderPhoto;
}

// -- start up ----------------------------------------------------------------

// The board is drawn straight, so the square clicked is the square meant.
board.build($('board'), (row, col) => editor.select(row, col));

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
  renderPhoto();
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

/**
 * Turn this turn's tiles a quarter turn.
 *
 * Not a view control - the board is always drawn straight. This corrects a *scan* that
 * came in the wrong way round, which happens on an opening photograph, where there are
 * no earlier tiles for the automatic matching to work from. It rewrites the cells, so
 * it is an edit like any other and undoes like one.
 */
$('turn-scan').onclick = () => {
  if (!turns.canEdit()) return;
  const turned = scan.rotateCells(state.currentTurn().cells, 1);
  state.replaceCells(turned, 'turned by hand');
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

$('new-game').onclick = () => {
  if (state.currentTurn().cells.length &&
      !confirm('Start a new game? The current one is lost unless you have exported it.')) return;
  const name = prompt('Name for this game', 'game-' + new Date().toISOString().slice(0, 10));
  if (name === null) return;
  state.load(state.createDocument(name || 'untitled'));
  say('new game started - scan or type the opening play');
};

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

// -- scanning ----------------------------------------------------------------
//
// A frame goes to the reader, the reader answers with a board, and the board fills the
// turn. Where the frame comes from - a photograph now, a webcam later, a camera on
// another machine after that - changes only the first of those three.

$('api-base').value = scan.apiBase();
$('api-base').onchange = e => { scan.setApiBase(e.target.value.trim()); checkReader(); };

async function checkReader() {
  const el = $('api-state');
  el.className = 'muted';
  el.textContent = 'checking…';
  try {
    const h = await scan.health();
    el.className = 'up';
    el.textContent = 'reader ready · ' + h.templates.letters + ' letters';
  } catch (err) {
    el.className = 'down';
    // Say which address failed. "Failed to fetch" on its own sends people looking in
    // the wrong place, and the usual cause is simply that the reader is not running.
    el.textContent = 'no reader at ' + scan.apiBase();
  }
}
checkReader();

async function runScan(intoNewTurn) {
  // A new document already has one empty turn waiting, so the first scan of a game
  // fills it rather than pushing it aside. Otherwise every game began with a blank
  // turn 1 that nothing could ever be put in, and stepping back to it showed an empty
  // board that looked like a fault.
  const atStart = state.get().turns.length === 1 && state.currentTurn().cells.length === 0;
  if (intoNewTurn && !atStart) turns.add();
  if (!turns.canEdit()) {
    say('that turn is locked - press "edit this turn" first', true);
    return;
  }
  const blob = await scan.nextFrame();
  if (!blob) return;

  say('reading ' + blob.name + '…');
  setBusy(true);
  try {
    const r = await scan.scanIntoCurrentTurn(blob);
    rememberFrame(state.get().currentTurn, r);
    const parts = [r.tiles + ' tiles'];
    if (r.blanks) parts.push(r.blanks + ' blank');
    if (r.unnamed) parts.push(r.unnamed + ' unreadable');
    parts.push(r.flagged + ' to check');
    if (r.rotated) parts.push('board turned ' + r.rotated + '° to match the last turn');
    if (r.registration && r.registration.confident === false) {
      parts.push('registration doubtful - consider re-taking the photograph');
    }
    say(parts.join(' · '), r.registration && r.registration.confident === false);
  } catch (err) {
    // A board the reader cannot find is an ordinary outcome, not a crash: bad light, a
    // hand across the board, a photograph taken too obliquely. Say so and leave the
    // turn alone so the operator can type it in instead.
    say('could not read it: ' + err.message, true);
  } finally {
    setBusy(false);
  }
}

function say(text, bad = false) {
  const el = $('scan-result');
  el.textContent = text;
  el.className = bad ? 'bad' : 'muted';
}

function setBusy(on) {
  $('scan-new').disabled = on;
  $('scan-here').disabled = on;
  $('scan-result').classList.toggle('busy', on);
}

$('scan-new').onclick = () => runScan(true);
$('scan-here').onclick = () => runScan(false);

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
