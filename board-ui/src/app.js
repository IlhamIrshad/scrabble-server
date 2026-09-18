// Wiring. The only module that knows about all the others.

import * as state from './state.js';
import * as board from './board.js';
import * as editor from './editor.js';
import * as turns from './turns.js';
import * as io from './io.js';
import { pool } from './pool.js';
import { checks } from './validate.js';
import * as scan from './scan.js';
import * as api from './api.js';

const $ = id => document.getElementById(id);

const svg = (paths, stroke = 'currentColor') =>
  '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="' + stroke + '" '
  + 'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + paths + '</svg>';

const TICK = svg('<polyline points="20 6 9 17 4 12"/>');
const ALERT = svg('<circle cx="12" cy="12" r="9"/><line x1="12" y1="8" x2="12" y2="13"/>'
  + '<line x1="12" y1="16.5" x2="12.01" y2="16.5"/>');
const CAMERA = svg('<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>');

// -- the photograph behind a turn ---------------------------------------------
//
// Held in memory rather than in the document. A photograph is a megabyte and a game is
// twenty turns, so writing them into the JSON would turn a readable file into
// something no editor will open, for a picture whose whole purpose is to be glanced at
// while the reading is being checked. They last as long as the page does, which is as
// long as the checking does.

const frames = new Map();          // turn index -> { photoUrl, debugUrl, name }
let tab = 'board';
// A scan writes to state, which re-renders, which would otherwise re-enable the very
// buttons that started it while it is still running.
let busy = false;

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

// The reader's own view is the one that says *why* a reading is wrong, and the original
// never will - so the strip that reports a scan links straight to it.
$('see-reader').onclick = e => {
  e.preventDefault();
  const debug = document.querySelector('input[name="photo-kind"][value="debug"]');
  if (debug) debug.checked = true;
  showTab('photo');
};

// -- start up ----------------------------------------------------------------

// The board is drawn straight, so the square clicked is the square meant.
board.build($('board'), (row, col) => editor.select(row, col));

editor.init(render, turns.canEdit);

const gameId = new URLSearchParams(location.search).get('game');
if (!gameId) location.replace('./');
if (!(await api.signedIn())) location.replace('./login.html?next=' + encodeURIComponent(location.search));

/**
 * Save on every change, but not on every keystroke.
 *
 * Typing a word is seven changes in two seconds and one game worth writing, so the
 * write waits for the typing to stop. `saving` is a promise rather than a flag so that
 * leaving the page can wait for the write in flight instead of racing it.
 */
let saveTimer = null;
let saving = Promise.resolve();

function scheduleSave() {
  clearTimeout(saveTimer);
  setSaved('saving…');
  saveTimer = setTimeout(() => {
    saving = api.saveGame(gameId, state.get())
      .then(() => setSaved('saved', 'ok'))
      .catch(err => {
        if (err.unauthorised) { location.replace('./login.html'); return; }
        // Never silent. A board that has stopped saving looks exactly like one that is
        // saving, right up until the browser is closed.
        setSaved('not saved: ' + err.message, 'bad');
      });
  }, 600);
}

function setSaved(text, state = '') {
  $('saved-text').textContent = text;
  $('saved-state').className = 'pill' + (state ? ' ' + state : '');
}

addEventListener('beforeunload', e => {
  // A beacon was tried here and removed: sendBeacon cannot carry an Authorization
  // header, so it would have needed an unauthenticated write endpoint - a hole in the
  // door to save six hundred milliseconds. Warning is honest and costs nothing.
  if (saveTimer) {
    e.preventDefault();
    e.returnValue = '';
  }
});

state.subscribe(() => { render(); scheduleSave(); });

try {
  state.load(await api.getGame(gameId));
  setSaved('saved', 'ok');
} catch (err) {
  if (err.unauthorised) location.replace('./login.html');
  state.load(state.createDocument(gameId));
  setSaved('could not load this game: ' + err.message, 'bad');
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
  $('annotation').value = turn.annotation;
  if ($('note').value !== turn.note) $('note').value = turn.note;

  $('direction').textContent = editor.view.direction === 'horizontal' ? 'across' : 'down';
  $('blank-mode').setAttribute('aria-pressed', String(editor.view.blankMode));
  $('blank-mode').classList.toggle('on', editor.view.blankMode);
  $('cursor-label').textContent = editor.view.selected
    ? state.cellLabel(editor.view.selected.row, editor.view.selected.col)
    : '';

  const locked = !turns.canEdit();
  $('lock-note').hidden = !locked;
  document.body.classList.toggle('locked', locked);

  $('undo').disabled = !state.canUndo();
  $('redo').disabled = !state.canRedo();
  $('del-turn').disabled = doc.turns.length <= 1;
  $('see-reader').hidden = !(frames.get(doc.currentTurn) || {}).debugUrl;

  renderNewTiles(locked);
  renderPool();
  renderChecks();
  renderPhoto();
  $('game-name').textContent = doc.sessionId;
}

/**
 * The tiles this turn added, as chips, doubtful ones first in the eye if not in order.
 *
 * This is the panel the whole diffing argument is for. Per turn an operator confirms
 * about seven new tiles rather than ninety-five, and that is what makes recording a
 * game during a timed round practical at all - so it is the first thing on the rail and
 * the only card with a primary button in it.
 */
function renderNewTiles(locked) {
  const turn = state.currentTurn();
  const fresh = state.newThisTurn();
  const byPosition = (a, b) => a.row - b.row || a.col - b.col;

  const added = turn.cells.filter(c => fresh.has(c.row + ',' + c.col)).sort(byPosition);
  // Older tiles the reader was never sure about. They are still on this board, still
  // doubtful, and still counted by the games list, so leaving them off this card would
  // make that number unreachable from the page that is meant to clear it.
  const carried = turn.cells
    .filter(c => !fresh.has(c.row + ',' + c.col) && state.doubtful(c))
    .sort(byPosition);

  chips($('new-tiles'), added);
  chips($('older-tiles'), carried);
  $('older-label').hidden = carried.length === 0;
  $('older-label').textContent = carried.length
    + (carried.length === 1 ? ' tile' : ' tiles') + ' carried forward, still unconfirmed';

  const waiting = added.concat(carried);
  // Two different kinds of unfinished. A tile with no letter cannot be confirmed into
  // one - somebody has to type it - so the button must not claim to deal with it.
  const unnamed = waiting.filter(c => c.letter == null);
  const unsure = waiting.filter(c => c.letter != null && state.doubtful(c));
  const pending = waiting.filter(c => c.confidence != null);

  $('new-summary').textContent = added.length
    ? added.length + ' tile' + (added.length === 1 ? '' : 's')
    : 'nothing new';

  const hint = $('confirm-hint');
  const button = $('confirm-turn');
  const names = list => list.slice(0, 3).map(c => state.cellLabel(c.row, c.col));

  if (!waiting.length) {
    hint.textContent = state.get().currentTurn === 0
      ? 'Nothing on the board yet. Scan a photograph, or click a square and type.'
      : 'Nothing has been added since the previous turn — a pass, an exchange, or a turn not scanned yet.';
  } else if (unnamed.length) {
    const where = names(unnamed);
    hint.innerHTML = '<b>' + where.join('</b>, <b>') + '</b>'
      + (unnamed.length > where.length ? ' and ' + (unnamed.length - where.length) + ' more' : '')
      + (unnamed.length === 1 ? ' has' : ' have') + ' no letter — unreadable, or a blank, '
      + 'whose designation is not in the photograph at all. Click one and type it.';
  } else if (unsure.length) {
    const where = names(unsure);
    hint.innerHTML = 'The reader is unsure of <b>' + where.join('</b>, <b>') + '</b>'
      + (unsure.length > where.length ? ' and ' + (unsure.length - where.length) + ' more' : '')
      + '. Click one to go to it and correct it, or confirm what it read.';
  } else if (pending.length) {
    hint.textContent = 'All ' + pending.length + ' were read confidently. Confirming records '
      + 'that a person looked, which is what stops them being counted again.';
  } else {
    hint.textContent = 'Checked. Every tile here was typed in or confirmed by hand.';
  }

  button.textContent = pending.length ? 'Confirm all ' + pending.length : 'Confirmed';
  button.disabled = locked || busy || pending.length === 0;
  $('scan-here').disabled = locked || busy;
}

/** A row of coordinate-and-letter chips; clicking one goes to that square. */
function chips(host, cells) {
  host.innerHTML = '';
  for (const cell of cells) {
    const chip = document.createElement('button');
    chip.className = 'tile-chip'
      + (state.doubtful(cell) ? ' doubt' : '')
      + (cell.isBlank ? ' blank' : '');
    const where = state.cellLabel(cell.row, cell.col);
    chip.innerHTML = '<span class="at">' + where + '</span>'
      + '<span class="ch' + (cell.letter == null ? ' unknown' : '') + '">'
      + (cell.letter == null ? '·' : cell.letter) + '</span>';
    chip.title = where + (cell.isBlank ? ' · blank' : '')
      + (cell.confidence != null ? ' · confidence ' + cell.confidence.toFixed(2) : ' · certain');
    // The correction happens on the board, not in the rail. Listing the squares is only
    // useful if it takes you to them.
    chip.onclick = () => { showTab('board'); editor.select(cell.row, cell.col); };
    host.appendChild(chip);
  }
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
    li.innerHTML = '<span class="mark">' + mark + '</span>'
      + '<span class="what">' + c.label + '</span>'
      + (c.detail ? '<span class="detail">' + c.detail + '</span>' : '');
    host.appendChild(li);
  }
}

// -- controls ----------------------------------------------------------------

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

$('confirm-turn').onclick = () => {
  // Everything the card showed as waiting: this turn's new tiles, and the older ones
  // still carrying a doubtful reading. Confirming exactly that set is what makes the
  // games list's count reach zero.
  const fresh = state.newThisTurn();
  const waiting = state.currentTurn().cells
    .filter(c => fresh.has(c.row + ',' + c.col) || state.doubtful(c));
  const n = state.confirmCells(waiting);
  if (n) say(n + ' tile' + (n === 1 ? '' : 's') + ' confirmed — undo takes it back in one press', 'ok');
};

$('annotation').onchange = e => state.setAnnotation(e.target.value);
$('note').oninput = e => state.setNote(e.target.value);

$('first').onclick = turns.first;
$('prev').onclick = turns.prev;
$('next').onclick = turns.next;
$('last').onclick = turns.last;
$('undo').onclick = () => state.undo();
$('redo').onclick = () => state.redo();
$('export').onclick = () => io.download();
$('sign-out').onclick = async () => { await api.logout(); location.replace('./login.html'); };

$('del-turn').onclick = () => {
  if (confirm('Delete the last turn? Its tiles cannot be recovered from the board.')) {
    turns.removeLast();
    more.close();
  }
};

$('load-sample').onclick = async () => {
  try {
    state.load(await io.loadFixture('fixtures/sample-game.json'));
    more.close();
  } catch (err) {
    alert(err.message);
  }
};

document.addEventListener('keydown', e => {
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  if (e.key === 'Home') { e.preventDefault(); turns.first(); }
  if (e.key === 'End') { e.preventDefault(); turns.last(); }
});

// -- the sheets --------------------------------------------------------------
//
// Eleven buttons in the footer became four moves: step back, step forward, more, scan.
// Nothing was taken away - import, export, undo, redo, delete-last and the sample game
// are all in here - but they no longer compete with the one control pressed every turn,
// and the destructive one no longer sits beside the navigation.

const more = $('more-dialog');
$('more').onclick = () => { checkReader(); more.showModal(); };
$('more-close').onclick = () => more.close();

const keys = $('keys-dialog');
$('keys').onclick = () => keys.showModal();
$('keys-close').onclick = () => keys.close();

// -- scanning ----------------------------------------------------------------
//
// A frame goes to the reader, the reader answers with a board, and the board fills the
// turn. Where the frame comes from - a photograph now, a webcam later, a camera on
// another machine after that - changes only the first of those three.

$('api-base').value = scan.apiBase();
$('api-base').onchange = e => {
  const next = e.target.value.trim();
  scan.setApiBase(next);
  // Games and frames go through `api.js`, scanning through `scan.js`; they are separate
  // concerns that happen to share one server today, and one address field that moved
  // only half of them would be a trap.
  api.setApiBase(next);
  checkReader();
};

async function checkReader() {
  const el = $('api-state');
  el.className = 'muted';
  el.textContent = 'checking…';
  try {
    const h = await scan.health();
    el.className = 'pass';
    el.textContent = 'ready · ' + h.templates.letters + ' letters';
  } catch (err) {
    el.className = 'fail';
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
    say('that turn is locked — press “edit it anyway” first', 'bad');
    return;
  }
  const blob = await scan.nextFrame();
  if (!blob) return;

  say('reading ' + blob.name + '…', 'busy');
  setBusy(true);
  try {
    const r = await scan.scanIntoCurrentTurn(blob);
    rememberFrame(state.get().currentTurn, r);

    // Keep the photograph with the game, not just in this tab. A transcript questioned
    // weeks later can only be settled by the picture, and by then the board has been
    // packed away. Failing to store it must not lose the reading, which is already in
    // the turn by this point - so it is reported and moved past.
    try {
      await api.addFrame(gameId, state.get().currentTurn, blob);
    } catch (err) {
      say('read fine, but the photograph could not be stored: ' + err.message, 'bad');
      return;
    }
    const parts = ['read ' + r.tiles + ' tiles'];
    if (r.blanks) parts.push(r.blanks + ' blank');
    if (r.unnamed) parts.push(r.unnamed + ' unreadable');
    if (r.rotated) parts.push('turned ' + r.rotated + '° to match the last turn');
    parts.push(r.flagged + ' to check');
    const doubtful = r.registration && r.registration.confident === false;
    if (doubtful) parts.push('registration doubtful — worth re-taking the photograph');
    say(parts.join(' · '), doubtful ? 'bad' : 'ok');
  } catch (err) {
    // A board the reader cannot find is an ordinary outcome, not a crash: bad light, a
    // hand across the board, a photograph taken too obliquely. Say so and leave the
    // turn alone so the operator can type it in instead.
    say('could not read it: ' + err.message, 'bad');
  } finally {
    setBusy(false);
  }
}

/**
 * What the reader made of it, in a sentence, under the board it just filled in.
 *
 * This is the program's whole output and it used to be one muted line in a panel on the
 * right - the least prominent place on the page. The mark beside it carries the same
 * state as the wording, so it is never colour alone.
 */
function say(text, state = '') {
  $('scan-result').textContent = text;
  $('scan-strip').className = state === 'bad' ? 'scan-strip bad'
    : (state === 'busy' ? 'scan-strip busy' : 'scan-strip');
  $('scan-mark').innerHTML = state === 'ok' ? TICK : (state === 'bad' ? ALERT : CAMERA);
}
say('Pick a photograph of the board; the reader fills the turn in.');

function setBusy(on) {
  busy = on;
  $('scan-new').disabled = on;
  $('scan-here').disabled = on;
  $('confirm-turn').disabled = on || $('confirm-turn').disabled;
}

$('scan-new').onclick = () => runScan(true);
$('scan-here').onclick = () => runScan(false);

// -- import ------------------------------------------------------------------

const dialog = $('import-dialog');
$('import').onclick = () => {
  $('import-problems').hidden = true;
  $('import-text').value = '';
  more.close();
  dialog.showModal();
};
$('import-cancel').onclick = () => dialog.close();

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
