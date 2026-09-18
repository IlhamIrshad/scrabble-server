// The landing page: every saved game, newest first.
//
// A row answers one question before you open it - is this game finished being checked?
// Turn and tile counts say how big a game is, which is rarely why anyone opens one.
// The number of tiles still waiting on a person is why, so it is the only thing on the
// page rendered in a colour.

import * as api from './api.js';
import * as state from './state.js';

const $ = id => document.getElementById(id);

// Anything here needs a token, so an unsigned visitor goes to the door rather than
// watching a list fail to load.
if (!(await api.signedIn())) location.replace('./login.html');

const svg = (paths, size = 15) =>
  '<svg viewBox="0 0 24 24" width="' + size + '" height="' + size + '" fill="none" '
  + 'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
  + 'stroke-linejoin="round" aria-hidden="true">' + paths + '</svg>';

const TICK = svg('<polyline points="20 6 9 17 4 12"/>', 12);
const DOTS = '<svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" '
  + 'aria-hidden="true"><circle cx="12" cy="5" r="1.8"/><circle cx="12" cy="12" r="1.8"/>'
  + '<circle cx="12" cy="19" r="1.8"/></svg>';
const ALERT = svg('<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>'
  + '<line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>', 24);

// -- how long ago ------------------------------------------------------------

function when(seconds) {
  const age = Date.now() / 1000 - seconds;
  if (age < 90) return 'just now';
  if (age < 3600) return Math.round(age / 60) + ' minutes ago';
  if (age < 86400) return Math.round(age / 3600) + ' hours ago';
  const days = Math.round(age / 86400);
  return days === 1 ? 'yesterday' : days + ' days ago';
}

/** The stamp itself, for when "8 days ago" is not precise enough to match a notebook. */
function exact(seconds) {
  const d = new Date(seconds * 1000);
  const month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][d.getMonth()];
  const pad = n => String(n).padStart(2, '0');
  return pad(d.getDate()) + ' ' + month + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

const startOfToday = () => new Date(new Date().setHours(0, 0, 0, 0)).getTime() / 1000;

// -- a row -------------------------------------------------------------------

/**
 * Tiles on the board at each turn, as bars.
 *
 * Cheap, and the only picture of a game available without opening it: a game that ran
 * to the end climbs and flattens, one abandoned after three turns does not. The last
 * bar is picked out because where a game stopped is the part worth seeing.
 */
function shape(game) {
  const el = document.createElement('div');
  el.className = 'shape';

  if (game.broken) {
    el.classList.add('none');
    el.innerHTML = ALERT;
    return el;
  }

  const counts = game.shape && game.shape.length ? game.shape : [game.tiles || 0];
  const top = Math.max(1, ...counts);
  // A game untouched for a week is not in progress, and its last bar should not claim
  // to be.
  if (Date.now() / 1000 - game.updatedAt > 86400) el.classList.add('cold');
  for (const n of counts) {
    const bar = document.createElement('i');
    bar.style.height = Math.max(8, Math.round((n / top) * 100)) + '%';
    el.appendChild(bar);
  }
  el.title = counts.length + ' turns, ending with ' + counts[counts.length - 1] + ' tiles on the board';
  return el;
}

function standing(game) {
  const el = document.createElement('span');
  el.className = 'pill';

  if (game.broken) return null;

  // Tiles waiting on a person come first whatever else is true. A game can arrive by
  // import with doubtful readings and no photographs of its own, and calling that one
  // "typed in by hand" would hide the only number here worth acting on.
  const left = game.toConfirm || 0;
  if (left) {
    el.className = 'pill doubt';
    el.innerHTML = '<span class="dot"></span><span>' + left + ' tile' + (left === 1 ? '' : 's') + ' to confirm</span>';
    el.title = 'Tiles the reader was unsure of, or could not name, that nobody has confirmed.';
    return el;
  }
  if (!game.frames) {
    el.textContent = 'typed in by hand';
    el.title = 'No photographs were stored for this game.';
    return el;
  }
  el.className = 'pill ok';
  el.innerHTML = TICK + '<span>all confirmed</span>';
  el.title = 'Every tile on the last turn was either read confidently or checked by a person.';
  return el;
}

function card(game) {
  const li = document.createElement('li');
  li.className = 'game' + (game.broken ? ' broken' : '');

  const open = document.createElement(game.broken ? 'div' : 'a');
  open.className = 'game-open';
  if (!game.broken) open.href = './board.html?game=' + encodeURIComponent(game.id);

  open.appendChild(shape(game));

  const middle = document.createElement('div');
  middle.style.flex = '1';
  middle.style.minWidth = '0';

  const title = document.createElement('div');
  title.className = 'game-name';
  title.textContent = game.sessionId;
  middle.appendChild(title);

  const facts = document.createElement('div');
  facts.className = 'game-facts';
  if (game.broken) {
    facts.textContent = 'this game’s file could not be read — games/' + game.id + '/game.json';
  } else {
    const bits = [
      game.turns + (game.turns === 1 ? ' turn' : ' turns'),
      game.tiles + ' tiles on the board',
    ];
    // Only worth saying when there are none: a game with no photographs was typed in
    // by hand, which is the thing you would want to know at a glance.
    if (!game.frames) bits.push('no photographs');
    else bits.push(game.frames + (game.frames === 1 ? ' photograph' : ' photographs'));
    facts.textContent = bits.join(' · ');
  }
  middle.appendChild(facts);
  open.appendChild(middle);
  li.appendChild(open);

  const mark = standing(game);
  if (mark) li.appendChild(mark);

  const seen = document.createElement('div');
  seen.className = 'game-when';
  seen.innerHTML = '<div>' + when(game.updatedAt) + '</div>'
    + '<div class="exact">' + exact(game.updatedAt) + '</div>';
  li.appendChild(seen);

  const more = document.createElement('button');
  more.className = 'icon quiet';
  more.setAttribute('aria-label', 'More actions for ' + game.sessionId);
  more.innerHTML = DOTS;
  more.onclick = () => openMenu(game);
  li.appendChild(more);

  return li;
}

// -- the per-row menu --------------------------------------------------------
//
// Delete used to sit in the row itself, one slip away from a game that has no second
// copy. Behind a sheet it needs a deliberate press, and there is room beside it for
// the thing people actually want from a list - a copy of the file - without opening
// the game to export it.

const menu = $('row-menu');
let menuGame = null;

function openMenu(game) {
  menuGame = game;
  $('row-menu-name').textContent = game.sessionId;
  $('row-menu-where').textContent = 'games/' + game.id + '/';
  $('row-open').disabled = !!game.broken;
  $('row-download').disabled = !!game.broken;
  menu.showModal();
}

$('row-close').onclick = () => menu.close();

$('row-open').onclick = () => {
  location.href = './board.html?game=' + encodeURIComponent(menuGame.id);
};

$('row-download').onclick = async () => {
  try {
    const doc = await api.getGame(menuGame.id);
    const blob = new Blob([JSON.stringify(doc, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = menuGame.id + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
    menu.close();
  } catch (err) {
    trouble(err);
    menu.close();
  }
};

$('row-delete').onclick = async () => {
  // Named in the question, because "are you sure?" is a question nobody reads and
  // there is no undo behind this one.
  if (!confirm('Delete "' + menuGame.sessionId + '" and its photographs?\n\nThis cannot be undone.')) return;
  const id = menuGame.id;
  menu.close();
  try {
    await api.deleteGame(id);
    await refresh();
  } catch (err) {
    trouble(err);
  }
};

// -- trouble -----------------------------------------------------------------

function trouble(err, title = '', command = '') {
  if (err && err.unauthorised) { location.replace('./login.html'); return; }
  const message = err && err.message ? err.message : String(err);
  $('trouble').hidden = false;
  $('trouble-title').textContent = title || message;
  $('trouble-body').textContent = title ? message : '';
  $('trouble-body').hidden = !title;
  $('trouble-cmd').hidden = !command;
  $('trouble-cmd').textContent = command;
}

// -- drawing the list --------------------------------------------------------

function group(label, games) {
  const wrap = document.createDocumentFragment();
  const head = document.createElement('p');
  head.className = 'group-label';
  head.textContent = label;
  wrap.appendChild(head);
  const list = document.createElement('ul');
  list.className = 'game-list';
  for (const g of games) list.appendChild(card(g));
  wrap.appendChild(list);
  return wrap;
}

function summarise(games) {
  if (!games.length) return '';
  const waiting = games.filter(g => !g.broken && (g.toConfirm || 0) > 0).length;
  const saved = games.length + ' saved';
  if (!waiting) return saved;
  return saved + ' · <span class="doubt">' + waiting + ' with tiles still to confirm</span>';
}

async function refresh() {
  $('trouble').hidden = true;
  try {
    const games = await api.listGames();
    $('loading').hidden = true;
    $('empty').hidden = games.length > 0;
    $('where').hidden = games.length === 0;
    $('count').innerHTML = summarise(games);

    const groups = $('groups');
    groups.innerHTML = '';
    const cut = startOfToday();
    const today = games.filter(g => g.updatedAt >= cut);
    const earlier = games.filter(g => g.updatedAt < cut);
    // Only worth a heading when there is something to separate. A lone "Today" over a
    // single row is furniture.
    if (today.length && earlier.length) {
      groups.appendChild(group('Today', today));
      groups.appendChild(group('Earlier', earlier));
    } else if (games.length) {
      const list = document.createElement('ul');
      list.className = 'game-list';
      list.style.marginTop = '20px';
      for (const g of games) list.appendChild(card(g));
      groups.appendChild(list);
    }
  } catch (err) {
    $('loading').hidden = true;
    trouble(err);
  }
}

// -- the reader --------------------------------------------------------------

async function checkReader() {
  const pill = $('reader-pill');
  const text = $('reader-text');
  try {
    const res = await fetch(api.apiBase() + '/health');
    if (!res.ok) throw new Error('unhealthy');
    pill.className = 'pill ok';
    text.textContent = 'reader up';
  } catch (err) {
    pill.className = 'pill bad';
    text.textContent = 'no reader';
    // The list itself needs the server, so if this fails the page has bigger problems
    // than a pill - but scanning is what it is really reporting on, and saying which
    // address failed saves looking in the wrong place.
    trouble(
      new Error('Games live on it, so the page needs it to sign in, to load anything and to scan.'),
      'Cannot reach the reader at ' + api.apiBase() + '.',
      'python -m uvicorn app.main:app --port 8000',
    );
  }
}

// -- making one --------------------------------------------------------------

async function newGame() {
  const name = prompt('Name for this game', 'game-' + new Date().toISOString().slice(0, 10));
  if (name === null) return;
  const id = api.idFor(name);
  try {
    // Written before opening it, so the game exists on disk from the outset and a
    // browser closed halfway through leaves a game rather than nothing.
    await api.saveGame(id, state.createDocument(name || id));
    location.href = './board.html?game=' + encodeURIComponent(id);
  } catch (err) {
    trouble(err);
  }
}

$('new-game').onclick = newGame;
$('empty-new').onclick = newGame;

const pickFile = () => $('import-file').click();
$('import').onclick = pickFile;
$('empty-import').onclick = pickFile;

$('import-file').onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const doc = JSON.parse(await file.text());
    const name = doc.sessionId || file.name.replace(/\.json$/i, '');
    const id = api.idFor(name);
    await api.saveGame(id, doc);
    location.href = './board.html?game=' + encodeURIComponent(id);
  } catch (err) {
    trouble(err);
  }
};

$('sign-out').onclick = async () => {
  await api.logout();
  location.replace('./login.html');
};

refresh();
checkReader();
