// The landing page: every saved game, newest first.

import * as api from './api.js';
import * as state from './state.js';

const $ = id => document.getElementById(id);

// Anything here needs a token, so an unsigned visitor goes to the door rather than
// watching a list fail to load.
if (!(await api.signedIn())) location.replace('./login.html');

$('who').textContent = 'signed in';

function when(seconds) {
  const age = Date.now() / 1000 - seconds;
  if (age < 90) return 'just now';
  if (age < 3600) return Math.round(age / 60) + ' minutes ago';
  if (age < 86400) return Math.round(age / 3600) + ' hours ago';
  const days = Math.round(age / 86400);
  return days === 1 ? 'yesterday' : days + ' days ago';
}

function card(game) {
  const li = document.createElement('li');
  li.className = 'game' + (game.broken ? ' broken' : '');

  const open = document.createElement('a');
  open.className = 'game-open';
  open.href = './board.html?game=' + encodeURIComponent(game.id);

  const title = document.createElement('span');
  title.className = 'game-name';
  title.textContent = game.sessionId;
  open.appendChild(title);

  const facts = document.createElement('span');
  facts.className = 'game-facts muted';
  if (game.broken) {
    facts.textContent = 'this game’s file could not be read';
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
  open.appendChild(facts);

  const seen = document.createElement('span');
  seen.className = 'game-when muted';
  seen.textContent = when(game.updatedAt);
  open.appendChild(seen);

  li.appendChild(open);

  const remove = document.createElement('button');
  remove.className = 'danger';
  remove.textContent = 'delete';
  remove.title = 'Delete this game and its photographs';
  remove.onclick = async () => {
    // Named in the question, because "are you sure?" is a question nobody reads and
    // there is no undo behind this one.
    if (!confirm('Delete "' + game.sessionId + '" and its photographs?\n\nThis cannot be undone.')) return;
    remove.disabled = true;
    try {
      await api.deleteGame(game.id);
      await refresh();
    } catch (err) {
      trouble(err);
      remove.disabled = false;
    }
  };
  li.appendChild(remove);
  return li;
}

function trouble(err) {
  if (err && err.unauthorised) { location.replace('./login.html'); return; }
  $('trouble').hidden = false;
  $('trouble').textContent = err && err.message ? err.message : String(err);
}

async function refresh() {
  $('trouble').hidden = true;
  try {
    const games = await api.listGames();
    $('loading').hidden = true;
    $('empty').hidden = games.length > 0;
    const list = $('list');
    list.innerHTML = '';
    for (const g of games) list.appendChild(card(g));
  } catch (err) {
    $('loading').hidden = true;
    trouble(err);
  }
}

// -- making one --------------------------------------------------------------

$('new-game').onclick = async () => {
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
};

$('import').onclick = () => $('import-file').click();
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
