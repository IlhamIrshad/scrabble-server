// Talking to the server about sign-in and saved games.
//
// Kept apart from scan.js, which is about reading a photograph. They happen to share a
// server today and need not always: one is a vision service and the other is where
// games live.

const TOKEN_KEY = 'board-ui.token';
const BASE_KEY = 'board-ui.apiBase';

export function apiBase() {
  try {
    return localStorage.getItem(BASE_KEY) || 'http://127.0.0.1:8000';
  } catch (err) {
    return 'http://127.0.0.1:8000';
  }
}

/**
 * Both this module and `scan.js` read the same stored address, and both may set it.
 * They are separate concerns - one is where games live, the other is a vision service -
 * but today they are one server, and asking twice for the same address would be a
 * question with no right answer.
 */
export function setApiBase(url) {
  try {
    localStorage.setItem(BASE_KEY, String(url).replace(/\/$/, ''));
  } catch (err) {
    /* the address just will not persist; this session still works */
  }
}

export function token() {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch (err) {
    return '';
  }
}

function setToken(value) {
  try {
    if (value) localStorage.setItem(TOKEN_KEY, value);
    else localStorage.removeItem(TOKEN_KEY);
  } catch (err) {
    /* signed in for this page only */
  }
}

/**
 * Every call to the server goes through here.
 *
 * A 401 means the token has gone - the server was restarted, which signs everyone out
 * by design - so it clears the stored one and says so plainly. Left unhandled, that
 * arrives as a page that quietly stops saving.
 */
async function call(path, { method = 'GET', body, form } = {}) {
  const headers = {};
  const t = token();
  if (t) headers.Authorization = 'Bearer ' + t;
  if (body !== undefined) headers['Content-Type'] = 'application/json';

  const res = await fetch(apiBase() + path, {
    method,
    headers,
    body: form ? form : (body !== undefined ? JSON.stringify(body) : undefined),
  });

  if (res.status === 401) {
    setToken('');
    const err = new Error('signed out');
    err.unauthorised = true;
    throw err;
  }
  if (!res.ok) {
    let detail = res.status + ' ' + res.statusText;
    try {
      const j = await res.json();
      if (j && j.detail) detail = j.detail;
    } catch (e) { /* keep the status line */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

// -- signing in --------------------------------------------------------------

export async function login(username, password) {
  const res = await fetch(apiBase() + '/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error(j.detail || 'could not sign in');
  }
  const out = await res.json();
  setToken(out.token);
  return out;
}

export async function logout() {
  try {
    await call('/auth/logout', { method: 'POST' });
  } catch (err) {
    /* the token is being thrown away regardless */
  }
  setToken('');
}

/** Whether the stored token still works. Cheap, and the only honest way to know. */
export async function signedIn() {
  if (!token()) return false;
  try {
    await call('/auth/check');
    return true;
  } catch (err) {
    return false;
  }
}

export async function passwordConfigured() {
  const res = await fetch(apiBase() + '/auth/configured');
  return res.ok ? res.json() : { configured: false, username: '' };
}

// -- games -------------------------------------------------------------------

export const listGames = () => call('/games').then(r => r.games);
export const getGame = id => call('/games/' + encodeURIComponent(id));
export const saveGame = (id, doc) => call('/games/' + encodeURIComponent(id), { method: 'PUT', body: doc });
export const deleteGame = id => call('/games/' + encodeURIComponent(id), { method: 'DELETE' });
export const listFrames = id => call('/games/' + encodeURIComponent(id) + '/frames').then(r => r.frames);

export function frameUrl(path) {
  return apiBase() + path;
}

export async function addFrame(id, turn, blob) {
  const form = new FormData();
  form.append('image', blob, blob.name || 'frame.jpg');
  return call('/games/' + encodeURIComponent(id) + '/frames/' + turn, { method: 'POST', form });
}

/** A game's id, from its name. Matches what the server would make of it. */
export function idFor(sessionId) {
  const cleaned = String(sessionId || '').trim().replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-._]+|[-._]+$/g, '');
  return cleaned.slice(0, 64) || 'game';
}
