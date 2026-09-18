// Import, export, autosave, and the query string. The seam with the Python side, so
// it is kept dumb and explicit: a document goes in, a document comes out.

import { SCHEMA_VERSION, SIZE } from './constants.js';
import * as state from './state.js';

const STORE_KEY = 'board-ui.document';

/**
 * Check a document enough to say *what* is wrong with it.
 *
 * Silence is the failure mode worth avoiding here. A file that half-loads and shows a
 * board missing three tiles looks like a vision problem and wastes an afternoon.
 */
export function validateDocument(d) {
  const problems = [];
  if (d == null || typeof d !== 'object') return ['not a JSON object'];
  if (d.schemaVersion !== SCHEMA_VERSION) {
    problems.push('schemaVersion is ' + JSON.stringify(d.schemaVersion) + ', expected ' + SCHEMA_VERSION);
  }
  if (!Array.isArray(d.turns) || d.turns.length === 0) {
    problems.push('turns must be a non-empty array');
    return problems;
  }
  d.turns.forEach((t, i) => {
    if (!Array.isArray(t.cells)) {
      problems.push('turn ' + i + ': cells must be an array');
      return;
    }
    t.cells.forEach((c, j) => {
      const where = 'turn ' + i + ', cell ' + j;
      if (!Number.isInteger(c.row) || c.row < 0 || c.row >= SIZE) problems.push(where + ': row out of range');
      if (!Number.isInteger(c.col) || c.col < 0 || c.col >= SIZE) problems.push(where + ': col out of range');
      // A null letter is legitimate - an unreadable tile, or a blank whose designation
      // is not known - so only a non-null value that is not a single letter is wrong.
      if (c.letter != null && !/^[A-Za-z]$/.test(String(c.letter))) {
        problems.push(where + ': letter must be a single letter or null, got ' + JSON.stringify(c.letter));
      }
      if (c.confidence != null && (typeof c.confidence !== 'number' || c.confidence < 0 || c.confidence > 1)) {
        problems.push(where + ': confidence must be null or between 0 and 1');
      }
    });
  });
  return problems;
}

export function importJSON(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    return { ok: false, problems: ['not valid JSON: ' + err.message] };
  }
  const problems = validateDocument(parsed);
  if (problems.length) return { ok: false, problems };
  state.load(parsed);
  return { ok: true, problems: [] };
}

export function exportJSON() {
  return JSON.stringify(state.get(), null, 2);
}

export function download() {
  const d = state.get();
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const blob = new Blob([exportJSON()], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = (d.sessionId || 'board') + '-' + stamp + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

// -- autosave ----------------------------------------------------------------
//
// Every storage call is wrapped: localStorage throws in private windows, when a quota
// is hit, and when a browser is configured to block it. None of those should stop the
// board from working.

export function save() {
  try {
    localStorage.setItem(STORE_KEY, exportJSON());
  } catch (err) {
    /* not fatal - the board keeps working without a saved copy */
  }
}

export function saved() {
  try {
    const text = localStorage.getItem(STORE_KEY);
    if (!text) return null;
    const parsed = JSON.parse(text);
    return validateDocument(parsed).length ? null : parsed;
  } catch (err) {
    return null;
  }
}

export function clearSaved() {
  try {
    localStorage.removeItem(STORE_KEY);
  } catch (err) {
    /* nothing to do */
  }
}

// -- query string ------------------------------------------------------------

/**
 * `?board=<url-encoded json>`. Being able to send someone a link to one exact board
 * is worth a great deal more when debugging than the ten lines it costs.
 */
export function fromQueryString() {
  const raw = new URLSearchParams(location.search).get('board');
  if (!raw) return null;
  try {
    const parsed = JSON.parse(decodeURIComponent(raw));
    return validateDocument(parsed).length ? null : parsed;
  } catch (err) {
    return null;
  }
}

export async function loadFixture(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error('could not load ' + path + ': ' + res.status);
  return res.json();
}
