// Running the vision pipeline from the UI: get a frame, post it, turn the answer into
// cells.
//
// The only thing that differs between a phone photo, a webcam and a camera on another
// machine is where the bytes come from. Everything after that is identical, so the
// frame source is a seam with one implementation today and room for the others.

import { SIZE } from './constants.js';
import * as state from './state.js';

const STORE_KEY = 'board-ui.apiBase';

export function apiBase() {
  try {
    return localStorage.getItem(STORE_KEY) || 'http://127.0.0.1:8000';
  } catch (err) {
    return 'http://127.0.0.1:8000';
  }
}

export function setApiBase(url) {
  try {
    localStorage.setItem(STORE_KEY, url.replace(/\/$/, ''));
  } catch (err) {
    /* the address just will not persist; scanning still works this session */
  }
}

// -- frame sources -----------------------------------------------------------

/** A photograph chosen from disk. What a phone gives you before a rig exists. */
export function fromFile() {
  return new Promise(resolve => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    // On a phone this offers the camera directly; on a desktop it is a file picker.
    input.capture = 'environment';
    input.onchange = () => resolve(input.files[0] || null);
    input.click();
  });
}

/**
 * A still from a live webcam. Not wired to a button yet, and deliberately written now
 * so that the shape of the seam is obvious: it returns a Blob exactly as `fromFile`
 * does, and nothing downstream can tell them apart.
 */
export async function fromCamera(video) {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

/**
 * Where the next frame comes from.
 *
 * The seam itself, rather than a description of one. A photograph off disk is the
 * source today; pointing this at a webcam is `setFrameSource(() => fromCamera(video))`
 * and nothing else in the application changes, which is the property the whole
 * arrangement exists for. It also means the source can be stood in for when testing,
 * without a file dialog appearing.
 */
let frameSource = fromFile;

export function setFrameSource(fn) {
  frameSource = fn || fromFile;
}

export function nextFrame() {
  return frameSource();
}

// -- the call ----------------------------------------------------------------

export async function detect(blob, { corners = null, debug = false } = {}) {
  const form = new FormData();
  form.append('image', blob, blob.name || 'frame.jpg');
  const params = new URLSearchParams();
  if (corners) params.set('corners', corners);
  // Asking for the debug render costs the reader an extra moment and gives back its own
  // view of the board: where it thought the edges were, how the rectified board came
  // out, and which pixels it took for ink. When a reading looks wrong that picture
  // usually says why, and no amount of staring at the original will.
  if (debug) params.set('debug', 'true');
  const query = params.toString() ? '?' + params.toString() : '';
  const res = await fetch(apiBase() + '/detect' + query, { method: 'POST', body: form });
  if (!res.ok) {
    let detail = res.status + ' ' + res.statusText;
    try {
      const body = await res.json();
      if (body && body.error) detail = body.error;
    } catch (err) { /* keep the status line */ }
    throw new Error(detail);
  }
  return res.json();
}

export async function health() {
  const res = await fetch(apiBase() + '/health');
  if (!res.ok) throw new Error('health check returned ' + res.status);
  return res.json();
}

// -- translating the answer --------------------------------------------------

/**
 * A pipeline response as board cells.
 *
 * Near enough one to one, with the one difference that matters: the pipeline reports a
 * tile it can see but cannot name - a blank, whose designation is not in the pixels,
 * or a tile that matched no reference shape well enough - and those arrive here with a
 * null letter rather than being dropped.
 */
export function toCells(response) {
  return (response.tiles || []).map(t => ({
    row: t.row,
    col: t.col,
    letter: t.letter == null ? null : String(t.letter).toUpperCase(),
    isBlank: !!t.blank,
    confidence: typeof t.confidence === 'number' ? t.confidence : null,
    source: 'ocr',
  }));
}

// -- orientation -------------------------------------------------------------

function rotateCells(cells, quarters) {
  return cells.map(c => {
    let r = c.row, col = c.col;
    for (let i = 0; i < quarters; i++) {
      const nr = col, nc = SIZE - 1 - r;
      r = nr; col = nc;
    }
    return { ...c, row: r, col };
  });
}

/**
 * Which way round the board was, decided by the tiles already known.
 *
 * The premium-square layout is perfectly 4-fold symmetric, so a board turned a quarter
 * turn between plays reads cleanly, passes every check the pipeline has, and puts every
 * coordinate somewhere else. Nothing in the image can settle it.
 *
 * The previous turn can. Tiles already on the board are not symmetric, so only one
 * orientation lines the new reading up with what was there before. It needs a few
 * tiles to be sure, which is why an opening scan is left alone - and there it does not
 * matter, because there is nothing yet to be inconsistent with.
 */
export function bestRotation(scanned, previousCells) {
  const anchor = previousCells || [];
  if (anchor.length < 4) return { quarters: 0, matched: 0, of: anchor.length, sure: false };

  const want = new Map(anchor.map(c => [c.row + ',' + c.col, c.letter]));
  let best = { quarters: 0, matched: -1, of: anchor.length, sure: false };
  const scores = [];
  for (let q = 0; q < 4; q++) {
    let matched = 0;
    for (const c of rotateCells(scanned, q)) {
      const key = c.row + ',' + c.col;
      if (!want.has(key)) continue;
      // A tile in the right place counts; the same letter as well counts for more, but
      // position alone is what the orientation actually turns on.
      matched += (want.get(key) === c.letter || c.letter == null) ? 1 : 0.5;
    }
    scores.push(matched);
    if (matched > best.matched) best = { quarters: q, matched, of: anchor.length, sure: false };
  }
  // Only act on a clear winner. A near-tie means the board is too sparse to tell, and
  // guessing at orientation is worse than leaving it as it was found.
  const sorted = [...scores].sort((a, b) => b - a);
  best.sure = sorted[0] >= anchor.length * 0.6 && sorted[0] > sorted[1] * 1.5;
  return best;
}

/**
 * Scan a frame and put the result into the current turn.
 *
 * Returns what happened rather than throwing on a poor read: a caller wants to show
 * the operator how many tiles came back, how many are doubtful and whether the board
 * had to be turned, all of which are ordinary outcomes rather than errors.
 */
export async function scanIntoCurrentTurn(blob, { debug = true } = {}) {
  const response = await detect(blob, { debug });
  let cells = toCells(response);

  const previous = state.previousTurn();
  const turn = bestRotation(cells, previous ? previous.cells : null);
  if (turn.sure && turn.quarters) cells = rotateCells(cells, turn.quarters);

  state.replaceCells(cells, 'scan');

  const flagged = cells.filter(c => c.confidence != null && c.confidence < 0.75).length;
  return {
    tiles: cells.length,
    flagged,
    unnamed: cells.filter(c => c.letter == null && !c.isBlank).length,
    blanks: cells.filter(c => c.isBlank).length,
    rotated: turn.sure ? turn.quarters * 90 : 0,
    rotationSure: turn.sure,
    anchor: turn.of,
    registration: response.registration || null,
    timing: response.timing_ms || null,
    // Where to find the two pictures afterwards: the photograph itself, and the
    // reader's annotated view of it.
    photo: blob,
    debugUrl: response.debug_image ? apiBase() + response.debug_image : null,
  };
}

export { rotateCells };
