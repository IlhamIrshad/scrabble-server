// The document model: board state, turns, and the edit log that undo is built on.
//
// This module never touches the DOM. Interactions call a function here, here emits a
// change, and renderers redraw. Keeping that one-directional is what lets the board
// component drop into the operator console later without a rewrite.

import { SIZE, SCHEMA_VERSION, CENTRE } from './constants.js';

let doc = null;
let undone = [];            // edits taken off the log by undo, awaiting redo
const listeners = new Set();

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit() {
  for (const fn of listeners) fn(doc);
}

const nowISO = () => new Date().toISOString();

export function emptyTurn(index, player) {
  return {
    index,
    player,
    annotation: 'play',
    cells: [],
    frameRef: null,
    capturedAt: nowISO(),
    note: '',
  };
}

export function createDocument(sessionId = 'untitled') {
  return {
    schemaVersion: SCHEMA_VERSION,
    sessionId,
    boardLayout: 'CrosswordGame',
    tileSet: 'english',
    rotation: 0,
    players: ['Player A', 'Player B'],
    turns: [emptyTurn(0, 0)],
    currentTurn: 0,
    editLog: [],
  };
}

export function load(next) {
  doc = normalise(next);
  undone = [];
  emit();
}

export function get() {
  return doc;
}

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));

/** Fill in anything an older or hand-written document leaves out. */
function normalise(d) {
  const out = { ...createDocument(d.sessionId || 'untitled'), ...d };
  out.editLog = Array.isArray(d.editLog) ? d.editLog : [];
  const turns = d.turns && d.turns.length ? d.turns : [emptyTurn(0, 0)];
  out.turns = turns.map((t, i) => ({
    ...emptyTurn(i, i % 2),
    ...t,
    index: i,
    cells: (t.cells || []).map(normaliseCell),
  }));
  out.currentTurn = clamp(out.currentTurn == null ? 0 : out.currentTurn, 0, out.turns.length - 1);
  return out;
}

function normaliseCell(c) {
  return {
    row: c.row,
    col: c.col,
    // Nullable on purpose. The camera can see that a square holds a tile without
    // being able to say which letter it is: a blank carries no designation in the
    // pixels at all, and a worn or shadowed tile sometimes matches no reference
    // shape well enough to name. Over 55 checked boards that is about 1.7 tiles per
    // board, so demanding a letter here would mean inventing one or dropping the
    // tile - and a dropped tile is invisible to everything downstream.
    letter: c.letter == null ? null : String(c.letter).toUpperCase(),
    isBlank: !!c.isBlank,
    confidence: c.confidence == null ? null : c.confidence,
    source: c.source || 'imported',
  };
}

// -- reading -----------------------------------------------------------------

export function currentTurn() {
  return doc.turns[doc.currentTurn];
}

export function previousTurn() {
  return doc.currentTurn > 0 ? doc.turns[doc.currentTurn - 1] : null;
}

/** A 15x15 lookup of a turn's cells, indexed [row][col]. */
export function grid(turn = currentTurn()) {
  const g = Array.from({ length: SIZE }, () => Array(SIZE).fill(null));
  for (const c of turn.cells) {
    if (c.row >= 0 && c.row < SIZE && c.col >= 0 && c.col < SIZE) g[c.row][c.col] = c;
  }
  return g;
}

export function cellAt(row, col, turn = currentTurn()) {
  return turn.cells.find(c => c.row === row && c.col === col) || null;
}

const key = (row, col) => row + ',' + col;

/**
 * Squares holding a tile now that held none in the turn before.
 *
 * Diffed at render time, never stored. Each turn carries the whole board precisely so
 * that a missing or corrupt turn cannot cascade into the rest, and so that any turn
 * can be drawn without replaying the game up to it.
 */
export function newThisTurn() {
  const prev = previousTurn();
  const before = new Set((prev ? prev.cells : []).map(c => key(c.row, c.col)));
  const out = new Set();
  for (const c of currentTurn().cells) {
    if (!before.has(key(c.row, c.col))) out.add(key(c.row, c.col));
  }
  return out;
}

// -- writing -----------------------------------------------------------------

function record(entry) {
  doc.editLog.push({ at: nowISO(), turnIndex: doc.currentTurn, ...entry });
  undone = [];              // a fresh edit abandons the redo branch
}

function snapshot(c) {
  if (!c) return null;
  return { letter: c.letter, isBlank: c.isBlank, confidence: c.confidence, source: c.source };
}

/** Place or replace a tile. `letter` may be null, meaning a tile of unknown letter. */
export function setCell(row, col, letter, isBlank = false, opts = {}) {
  const turn = currentTurn();
  const before = snapshot(cellAt(row, col, turn));
  const after = {
    letter: letter == null ? null : String(letter).toUpperCase(),
    isBlank: !!isBlank,
    // Null means "not from OCR", and renders as certain. A manually typed tile is the
    // most trustworthy thing on the board; treating null as zero confidence would
    // flag every one of them, which is exactly backwards.
    confidence: null,
    source: 'manual',
  };
  turn.cells = turn.cells.filter(c => !(c.row === row && c.col === col));
  turn.cells.push({ row, col, ...after });
  if (opts.log !== false) record({ action: 'set', row, col, before, after });
  emit();
}

export function clearCell(row, col, opts = {}) {
  const turn = currentTurn();
  const before = snapshot(cellAt(row, col, turn));
  if (!before) return;
  turn.cells = turn.cells.filter(c => !(c.row === row && c.col === col));
  if (opts.log !== false) record({ action: 'clear', row, col, before, after: null });
  emit();
}

/**
 * Replace everything on the current turn in one go, as a scan does.
 *
 * Logged as a single entry so that undo treats a scan as one act. Walking a scan back
 * ninety cells at a time would be unusable, and worse, it would leave the board in
 * states that were never on it.
 */
export function replaceCells(cells, why = 'scan') {
  const turn = currentTurn();
  const before = turn.cells.map(c => ({ ...c }));
  turn.cells = cells.map(c => ({ ...c }));
  record({ action: 'replaceCells', why, before: { cells: before }, after: { cells: turn.cells.map(c => ({ ...c })) } });
  emit();
}

export function setRotation(rotation) {
  const before = doc.rotation;
  doc.rotation = ((rotation % 360) + 360) % 360;
  record({ action: 'rotate', before: { rotation: before }, after: { rotation: doc.rotation } });
  emit();
}

export function setAnnotation(annotation) {
  const t = currentTurn();
  const before = t.annotation;
  t.annotation = annotation;
  record({ action: 'annotate', before: { annotation: before }, after: { annotation } });
  emit();
}

export function setNote(note) {
  currentTurn().note = note;
  emit();
}

export function setPlayers(players) {
  doc.players = players;
  emit();
}

export function setSessionId(id) {
  doc.sessionId = id;
  emit();
}

/**
 * Copy the board forward into a new turn, and hand play to the other player.
 *
 * This is the stand-in for a clock press. When the trigger becomes real hardware it
 * should call exactly this and nothing else, so that swapping one for the other is a
 * single change.
 */
export function addTurn() {
  const from = currentTurn();
  const turn = emptyTurn(doc.turns.length, (from.player + 1) % 2);
  turn.cells = from.cells.map(c => ({ ...c }));
  doc.turns.push(turn);
  doc.currentTurn = doc.turns.length - 1;
  record({ action: 'addTurn', before: null, after: { index: turn.index } });
  emit();
}

/** Only the last turn may go: deleting from the middle renumbers everything after it. */
export function deleteLastTurn() {
  if (doc.turns.length <= 1) return;
  const gone = doc.turns.pop();
  doc.currentTurn = clamp(doc.currentTurn, 0, doc.turns.length - 1);
  record({ action: 'deleteTurn', before: { index: gone.index, cells: gone.cells }, after: null });
  emit();
}

export function goToTurn(index) {
  doc.currentTurn = clamp(index, 0, doc.turns.length - 1);
  emit();
}

// -- undo --------------------------------------------------------------------

/**
 * Undo walks the edit log rather than a stack of saved documents, so what a reviewer
 * reads in the exported log is exactly what can be stepped back through.
 */
export function undo() {
  const entry = doc.editLog.pop();
  if (!entry) return false;
  apply(entry, 'before');
  undone.push(entry);
  emit();
  return true;
}

export function redo() {
  const entry = undone.pop();
  if (!entry) return false;
  apply(entry, 'after');
  doc.editLog.push(entry);
  emit();
  return true;
}

function apply(entry, side) {
  const target = entry[side];
  const turn = doc.turns[entry.turnIndex] || currentTurn();
  switch (entry.action) {
    case 'set':
    case 'clear': {
      turn.cells = turn.cells.filter(c => !(c.row === entry.row && c.col === entry.col));
      if (target) turn.cells.push({ row: entry.row, col: entry.col, ...target });
      doc.currentTurn = clamp(entry.turnIndex, 0, doc.turns.length - 1);
      break;
    }
    case 'replaceCells':
      turn.cells = (target.cells || []).map(c => ({ ...c }));
      doc.currentTurn = clamp(entry.turnIndex, 0, doc.turns.length - 1);
      break;
    case 'rotate':
      doc.rotation = target.rotation;
      break;
    case 'annotate':
      turn.annotation = target.annotation;
      break;
    case 'addTurn':
      if (side === 'before') doc.turns.pop();
      else doc.turns.push(emptyTurn(doc.turns.length, doc.turns.length % 2));
      doc.currentTurn = clamp(doc.currentTurn, 0, doc.turns.length - 1);
      break;
    case 'deleteTurn':
      if (side === 'before') {
        const back = emptyTurn(doc.turns.length, doc.turns.length % 2);
        back.cells = (entry.before.cells || []).map(c => ({ ...c }));
        doc.turns.push(back);
      } else {
        doc.turns.pop();
      }
      doc.currentTurn = clamp(doc.currentTurn, 0, doc.turns.length - 1);
      break;
    default:
      break;
  }
}

export const canUndo = () => doc.editLog.length > 0;
export const canRedo = () => undone.length > 0;

// -- coordinates -------------------------------------------------------------

/**
 * A square's name: column letter, then row number, so the centre is H8.
 *
 * The transposed form agrees on exactly that one square and disagrees everywhere
 * else, which is how it survives being looked at. The Python side had it the wrong way
 * round until this specification was checked against it.
 */
export function cellLabel(row, col) {
  return String.fromCharCode(65 + col) + (row + 1);
}

/**
 * A *play's* name, which is not the same thing. A horizontal play is named row first
 * (8H); a vertical play is named column first (H8). The GCG writer will need this.
 */
export function playLabel(row, col, direction) {
  const letter = String.fromCharCode(65 + col);
  const number = row + 1;
  return direction === 'horizontal' ? number + letter : letter + number;
}

export { CENTRE };
