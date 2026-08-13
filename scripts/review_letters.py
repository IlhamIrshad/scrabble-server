#!/usr/bin/env python3
"""Confirm the letters on every correctly registered board, by marking only the wrong ones.

    python scripts/review_letters.py render      # slow, once (~3s a photo)
    python scripts/review_letters.py serve       # then mark, in a browser
    python scripts/review_letters.py report      # per-letter accuracy and confusions

This is the ground truth everything else has been standing in for. Without it, glyph
accuracy has to be inferred - from whether runs of letters are real words, and from
how far the letters read depart from what a tile set contains. Both work, and both are
indirect: the lexicon could explain only 25 of 167 bad runs unambiguously, and the tile
counts give a total without saying which cell was wrong.

Marking every cell would be 8000 judgements. Marking only the *wrong* ones is a few
dozen, and says exactly the same thing, because a board is not finished until it is
confirmed - so a cell nobody corrected on a confirmed board is a cell that read
correctly. That is the whole design: read the board, click what is wrong, move on.

Only boards whose registration a person already confirmed are offered. On a board that
is registered wrongly the letters are wrong for a reason that has nothing to do with
reading them, and mixing the two would poison the very numbers this exists to produce.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.vision import pipeline, registration

PHOTOS = "photos"
REVIEW = os.path.join("fixtures", "letters")
INDEX = os.path.join(REVIEW, "index.json")
VERDICTS = os.path.join("fixtures", "registration_verdicts.csv")
CORRECTIONS = os.path.join("fixtures", "letter_corrections.csv")
DONE = os.path.join("fixtures", "letter_reviewed.csv")
# Boards set aside rather than judged. Kept apart from DONE and never counted: on a
# confirmed board an unmarked cell means "this read correctly", and a board abandoned
# half way through has unmarked cells that mean nothing at all. Folding the two
# together would quietly credit every cell the reviewer never got to.
SKIPPED = os.path.join("fixtures", "letter_skipped.csv")
BOARD_PX = 900
EMPTY, BLANK = ".", "?"


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def confirmed_photos() -> list[str]:
    """Photographs whose registration a person judged correct."""
    if not os.path.exists(VERDICTS):
        raise SystemExit(f"no registration verdicts yet in {VERDICTS}")
    with open(VERDICTS, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(r["file"] for r in rows if r.get("verdict") == "yes")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def board_image(image: np.ndarray, result) -> np.ndarray:
    """The rectified board with the grid drawn on, and nothing written over a tile.

    The read letters go in the browser, as text beside each cell, rather than being
    burnt into the picture. A letter painted over the tile hides the very thing being
    checked, and this whole exercise is comparing what the tile says against what the
    pipeline read.
    """
    corners = np.array(result.registration.corners, np.float32)
    board, _ = registration.warp_board(image, corners)
    board = cv2.resize(board, (BOARD_PX, BOARD_PX), interpolation=cv2.INTER_AREA)
    step = BOARD_PX / 15.0
    for i in range(16):
        p = int(round(i * step))
        cv2.line(board, (p, 0), (p, BOARD_PX), (0, 0, 255), 1)
        cv2.line(board, (0, p), (BOARD_PX, p), (0, 0, 255), 1)
    return board


def render(args) -> None:
    os.makedirs(REVIEW, exist_ok=True)
    names = confirmed_photos()
    if args.limit:
        names = names[: args.limit]
    index = {} if args.again else (_read_json(INDEX) if os.path.exists(INDEX) else {})

    for i, name in enumerate(names, 1):
        out = os.path.join(REVIEW, os.path.splitext(name)[0] + ".jpg")
        if name in index and os.path.exists(out) and not args.again:
            continue
        image = cv2.imread(os.path.join(PHOTOS, name))
        if image is None:
            continue
        started = time.perf_counter()
        try:
            result = pipeline.detect(image)
        except registration.RegistrationError:
            continue
        cv2.imwrite(out, board_image(image, result), [int(cv2.IMWRITE_JPEG_QUALITY), 88])

        low = {(c.row, c.col) for c in result.low_confidence}
        index[name] = {
            "image": os.path.basename(out),
            "board": list(result.board),
            # Flagged cells are drawn differently so the eye goes to them first;
            # they are where the pipeline itself expects to be wrong.
            "low": sorted([r, c] for r, c in low),
            "tiles": result.tile_count,
            "ms": round((time.perf_counter() - started) * 1000),
        }
        json.dump(index, open(INDEX, "w", encoding="utf-8"), indent=1)
        print(f"  [{i}/{len(names)}] {name}  tiles={result.tile_count}", flush=True)

    json.dump(index, open(INDEX, "w", encoding="utf-8"), indent=1)
    print(f"\nrendered {len(index)} boards; now: "
          f"python scripts/review_letters.py serve")


# ---------------------------------------------------------------------------
# reviewing
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><meta charset=utf-8>
<title>letter review</title>
<style>
 body{margin:0;background:#141414;color:#ddd;font:14px system-ui;text-align:center}
 #bar{padding:7px;position:sticky;top:0;background:#1e1e1e;border-bottom:1px solid #333;z-index:5}
 #wrap{position:relative;display:inline-block;margin-top:6px}
 #shot{display:block;max-height:calc(100vh - 140px);width:auto}
 .cell{position:absolute;box-sizing:border-box;cursor:pointer}
 .cell span{position:absolute;top:0;left:1px;font:700 11px/1 system-ui;
   color:#6f6;text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none}
 .cell.low span{color:#fd0}
 .cell.sel{outline:3px solid #fff;outline-offset:-3px}
 .cell.fixed span{color:#f66}
 .cell.fixed{background:rgba(255,60,60,.28)}
 b{color:#fff} .k{color:#7cf} .done{color:#8d8}
 #hint{padding:5px;color:#888}
 button{font:13px system-ui;margin:0 3px;padding:5px 11px;border-radius:5px;
  border:1px solid #444;background:#2a2a2a;color:#ddd;cursor:pointer}
</style>
<div id=bar><span id=status>loading…</span></div>
<div id=wrap><img id=shot><div id=cells></div></div>
<div id=hint>
 <button onclick="commit()">Enter — confirm, next</button>
 <button onclick="skip()">S — skip this board</button>
 <button onclick="back()">← previous</button>
 <span class=k>click a cell, then type the right letter &nbsp;·&nbsp;
 <b>.</b> empty &nbsp; <b>?</b> blank &nbsp; <b>Esc</b> unselect</span>
</div>
<script>
let cur=null, fixes={}, sel=null;
const img=document.getElementById('shot'), host=document.getElementById('cells');

async function load(){
  const r=await fetch('/next'); cur=await r.json(); fixes={}; sel=null;
  if(cur.done){ document.getElementById('status').innerHTML='<b class=done>All done — '+cur.summary+'</b>';
    img.src=''; host.innerHTML=''; return; }
  img.src='/img/'+cur.image+'?t='+Date.now();
  img.onload=draw;
  document.getElementById('status').innerHTML='<b>'+cur.file+'</b> &nbsp; '+cur.index+'/'+cur.total+
    ' &nbsp; tiles <b>'+cur.tiles+'</b> &nbsp; <span class=k>'+cur.low.length+' flagged</span>'+
    ' &nbsp;·&nbsp; <span class=done>'+cur.judged+' done</span>, '+cur.skipped+' skipped';
}

function draw(){
  const w=img.clientWidth, h=img.clientHeight, s=w/15;
  const low=new Set(cur.low.map(p=>p[0]+','+p[1]));
  host.innerHTML='';
  for(let r=0;r<15;r++) for(let c=0;c<15;c++){
    const ch=cur.board[r][c];
    const d=document.createElement('div');
    d.className='cell'+(low.has(r+','+c)?' low':'');
    d.style.left=(c*s)+'px'; d.style.top=(r*h/15)+'px';
    d.style.width=s+'px'; d.style.height=(h/15)+'px';
    d.dataset.r=r; d.dataset.c=c; d.dataset.read=ch;
    if(ch!=='.'){ const t=document.createElement('span'); t.textContent=ch; d.appendChild(t); }
    d.onclick=()=>select(d);
    host.appendChild(d);
  }
}

function select(d){
  if(sel) sel.classList.remove('sel');
  sel=d; d.classList.add('sel');
}

function setTruth(ch){
  if(!sel) return;
  const r=sel.dataset.r, c=sel.dataset.c, read=sel.dataset.read;
  if(ch===read){ delete fixes[r+','+c]; sel.classList.remove('fixed'); render(sel,read); }
  else { fixes[r+','+c]={row:+r,col:+c,read:read,truth:ch};
         sel.classList.add('fixed'); render(sel,ch); }
  sel.classList.remove('sel'); sel=null;
}

function render(d,ch){
  d.innerHTML='';
  if(ch!=='.'){ const t=document.createElement('span'); t.textContent=ch; d.appendChild(t); }
}

async function commit(){
  if(!cur||cur.done) return;
  await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:cur.file, fixes:Object.values(fixes)})});
  load();
}
async function skip(){
  if(!cur||cur.done) return;
  await fetch('/skip',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:cur.file})});
  load();
}
async function back(){ await fetch('/back',{method:'POST'}); load(); }

document.addEventListener('keydown',e=>{
  if(e.key==='Enter'){ commit(); return; }
  if(e.key==='ArrowLeft'){ back(); return; }
  if(!sel && (e.key==='s'||e.key==='S')){ skip(); return; }
  if(e.key==='Escape'){ if(sel) sel.classList.remove('sel'); sel=null; return; }
  if(!sel) return;
  if(e.key==='.'||e.key==='Delete'||e.key==='Backspace'){ setTruth('.'); return; }
  if(e.key==='?'||e.key==='/'){ setTruth('?'); return; }
  if(/^[a-zA-Z]$/.test(e.key)) setTruth(e.key.toUpperCase());
});
window.addEventListener('resize',()=>{ if(cur&&!cur.done) draw(); });
load();
</script>
"""


def _load_list(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [r[0] for r in csv.reader(fh) if r and r[0] != "file"]


def _append(path: str, name: str) -> None:
    first = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if first:
            w.writerow(["file"])
        w.writerow([name])


def _rewrite(path: str, names: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file"])
        w.writerows([[n] for n in names])


def load_done() -> list[str]:
    return _load_list(DONE)


def load_skipped() -> list[str]:
    return _load_list(SKIPPED)


def serve(args) -> None:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    index = _read_json(INDEX)
    names = sorted(index)
    app = FastAPI()

    @app.get("/")
    def root():
        return HTMLResponse(PAGE)

    @app.get("/img/{name}")
    def img(name: str):
        return FileResponse(os.path.join(REVIEW, os.path.basename(name)))

    @app.get("/next")
    def nxt():
        done, skipped = load_done(), load_skipped()
        seen = set(done) | set(skipped)
        todo = [n for n in names if n not in seen]
        if not todo:
            fixes = sum(1 for _ in _corrections())
            return JSONResponse({"done": True, "summary":
                                 f"{len(done)} boards confirmed, {fixes} cells "
                                 f"corrected, {len(skipped)} skipped"})
        name = todo[0]
        facts = index[name]
        return JSONResponse({
            "done": False, "file": name, "image": facts["image"],
            "board": facts["board"], "low": facts["low"], "tiles": facts["tiles"],
            "index": len(names) - len(todo) + 1, "total": len(names),
            "judged": len(done), "skipped": len(skipped),
        })

    @app.post("/vote")
    async def vote(body: dict):
        name = body["file"]
        new = not os.path.exists(CORRECTIONS)
        with open(CORRECTIONS, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["file", "row", "col", "read", "truth"])
            for f in body.get("fixes", []):
                w.writerow([name, f["row"], f["col"], f["read"], f["truth"]])
        _append(DONE, name)
        return {"ok": True}

    @app.post("/skip")
    async def skip(body: dict):
        _append(SKIPPED, body["file"])
        return {"ok": True}

    @app.post("/back")
    def back():
        """Undo the most recent board, whether it was confirmed or skipped."""
        done, skipped = load_done(), load_skipped()
        if not done and not skipped:
            return {"ok": False}
        # Which one to undo is decided by position in the queue, not by which file was
        # written last: the reviewer means "the board I just saw", and confirmed and
        # skipped boards interleave.
        order = {n: i for i, n in enumerate(names)}
        last_done = max(done, key=lambda n: order.get(n, -1), default=None)
        last_skip = max(skipped, key=lambda n: order.get(n, -1), default=None)
        if last_skip is not None and (
            last_done is None or order.get(last_skip, -1) > order.get(last_done, -1)
        ):
            _rewrite(SKIPPED, [n for n in skipped if n != last_skip])
            return {"ok": True}

        last = last_done
        _rewrite(DONE, [n for n in done if n != last])
        rows = [r for r in _corrections() if r["file"] != last]
        with open(CORRECTIONS, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "row", "col", "read", "truth"])
            for r in rows:
                w.writerow([r["file"], r["row"], r["col"], r["read"], r["truth"]])
        return {"ok": True}

    import uvicorn
    print(f"\n  open  http://127.0.0.1:{args.port}\n"
          f"  click a cell, then type the right letter; '.' empty, '?' blank\n"
          f"  Enter confirms and moves on; S skips this board; <- goes back\n"
          f"  saved to {CORRECTIONS} as you go; Ctrl-C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def _corrections() -> list[dict]:
    if not os.path.exists(CORRECTIONS):
        return []
    with open(CORRECTIONS, encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("file")]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(args) -> None:
    """Per-letter accuracy and the confusions behind it, from confirmed boards only."""
    done = load_done()
    if not done:
        raise SystemExit(f"no boards confirmed yet in {DONE}")
    index = _read_json(INDEX)
    fixes = _corrections()

    # Every cell on a confirmed board read correctly unless it was corrected, so the
    # totals come from the boards and only the errors come from the corrections file.
    read_counts: Counter = Counter()
    for name in done:
        for row in index[name]["board"]:
            read_counts.update(ch for ch in row if ch not in (EMPTY,))
    wrong: Counter = Counter()
    confusion: Counter = Counter()
    for f in fixes:
        wrong[f["read"]] += 1
        confusion[(f["read"], f["truth"])] += 1

    tiles = sum(read_counts.values())
    skipped = load_skipped()
    print(f"{len(done)} boards confirmed, {tiles} tiles read, "
          f"{len(fixes)} corrected -> {1 - len(fixes) / max(tiles, 1):.2%} per-tile accuracy")
    if skipped:
        # Named rather than merely counted, because a skip is usually a judgement -
        # too messy, too many leftovers - and a run of them says something about the
        # boards themselves that the accuracy figure deliberately excludes.
        print(f"{len(skipped)} boards skipped and not counted: "
              + ", ".join(sorted(skipped)[:6]) + (" ..." if len(skipped) > 6 else ""))
    print()

    print("  worst letters, as read:")
    rows = [(wrong[ch] / read_counts[ch], ch, wrong[ch], read_counts[ch])
            for ch in read_counts if read_counts[ch] >= args.min_seen]
    for rate, ch, bad, seen in sorted(rows, reverse=True)[:12]:
        if bad:
            print(f"    {ch}  wrong {bad:4d} of {seen:4d} read  ({rate:.1%})")

    print("\n  confusions (read -> actually):")
    for (got, want), n in confusion.most_common(20):
        print(f"    {got} -> {want}   x{n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["render", "serve", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--again", action="store_true", help="re-render everything")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--min-seen", type=int, default=10,
                    help="letters read fewer times than this are not rated")
    args = ap.parse_args()
    {"render": render, "serve": serve, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
