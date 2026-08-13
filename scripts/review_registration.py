#!/usr/bin/env python3
"""Flip through every photograph and judge whether the board was found correctly.

    python scripts/review_registration.py render      # slow, once (~2s a photo)
    python scripts/review_registration.py serve       # then judge, in a browser

Registration is the one verdict a person can give in about a second and a machine
cannot give at all: either the grid lines sit on the board's cells or they do not.
Letters are a separate, much slower judgement, and this deliberately does not ask
for them.

Two commands because rendering is the slow part. `render` runs the real pipeline
over every photograph and writes a side-by-side image - the photograph with the
quad it chose, and the rectified board with the 15x15 grid drawn on it. `serve`
then flips through those instantly.

Verdicts are written to fixtures/registration_verdicts.csv after every keypress,
so stopping halfway loses nothing and re-running resumes where you left off.

Why it is worth your time: these verdicts are ground truth for *registration*,
which lets the automatic checks be calibrated against human judgement instead of
against my guesses. The screening script can then tell a misregistered board from a
correctly registered one on its own.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision import pipeline, registration

PHOTOS = "photos"
REVIEW = os.path.join("fixtures", "review")
VERDICTS = os.path.join("fixtures", "registration_verdicts.csv")
INDEX = os.path.join(REVIEW, "index.json")
STILLS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp"}
PANEL = 820


def photo_list(folder: str) -> list[str]:
    return [n for n in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, n))
            and os.path.splitext(n)[1].lower() in STILLS]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def review_image(image: np.ndarray, result) -> np.ndarray:
    """The photograph with the chosen quad, beside the rectified board plus grid."""
    corners = np.array(result.registration.corners, np.float32)

    src = image.copy()
    cv2.polylines(src, [corners.astype(np.int32)], True, (0, 255, 0),
                  max(3, src.shape[1] // 350))
    for x, y in corners:
        cv2.circle(src, (int(x), int(y)), max(7, src.shape[1] // 150), (0, 0, 255), -1)

    board, _ = registration.warp_board(image, corners)
    size = board.shape[0]
    step = size / 15.0
    for i in range(16):
        p = int(round(i * step))
        cv2.line(board, (p, 0), (p, size), (0, 0, 255), 1)
        cv2.line(board, (0, p), (size, p), (0, 0, 255), 1)

    s = PANEL / src.shape[1]
    left = cv2.resize(src, (PANEL, int(src.shape[0] * s)))
    right = cv2.resize(board, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    height = max(left.shape[0], right.shape[0])
    pad = lambda p: np.vstack(
        [p, np.zeros((height - p.shape[0], PANEL, 3), np.uint8)])
    return np.hstack([pad(left), pad(right)])


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def render(args) -> None:
    os.makedirs(REVIEW, exist_ok=True)
    names = photo_list(PHOTOS)
    if args.limit:
        names = names[: args.limit]
    index = {}
    if os.path.exists(INDEX) and not args.again:
        index = _read_json(INDEX)

    for i, name in enumerate(names, 1):
        out = os.path.join(REVIEW, os.path.splitext(name)[0] + ".jpg")
        if name in index and os.path.exists(out) and not args.again:
            continue
        image = cv2.imread(os.path.join(PHOTOS, name))
        if image is None:
            index[name] = {"ok": False, "note": "unreadable"}
            continue
        started = time.perf_counter()
        try:
            result = pipeline.detect(image)
        except registration.RegistrationError as exc:
            index[name] = {"ok": False, "note": str(exc)[:90]}
            print(f"  [{i}/{len(names)}] {name}  NO BOARD")
            continue
        cv2.imwrite(out, review_image(image, result),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        index[name] = {
            "ok": True,
            "image": os.path.basename(out),
            "tiles": result.tile_count,
            "score": round(result.registration.grid_score, 1),
            "alignment": round(result.registration.grid_alignment, 1),
            "method": result.registration.method,
            "low_conf": len(result.low_confidence),
            "ms": round((time.perf_counter() - started) * 1000),
        }
        json.dump(index, open(INDEX, "w", encoding="utf-8"), indent=1)
        print(f"  [{i}/{len(names)}] {name}  tiles={result.tile_count} "
              f"score={index[name]['score']}", flush=True)

    json.dump(index, open(INDEX, "w", encoding="utf-8"), indent=1)
    print(f"\nrendered {sum(1 for v in index.values() if v.get('ok'))} of "
          f"{len(names)}; {sum(1 for v in index.values() if not v.get('ok'))} "
          f"found no board at all")


# ---------------------------------------------------------------------------
# reviewing
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><meta charset=utf-8>
<title>registration review</title>
<style>
 body{margin:0;background:#141414;color:#ddd;font:14px system-ui;text-align:center}
 #bar{padding:8px;position:sticky;top:0;background:#1e1e1e;border-bottom:1px solid #333}
 img{max-width:99vw;max-height:calc(100vh - 108px);object-fit:contain}
 b{color:#fff} .k{color:#7cf} .done{color:#8d8}
 button{font:14px system-ui;margin:0 4px;padding:6px 14px;border-radius:5px;
  border:1px solid #444;background:#2a2a2a;color:#ddd;cursor:pointer}
 #hint{padding:6px;color:#888}
</style>
<div id=bar><span id=status>loading…</span></div>
<img id=shot>
<div id=hint>
 <button onclick="vote('yes')">Y — grid is correct</button>
 <button onclick="vote('no')">N — wrong</button>
 <button onclick="vote('unsure')">U — unsure</button>
 <button onclick="back()">← back</button>
 &nbsp;<span class=k>keys: Y / N / U / ←</span>
</div>
<script>
let cur = null;
async function load(){
  const r = await fetch('/next'); cur = await r.json();
  if(cur.done){ document.getElementById('status').innerHTML =
      '<b class=done>All done — '+cur.summary+'</b>';
    document.getElementById('shot').src=''; return; }
  document.getElementById('shot').src = '/img/'+cur.image+'?t='+Date.now();
  document.getElementById('status').innerHTML =
    '<b>'+cur.file+'</b> &nbsp; '+cur.index+'/'+cur.total+
    ' &nbsp; tiles <b>'+cur.tiles+'</b> &nbsp; score '+cur.score+
    ' &nbsp; align '+cur.alignment+' &nbsp; '+cur.method+
    ' &nbsp;·&nbsp; <span class=done>'+cur.judged+' judged</span>';
}
async function vote(v){
  if(!cur||cur.done) return;
  await fetch('/vote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({file:cur.file,verdict:v})});
  load();
}
async function back(){ await fetch('/back',{method:'POST'}); load(); }
document.addEventListener('keydown',e=>{
  const k=e.key.toLowerCase();
  if(k==='y')vote('yes'); else if(k==='n')vote('no');
  else if(k==='u')vote('unsure'); else if(e.key==='ArrowLeft')back();
});
load();
</script>
"""


def load_verdicts() -> dict[str, str]:
    if not os.path.exists(VERDICTS):
        return {}
    with open(VERDICTS, encoding="utf-8") as fh:
        return {r["file"]: r["verdict"] for r in csv.DictReader(fh) if r.get("file")}


def save_verdict(file: str, verdict: str, facts: dict) -> None:
    new = not os.path.exists(VERDICTS)
    with open(VERDICTS, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "verdict", "tiles", "score", "alignment", "method"])
        w.writerow([file, verdict, facts.get("tiles", ""), facts.get("score", ""),
                    facts.get("alignment", ""), facts.get("method", "")])


def serve(args) -> None:
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    index = _read_json(INDEX)
    renderable = [n for n, v in index.items() if v.get("ok")]
    app = FastAPI()
    history: list[str] = []

    @app.get("/")
    def root():
        return HTMLResponse(PAGE)

    @app.get("/img/{name}")
    def img(name: str):
        return FileResponse(os.path.join(REVIEW, os.path.basename(name)))

    @app.get("/next")
    def nxt():
        done = load_verdicts()
        todo = [n for n in renderable if n not in done]
        if not todo:
            counts: dict[str, int] = {}
            for v in done.values():
                counts[v] = counts.get(v, 0) + 1
            return JSONResponse({"done": True, "summary": ", ".join(
                f"{k}: {v}" for k, v in sorted(counts.items()))})
        name = todo[0]
        facts = index[name]
        return JSONResponse({
            "done": False, "file": name, "image": facts["image"],
            "tiles": facts["tiles"], "score": facts["score"],
            "alignment": facts["alignment"], "method": facts["method"],
            "index": len(renderable) - len(todo) + 1, "total": len(renderable),
            "judged": len(done),
        })

    @app.post("/vote")
    async def vote(body: dict):
        save_verdict(body["file"], body["verdict"], index.get(body["file"], {}))
        history.append(body["file"])
        return {"ok": True}

    @app.post("/back")
    def back():
        """Drop the most recent verdict so it can be judged again."""
        if not os.path.exists(VERDICTS):
            return {"ok": False}
        with open(VERDICTS, encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if len(rows) > 1:
            with open(VERDICTS, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows[:-1])
        return {"ok": True}

    import uvicorn
    print(f"\n  open  http://127.0.0.1:{args.port}\n"
          f"  keys: Y correct / N wrong / U unsure / <- back\n"
          f"  verdicts append to {VERDICTS} as you go; Ctrl-C to stop\n")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def report(args) -> None:
    """What the verdicts say, and which measurable signal predicts them."""
    done = load_verdicts()
    if not done:
        raise SystemExit(f"no verdicts yet in {VERDICTS}")
    index = _read_json(INDEX)
    counts: dict[str, int] = {}
    for v in done.values():
        counts[v] = counts.get(v, 0) + 1
    print(f"{len(done)} judged: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    for field in ("tiles", "score", "alignment"):
        print(f"\n  {field} by verdict:")
        for verdict in sorted(set(done.values())):
            vals = [float(index[f][field]) for f, v in done.items()
                    if v == verdict and f in index and index[f].get(field) != ""]
            if vals:
                arr = np.array(vals)
                print(f"    {verdict:7s} n={len(arr):3d}  min {arr.min():7.1f}  "
                      f"median {np.median(arr):7.1f}  max {arr.max():7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["render", "serve", "report"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--again", action="store_true", help="re-render everything")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    {"render": render, "serve": serve, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
