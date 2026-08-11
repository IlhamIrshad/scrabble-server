#!/usr/bin/env python3
"""Screen a folder of board photographs without any ground truth.

    python scripts/screen_photos.py photos/ --out fixtures/screen.csv

Labelling a board by hand is expensive, so this measures everything that can be
checked without labels. Three of the checks are *proofs* rather than heuristics -
they can show a reading is wrong, though never that it is right:

* **Tile supply.** An English set holds exactly one Q, one Z, four S and so on. A
  reading using two Qs is wrong in at least one of them, no labels needed. This is
  the strongest label-free signal there is, and it catches precisely the confusions
  template matching is worst at.
* **Connectivity.** Every tile on a legal board connects to the centre square
  through other tiles. A tile alone in open space was not played there.
* **The lexicon.** Every maximal run of two or more letters is a word. Blind to a
  misread that still spells something, and to a wrong cell that forms no run, so
  it is a rate to watch rather than an accuracy.

It also records what the detector believed about each photograph - which proposal
won, how strong the lattice was, how well it aligned - and the tile and board face
colours it measured, which is what lets `--cluster` group the photographs by which
physical board they show. That grouping is the point of the exercise: a corpus of
one board photographed eighty times teaches far less than eighty boards, and the
sample worth labelling is the one that spans them.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.vision import classify, grid, pipeline, registration  # noqa: E402

STILLS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp"}

# An English set, and the reason the supply check works.
SUPPLY = {
    "A": 9, "B": 2, "C": 2, "D": 4, "E": 12, "F": 2, "G": 3, "H": 2, "I": 9,
    "J": 1, "K": 1, "L": 4, "M": 2, "N": 6, "O": 8, "P": 2, "Q": 1, "R": 6,
    "S": 4, "T": 6, "U": 4, "V": 2, "W": 2, "X": 1, "Y": 2, "Z": 1,
}
BLANKS = 2

FIELDS = [
    "file", "width", "height", "ok", "reason", "method", "grid_score",
    "grid_alignment", "period_frac", "locked", "tiles", "blanks", "unknown",
    "low_conf", "words_ok", "words_total", "supply_excess", "islands",
    "loose_tiles", "tile_face", "board_face", "ms",
]


def find_photos(folder: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in STILLS:
            out.append(path)
    return out


def load_lexicon(path: str | None) -> set[str]:
    """Zyzzyva word files are 'WORD definition [pos]', one per line."""
    if not path or not os.path.exists(path):
        return set()
    words = set()
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            head = line.split(" ", 1)[0].strip().upper()
            if head.isalpha():
                words.add(head)
    return words


def runs(board: list[str]) -> list[str]:
    """Every maximal run of two or more cells, across and down."""
    grids = [board, ["".join(board[r][c] for r in range(15)) for c in range(15)]]
    out = []
    for lines in grids:
        for line in lines:
            for part in "".join(ch if ch != "." else " " for ch in line).split():
                if len(part) >= 2:
                    out.append(part)
    return out


def supply_excess(board: list[str]) -> str:
    """Letters used more often than a set contains. Any entry proves a misread."""
    used = Counter(ch for row in board for ch in row if ch.isalpha())
    over = {ch: n - SUPPLY[ch] for ch, n in used.items() if n > SUPPLY.get(ch, 0)}
    blanks = sum(row.count("?") for row in board)
    if blanks > BLANKS:
        over["?"] = blanks - BLANKS
    return ";".join(f"{ch}+{n}" for ch, n in sorted(over.items()))


def islands(board: list[str]) -> tuple[int, int]:
    """(groups of connected tiles, tiles not in the group holding the centre)."""
    occupied = {(r, c) for r in range(15) for c in range(15) if board[r][c] != "."}
    seen: set[tuple[int, int]] = set()
    groups = []
    for start in occupied:
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in occupied and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(group)
    if not groups:
        return 0, 0
    main = max(groups, key=len)
    if (7, 7) in occupied:
        main = next((g for g in groups if (7, 7) in g), main)
    return len(groups), len(occupied) - len(main)


def measured_colours(cells: list[grid.Cell]) -> tuple[str, str]:
    """The tile and board face colours the pipeline settled on, for clustering."""
    tol = settings.tile_face_delta_e
    tile = grid._dominant_colour(
        [c.face for c in cells if c.occupied and c.face is not None], tol
    )
    board = grid._dominant_colour(
        [c.face for c in cells if not c.occupied and c.face is not None], tol
    )
    fmt = lambda v: "" if v is None else ",".join(f"{x:.1f}" for x in v)  # noqa: E731
    return fmt(tile), fmt(board)


def _cells_for_colours(image: np.ndarray, result) -> list[grid.Cell]:
    """Re-warp with the chosen corners just to recover the per-cell face colours.

    The response does not carry them, and they are only wanted for grouping the
    corpus by which physical board it shows.
    """
    corners = np.array(result.registration.corners, np.float32)
    board_img, _ = registration.warp_board(image, corners)
    return grid.extract_cells(board_img)[0]


def screen(path: str, bank, values, lexicon: set[str]) -> dict:
    row = dict.fromkeys(FIELDS, "")
    row["file"] = os.path.basename(path)
    started = time.perf_counter()

    image = cv2.imread(path)
    if image is None:
        row.update(ok=0, reason="unreadable")
        return row
    row["height"], row["width"] = image.shape[:2]

    # Through the real entry point, so this screens what actually ships rather
    # than a re-implementation of it that can drift.
    try:
        result = pipeline.detect(image)
    except registration.RegistrationError:
        row.update(ok=0, reason="no board found",
                   ms=round((time.perf_counter() - started) * 1000))
        return row

    reg, info, board = result.registration, result.registration.grid, result.board
    low = len(result.low_confidence)
    cells = _cells_for_colours(image, result)
    checkable = [w for w in runs(board) if w.isalpha()]
    groups, loose = islands(board)
    tile_face, board_face = measured_colours(cells)

    row.update(
        ok=1, method=reg.method, grid_score=round(reg.grid_score, 2),
        grid_alignment=round(reg.grid_alignment, 2),
        period_frac=info.get("period_frac", ""), locked=int(bool(info.get("locked"))),
        tiles=sum(ch != "." for r in board for ch in r),
        blanks=sum(r.count("?") for r in board),
        unknown=sum(r.count("*") for r in board),
        low_conf=low,
        words_ok=sum(1 for w in checkable if w in lexicon) if lexicon else "",
        words_total=len(checkable) if lexicon else "",
        supply_excess=supply_excess(board), islands=groups, loose_tiles=loose,
        tile_face=tile_face, board_face=board_face,
        ms=round((time.perf_counter() - started) * 1000),
    )
    return row


def cluster(rows: list[dict], tol: float) -> list[list[dict]]:
    """Group photographs by the tile and board colours measured in them."""
    usable = [r for r in rows if r.get("tile_face") and r.get("board_face")]
    points, clusters = [], []
    for r in usable:
        v = np.array([float(x) for x in
                      (r["tile_face"] + "," + r["board_face"]).split(",")])
        for centre, members in clusters:
            if float(np.linalg.norm(v - centre)) <= tol:
                members.append(r)
                break
        else:
            clusters.append((v, [r]))
        points.append(v)
    return [members for _, members in
            sorted(clusters, key=lambda c: -len(c[1]))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default="fixtures/screen.csv")
    ap.add_argument("--lexicon", default=None,
                    help="a word list, one word per line or Zyzzyva format")
    ap.add_argument("--cluster-tol", type=float, default=18.0,
                    help="Delta-E within which two photographs are the same board")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    photos = find_photos(args.folder)
    if args.limit:
        photos = photos[: args.limit]
    lexicon = load_lexicon(args.lexicon)
    print(f"{len(photos)} photographs, "
          f"lexicon: {len(lexicon) or 'none - skipping the word check'}")

    bank = classify.bank()
    values = classify.values()
    print(f"templates: {bank.source}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows: list[dict] = []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for i, path in enumerate(photos, 1):
            row = screen(path, bank, values, lexicon)
            rows.append(row)
            writer.writerow(row)
            fh.flush()          # so a long run can be watched, and survives a kill
            flag = "" if row["ok"] else f"  <-- {row['reason']}"
            print(f"  [{i:3d}/{len(photos)}] {row['file'][:34]:36s} "
                  f"tiles={str(row['tiles']):>3} score={str(row['grid_score']):>7} "
                  f"{row['ms']}ms{flag}", flush=True)

    report(rows, lexicon, args.cluster_tol, args.out)


def report(rows: list[dict], lexicon: set[str], tol: float, out: str) -> None:
    ok = [r for r in rows if r["ok"]]
    print(f"\n{'=' * 70}\nregistration")
    print(f"  found a board on          {len(ok)}/{len(rows)}")
    for r in rows:
        if not r["ok"]:
            print(f"    FAILED  {r['file']}  ({r['reason']})")
    if ok:
        scores = np.array([float(r["grid_score"]) for r in ok])
        print(f"  grid score  min {scores.min():.1f}   "
              f"median {np.median(scores):.1f}   max {scores.max():.1f}")
        print(f"  below {settings.min_grid_score} (would be rejected): "
              f"{int((scores < settings.min_grid_score).sum())}")
        print(f"  grid lock held on         {sum(int(r['locked']) for r in ok)}/{len(ok)}")
        methods = Counter(str(r["method"]).split("[")[0] for r in ok)
        print(f"  winning detector          "
              f"{', '.join(f'{k}={v}' for k, v in methods.most_common())}")

        tiles = np.array([int(r["tiles"]) for r in ok])
        print(f"\ntiles per board  min {tiles.min()}  median {int(np.median(tiles))}  "
              f"max {tiles.max()}")

        print("\nlabel-free checks that can prove a reading wrong")
        bad_supply = [r for r in ok if r["supply_excess"]]
        print(f"  impossible tile counts    {len(bad_supply)}/{len(ok)} boards")
        for r in bad_supply[:10]:
            print(f"    {r['file'][:38]:40s} {r['supply_excess']}")
        loose = [r for r in ok if int(r["loose_tiles"]) > 0]
        print(f"  tiles not joined to the rest {len(loose)}/{len(ok)} boards"
              + (f"  (worst {max(int(r['loose_tiles']) for r in loose)})" if loose else ""))

        if lexicon:
            good = sum(int(r["words_ok"]) for r in ok)
            total = sum(int(r["words_total"]) for r in ok)
            print(f"\nlexicon   {good}/{total} runs are real words "
                  f"({100.0 * good / max(total, 1):.1f}%)")
            per = sorted(
                ((int(r["words_ok"]) / max(int(r["words_total"]), 1), r) for r in ok
                 if int(r["words_total"]) >= 5), key=lambda t: t[0])
            print("  worst boards:")
            for rate, r in per[:8]:
                print(f"    {100 * rate:5.1f}%  {r['file'][:38]:40s} "
                      f"tiles={r['tiles']} score={r['grid_score']}")

        groups = cluster(ok, tol)
        print(f"\nboards present: {len(groups)} colour groups "
              f"(tile+board face within {tol} Delta-E)")
        for n, members in enumerate(groups, 1):
            rep = members[0]
            rate = ""
            if lexicon:
                g = sum(int(m["words_ok"]) for m in members)
                t = sum(int(m["words_total"]) for m in members)
                rate = f"  words {100.0 * g / max(t, 1):5.1f}%"
            print(f"  group {n:2d}  {len(members):3d} photos  "
                  f"tile={rep['tile_face']:18s} board={rep['board_face']:18s}{rate}")
            print(f"            e.g. {', '.join(m['file'] for m in members[:3])}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
