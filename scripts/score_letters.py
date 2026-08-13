#!/usr/bin/env python3
"""Score the reader against the cells a person actually checked.

    python scripts/score_letters.py

Ground truth comes from `review_letters.py`: on a board a person confirmed, every
cell read correctly except the ones they corrected. So the truth board is the board
the pipeline produced *at review time* with those corrections applied - which is why
the corrections file records what was read as well as what was there, and why a
correction that no longer matches what the pipeline reads is reported rather than
silently assumed still to apply.

Reports per-cell accuracy split by the thing that went wrong, because the classes
have completely different causes and fixes: whether a cell holds a tile at all,
whether a tile is blank, and which letter is on it.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.review_letters as review
from app.vision import pipeline, registration

EMPTY, BLANK = ".", "?"


def truth_boards() -> dict[str, list[list[str]]]:
    """The confirmed board for every reviewed photograph, corrections applied."""
    index = review._read_json(review.INDEX)
    done = review.load_done()
    fixes = defaultdict(list)
    for f in review._corrections():
        fixes[f["file"]].append(f)

    out: dict[str, list[list[str]]] = {}
    for name in done:
        if name not in index:
            continue
        board = [list(row) for row in index[name]["board"]]
        for f in fixes[name]:
            board[int(f["row"])][int(f["col"])] = f["truth"]
        out[name] = board
    return out


def classify(read: str, want: str) -> str | None:
    if read == want:
        return None
    if read == EMPTY:
        return "missed a blank" if want == BLANK else "missed a lettered tile"
    if want == EMPTY:
        return "phantom tile"
    if read == BLANK:
        return "called blank, was a letter"
    if want == BLANK:
        return "called a letter, was blank"
    if read == "*":
        return "unreadable, but was a letter"
    return "wrong letter"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--confusions", type=int, default=12)
    ap.add_argument("--only", default=None,
                    help="path to a file listing the photographs to score, one per line")
    args = ap.parse_args()

    truth = truth_boards()
    only = review.only_filter(args.only)
    names = sorted(n for n in truth if only is None or n in only)[: args.limit or None]
    if not names:
        raise SystemExit("no confirmed boards; run review_letters.py first")

    kinds: Counter = Counter()
    confusion: Counter = Counter()
    cells = tiles = 0
    stale = 0
    for i, name in enumerate(names, 1):
        image = cv2.imread(os.path.join(review.PHOTOS, name))
        if image is None:
            continue
        try:
            result = pipeline.detect(image)
        except registration.RegistrationError:
            stale += 1
            continue
        got = [list(row) for row in result.board]
        want = truth[name]
        for r in range(15):
            for c in range(15):
                cells += 1
                if want[r][c] != EMPTY:
                    tiles += 1
                kind = classify(got[r][c], want[r][c])
                if kind:
                    kinds[kind] += 1
                    confusion[(got[r][c], want[r][c])] += 1
        print(f"  [{i}/{len(names)}] {name}", flush=True)

    errors = sum(kinds.values())
    print(f"\n{len(names)} confirmed boards, {cells} cells, {tiles} of them holding a tile")
    print(f"  cell accuracy   {1 - errors / max(cells, 1):.3%}   ({errors} wrong)")
    placed = sum(v for k, v in kinds.items() if "missed" in k or k == "phantom tile")
    print(f"  of those, {placed} are about whether a cell holds a tile "
          f"and {errors - placed} about what is on it")
    if stale:
        print(f"  {stale} boards no longer register at all and were not scored")

    print("\n  by kind:")
    for k, v in kinds.most_common():
        print(f"    {k:30s} {v:4d}")

    print("\n  confusions (read -> actually):")
    for (got_ch, want_ch), n in confusion.most_common(args.confusions):
        print(f"    {got_ch} -> {want_ch}   x{n}")


if __name__ == "__main__":
    main()
