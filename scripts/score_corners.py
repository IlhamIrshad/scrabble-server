#!/usr/bin/env python3
"""Score registration against the confirmed corners, in cells.

    python scripts/score_corners.py

Pixels are the wrong unit: the same error matters more on a small board than a
large one. So error is reported as a fraction of a cell, which is the unit that
decides whether a glyph lands in its own cell or its neighbour's. Anything under
about a third of a cell reads correctly; anything over it starts losing tiles, and
a whole cell out relabels the entire board.

Run it before and after any change to registration. It needs no review and no
labelling: the ground truth was collected once by `harvest_corners.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision import pipeline, registration

PHOTOS = "photos"
TRUTH = os.path.join("fixtures", "corners.json")


def error_in_cells(got: np.ndarray, want: np.ndarray, cell_px: float) -> float:
    """Largest corner displacement, in cells."""
    a = registration.order_corners(got)
    b = registration.order_corners(want)
    return float(np.linalg.norm(a - b, axis=1).max() / max(cell_px, 1e-6))


def grid_error_in_cells(
    image: np.ndarray, got: np.ndarray, want: np.ndarray, cell_px: float
) -> float:
    """Largest displacement between the two registrations' cell centres, in cells.

    The figure to judge a change on. Corner displacement measures the quads that go
    *in*, and the grid lock then re-fits pitch and phase, so it can absorb a large
    corner difference or manufacture one: among the boards where two confirmed
    answers disagree, one pair 1.67 cells apart at the corners lands 0.15 apart cell
    for cell, and another 0.37 apart lands 1.66 apart. Cell centres are what decide
    which letter is recorded where, so they are what gets scored.

    Both registrations are put through the *current* code, so this asks whether today
    it puts the cells where the confirmed answer put them - not whether it reproduces
    a quad that some earlier version happened to propose.
    """
    a = registration.cell_centres(image, got)
    b = registration.cell_centres(image, want)
    return float(np.linalg.norm(a - b, axis=1).max() / max(cell_px, 1e-6))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", default=TRUTH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    truth = json.load(open(args.truth, encoding="utf-8"))
    names = sorted(truth)[: args.limit or None]
    print(f"scoring {len(names)} confirmed boards\n")

    errors, failed = [], []
    for name in names:
        image = cv2.imread(os.path.join(PHOTOS, name))
        if image is None:
            continue
        want = np.array(truth[name]["corners"], np.float32)
        try:
            # The shipping path, not `find_board`. The pipeline reads several
            # candidates and may choose a different one, so scoring `find_board`
            # measures something that is never served to anybody.
            result = pipeline.detect(image)
        except registration.RegistrationError:
            failed.append(name)
            continue
        got = np.array(result.registration.corners, np.float32)
        cell_px = truth[name]["cell_px"]
        errors.append((grid_error_in_cells(image, got, want, cell_px),
                       error_in_cells(got, want, cell_px),
                       name, result.registration.method))

    if not errors:
        raise SystemExit("nothing scored")
    grids = np.array([g for g, _, _, _ in errors])
    quads = np.array([c for _, c, _, _ in errors])
    print(f"registration failed outright on {len(failed)}")
    for label, arr in (("cell centres", grids), ("quad corners", quads)):
        print(f"{label:14s}  median {np.median(arr):.3f}   "
              f"90th {np.quantile(arr, 0.9):.3f}   worst {arr.max():.3f}")
    print("\nby how far the cells moved - the figure that decides the transcript:")
    for limit in (0.15, 0.33, 0.5, 1.0):
        print(f"  within {limit:4.2f} of a cell: {int((grids <= limit).sum()):3d}/{len(grids)}")

    print(f"\nworst {args.worst}:")
    for g, c, name, method in sorted(errors, reverse=True)[: args.worst]:
        print(f"  cells {g:6.2f}  corners {c:6.2f}   {name[:32]:34s} {method}")


if __name__ == "__main__":
    main()
