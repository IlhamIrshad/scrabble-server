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
        errors.append((error_in_cells(got, want, truth[name]["cell_px"]),
                       name, result.registration.method))

    if not errors:
        raise SystemExit("nothing scored")
    arr = np.array([e for e, _, _ in errors])
    print(f"registration failed outright on {len(failed)}")
    print(f"corner error in cells:  median {np.median(arr):.3f}   "
          f"90th {np.quantile(arr, 0.9):.3f}   worst {arr.max():.3f}")
    for limit in (0.15, 0.33, 0.5, 1.0):
        print(f"  within {limit:4.2f} of a cell: {int((arr <= limit).sum()):3d}/{len(arr)}")

    print(f"\nworst {args.worst}:")
    for err, name, method in sorted(errors, reverse=True)[: args.worst]:
        print(f"  {err:6.2f} cells  {name[:38]:40s} {method}")


if __name__ == "__main__":
    main()
