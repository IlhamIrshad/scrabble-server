#!/usr/bin/env python3
"""Turn accepted registration verdicts into durable ground truth.

    python scripts/harvest_corners.py

A pass/fail verdict judges one particular output, so it expires the moment the
detector changes. The *corners* behind a pass do not: if a person confirmed that
the grid sat on the cells, then those four corners were where the board was, and
they stay correct for ever.

So this re-runs the detector over every photograph marked `yes`, records the
corners it chose, and writes them to fixtures/corners.json. Any later change can
then be scored automatically against real photographs - "how far, in cells, does
the new answer sit from the confirmed one" - instead of costing another pass of
manual review every time something is tried.

Only `yes` verdicts are harvested. A `no` tells us the corners were wrong but not
where the right ones were, which is a different and much slower thing to collect.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision import pipeline, registration

PHOTOS = "photos"
VERDICTS = os.path.join("fixtures", "registration_verdicts.csv")
OUT = os.path.join("fixtures", "corners.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(VERDICTS, encoding="utf-8") as fh:
        verdicts = {r["file"]: r["verdict"] for r in csv.DictReader(fh)}
    accepted = sorted(f for f, v in verdicts.items() if v == "yes")
    print(f"{len(accepted)} photographs were confirmed correct; harvesting corners")

    truth: dict[str, dict] = {}
    for i, name in enumerate(accepted, 1):
        image = cv2.imread(os.path.join(PHOTOS, name))
        if image is None:
            print(f"  [{i}/{len(accepted)}] {name}  UNREADABLE")
            continue
        try:
            # Through `pipeline.detect`, because that is what rendered the images
            # the verdicts were given on. `find_board` returns only the first guess
            # and the pipeline may read several and choose a different one, so
            # harvesting through it would record corners nobody ever approved.
            result = pipeline.detect(image)
        except registration.RegistrationError as exc:
            # Possible if the detector has changed since the review.
            print(f"  [{i}/{len(accepted)}] {name}  NO BOARD NOW: {str(exc)[:60]}")
            continue
        corners = registration.order_corners(
            np.array(result.registration.corners, np.float32)
        )
        side = float(np.linalg.norm(corners[1] - corners[0]))
        truth[name] = {
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in corners],
            "cell_px": round(side / 15.0, 2),
            "height": image.shape[0], "width": image.shape[1],
            "method_when_harvested": result.registration.method,
        }
        print(f"  [{i}/{len(accepted)}] {name}  cell={truth[name]['cell_px']}px",
              flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1)
    print(f"\nwrote {args.out} with {len(truth)} confirmed boards")
    print("score any future change against it with scripts/score_corners.py")


if __name__ == "__main__":
    main()
