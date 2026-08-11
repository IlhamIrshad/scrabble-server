#!/usr/bin/env python3
"""Build a template bank from pictures of real tiles, and save it.

A reference glyph only has to be the right *shape*, and no font is the shape of
the letters on a tile. Rendered references confuse D with O, C with O and S with
B - pairs that differ in one feature whose shape is exactly what varies between
typefaces - and those confusions were most of the remaining letter errors on real
photographs. A tile set's own artwork does not have them.

    python scripts/build_templates.py out/protiles.npz \\
        "C:/Program Files/NASPA Zyzzyva 3.4.2/data/tiles/white-pro"

Then point the server at the result, which is how tile art reaches a container
that has no tile app installed:

    TEMPLATE_BANK=out/protiles.npz

Any folder of images named A.png ... Z.png will do - one file per letter, cropped
to the tile. Several folders may be given and a letter then scores as its
best-matching shape across all of them. The rendered font is included as one more
shape per letter unless --no-font is passed: measured, that is strictly better
than either source alone, because it also covers boards set in a typeface the tile
art does not describe.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vision.classify import (  # noqa: E402
    LETTERS,
    TemplateBank,
    build_from_font,
    build_from_tiles,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="path of the .npz to write")
    ap.add_argument("tiles", nargs="*", help="folders of A.png ... Z.png tile images")
    ap.add_argument("--no-font", action="store_true",
                    help="do not include the rendered font as a fallback shape")
    args = ap.parse_args()

    gathered: dict[str, list] = {}
    used: list[str] = []
    for folder in args.tiles:
        found = build_from_tiles(folder)
        missing = [ch for ch in LETTERS if ch not in found]
        print(f"{folder}: {len(found)} letters"
              + (f", missing {''.join(missing)}" if missing else ""))
        if not found:
            continue
        used.append(f"{os.path.basename(folder.rstrip('/\\'))}({len(found)})")
        for ch, mask in found.items():
            gathered.setdefault(ch, []).append(mask)

    if not args.no_font:
        rendered, font = build_from_font()
        for ch, mask in rendered.items():
            gathered.setdefault(ch, []).append(mask)
        used.append(f"font:{os.path.basename(font)}")
        print(f"{font}: 26 letters (fallback)")

    if not gathered:
        raise SystemExit("nothing to build: give a folder of tile images")

    bank = TemplateBank(gathered, "tiles:" + "+".join(used))
    bank.save(args.out)
    print(f"\nwrote {args.out}\n  source: {bank.source}\n  "
          f"{len(bank.letters)} letters, {bank.stack.shape[0]} reference shapes")


if __name__ == "__main__":
    main()
