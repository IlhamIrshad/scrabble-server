"""Letter classification by template matching. No training, no dataset.

Scrabble tiles are the easiest case in character recognition: 26 classes, one
known font, always upright, always on a known grid. So we do not need a model
that has learned what letters look like in general - we only need one reference
picture of each letter, normalised the same way the extracted glyphs are.

Templates come from either:
  * a rendered font (default, zero input required), or
  * averaged real crops from your own board, via `build_from_samples`.

The second is strictly better once you have a few crops per letter, and it is
still a reference sheet rather than a training set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import settings
from .grid import normalize_glyph

LETTERS = [chr(ord("A") + i) for i in range(26)]

# English point values. A tile states its letter twice - as the letter, and as a
# number that only some letters share - and the second statement separates exactly
# the pairs overlap struggles with: Q is 10 where O is 1, D is 2, C is 3, and S is 1
# against B's 3.
POINTS = {
    **dict.fromkeys("AEILNORSTU", 1), **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3), **dict.fromkeys("FHVWY", 4),
    "K": 5, **dict.fromkeys("JX", 8), **dict.fromkeys("QZ", 10),
}
# Value banks are keyed by the printed number, as text.
VALUES = [str(v) for v in sorted(set(POINTS.values()))]


@dataclass
class Match:
    letter: str | None
    confidence: float
    best_iou: float
    ranked: list[tuple[str, float]]      # (letter, iou), descending, top 5
    # The point value read off the tile, when one was legible and matched.
    value: str | None = None
    value_iou: float = 0.0


# ---------------------------------------------------------------------------
# building templates
# ---------------------------------------------------------------------------


def _render_letter(letter: str, font: ImageFont.FreeTypeFont, canvas: int) -> np.ndarray:
    img = Image.new("L", (canvas, canvas), 0)
    draw = ImageDraw.Draw(img)
    # anchor="mm" centres on the glyph's own metrics; the subsequent
    # normalise step crops to ink anyway, so exact placement is not critical.
    draw.text((canvas // 2, canvas // 2), letter, fill=255, font=font, anchor="mm")
    return np.asarray(img)


# Tried in order when the configured font cannot be opened. Any plain bold sans
# works; templates are normalised to ink, so small metric differences wash out.
# Arial Bold is actually the closest of these to a real Scrabble tile.
_FONT_CANDIDATES = (
    # Windows
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\ariblk.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def resolve_font(preferred: str | None = None) -> tuple[ImageFont.FreeTypeFont, str]:
    """Load the first usable font, returning it and the path it came from.

    The configured font is only a preference. Falling back matters because the
    default has to name a real file on disk, and no single path exists on Linux,
    Windows and macOS at once - which is exactly how this failed the first time
    it ran outside its container.
    """
    size = int(256 * 0.62)
    tried: list[str] = []
    for path in (preferred, settings.template_font, *_FONT_CANDIDATES):
        if not path or path in tried:
            continue
        tried.append(path)
        try:
            return ImageFont.truetype(path, size), path
        except OSError:
            continue

    raise RuntimeError(
        "No usable font found for building letter templates. Set TEMPLATE_FONT to "
        "a .ttf or .ttc file that exists on this machine - on Windows, "
        r"C:\Windows\Fonts\arialbd.ttf. Tried: " + ", ".join(tried)
    )


def build_from_font(font_path: str | None = None) -> tuple[dict[str, np.ndarray], str]:
    """Render one reference glyph per letter. Returns (templates, font used).

    Rendered crisply and scaled down cleanly, which is not how a photographed
    glyph arrives. Reducing the reference to a real glyph's size and softness first
    sounds like the obvious correction and measures worse - twice the misreads over
    the synthetic set - so it is not done. The one confusion left standing, a 'Q'
    read as an 'O', is a difference of a few percent of overlap; see the note in
    `TemplateBank.match`.
    """
    canvas = 256
    font, used = resolve_font(font_path)
    templates = {ch: normalize_glyph(_render_letter(ch, font, canvas)) for ch in LETTERS}
    return templates, used


def _letter_from_tile(image: np.ndarray) -> np.ndarray | None:
    """Isolate the letter on a picture of a tile, as a binary mask.

    Three things have to be got right and none of them needs to be configured:

    * **Which way the ink runs.** A tile set comes in light-on-dark and
      dark-on-light versions of the same artwork, so the ink is whichever class
      Otsu leaves in the minority - a letter never covers half its tile.
    * **Dropping the point value.** It is a separate connected component, and
      smaller than the letter, so taking the largest component removes it. This is
      the same reasoning as `grid._shed_satellites`, and easier here because the
      two never touch at this resolution.
    * **Ignoring a printed frame.** Some sets draw a border round the tile, which
      is a large dark component; components touching the edge of the image are
      skipped, which leaves the letter.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = dark if dark.mean() < 127 else cv2.bitwise_not(dark)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = ink.shape
    best, best_area = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue
        if area > best_area:
            best, best_area = i, int(area)
    if best is None:
        return None
    return (labels == best).astype(np.uint8) * 255


def build_from_tiles(directory: str) -> dict[str, np.ndarray]:
    """Templates taken from pictures of real tiles - `A.png` ... `Z.png` in a folder.

    Strictly better than rendering a font, and the reason is narrow but expensive:
    a reference glyph only has to be the right *shape*, and no font is the shape of
    the letters on a tile. Measured on the test photographs, a rendered Arial Bold
    reference confuses D with O, C with O and S with B, because those pairs differ
    in a feature whose shape is exactly what varies between typefaces. Real tile
    art has the tiles' own D, and the confusions go away.

    Missing or unreadable letters are skipped rather than fatal, so a partial set
    still helps; `TemplateBank` simply has fewer letters to offer.
    """
    root = Path(directory)
    out: dict[str, np.ndarray] = {}
    for ch in LETTERS:
        for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            path = root / f"{ch}{suffix}"
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mask = _letter_from_tile(image)
            if mask is not None:
                out[ch] = normalize_glyph(mask)
            break
    return out


def _marks_beside_letter(ink: np.ndarray, letter_label: int, labels: np.ndarray,
                         stats: np.ndarray, centroids: np.ndarray) -> np.ndarray | None:
    """Everything in a tile picture that is not the letter, unioned.

    On a tile that is the printed point value and nothing else. A value of 10 is two
    components, so they are unioned rather than taken one at a time - and their
    combined width is itself most of the signal, since no single digit is that wide.
    """
    _, ly, lw, lh, _ = (int(v) for v in stats[letter_label])
    keep = np.zeros(ink.shape, np.uint8)
    found = 0
    for i in range(1, stats.shape[0]):
        if i == letter_label:
            continue
        w, h, area = (int(stats[i][j]) for j in (2, 3, 4))
        if area < 4:
            continue
        # Bigger than the letter, or level with its top, and it is not a digit.
        if max(w, h) >= max(lw, lh):
            continue
        if centroids[i][1] < ly + 0.3 * lh:
            continue
        keep |= (labels == i).astype(np.uint8) * 255
        found += 1
    return keep if found else None


def build_values_from_tiles(directory: str) -> dict[str, list[np.ndarray]]:
    """Digit references, taken from the point values printed on real tiles.

    The labelling comes free: the file is named for its letter, and the letter's
    point value is known, so `Q.png` yields a reference for "10" without anyone
    labelling anything. Several letters share a value, so each value ends up with
    several reference shapes, which is exactly what the matcher wants.
    """
    root = Path(directory)
    out: dict[str, list[np.ndarray]] = {}
    for ch in LETTERS:
        for suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            path = root / f"{ch}{suffix}"
            if not path.exists():
                continue
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            _, dark = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )
            ink = dark if dark.mean() < 127 else cv2.bitwise_not(dark)
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                ink, connectivity=8
            )
            h, w = ink.shape
            letter, best_area = None, 0
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
                    continue
                if area > best_area:
                    letter, best_area = i, int(area)
            if letter is None:
                continue
            marks = _marks_beside_letter(ink, letter, labels, stats, centroids)
            if marks is not None:
                out.setdefault(str(POINTS[ch]), []).append(normalize_glyph(marks))
            break
    return out


def build_values_from_font(font_path: str | None = None) -> dict[str, list[np.ndarray]]:
    """Digit references rendered from a font, so this works with no tile art."""
    canvas = 256
    font, _ = resolve_font(font_path)
    out: dict[str, list[np.ndarray]] = {}
    for value in VALUES:
        out[value] = [normalize_glyph(_render_letter(value, font, canvas))]
    return out


def build_from_samples(samples: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Average already-normalised glyph crops into one template per letter."""
    out: dict[str, np.ndarray] = {}
    for letter, crops in samples.items():
        if not crops:
            continue
        stack = np.stack([c.astype(np.float32) for c in crops])
        out[letter.upper()] = (stack.mean(axis=0) >= 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------


class TemplateBank:
    """Reference glyphs, one or more shapes per letter.

    More than one is allowed because a server does not get to choose which tiles
    are on the table, and the same letter is a different shape in different tile
    sets. A letter scores as its best-matching variant, so adding a set can only
    help the letters it describes better - and offering both a tile set and a
    rendered font is what lets one deployment read boards it was not built for.
    """

    def __init__(
        self, templates: dict[str, np.ndarray | list[np.ndarray]], source: str,
        alphabet: list[str] | None = None,
    ):
        self.source = source
        self.alphabet = LETTERS if alphabet is None else alphabet
        self.letters = [ch for ch in self.alphabet if ch in templates]

        variants: list[np.ndarray] = []
        owner: list[int] = []
        for index, ch in enumerate(self.letters):
            shapes = templates[ch]
            if isinstance(shapes, np.ndarray):
                shapes = [shapes]
            for shape in shapes:
                variants.append(np.asarray(shape).astype(bool))
                owner.append(index)

        self.stack = np.stack(variants)
        self._owner = np.asarray(owner)
        self._counts = self.stack.reshape(len(variants), -1).sum(axis=1)

        # How many holes each letter's reference shapes enclose, measured rather than
        # declared. A table of "O has one, C has none" would be a statement about a
        # typeface, and the whole point of the bank is that it does not know which
        # typeface is on the table - some sets draw a 'G' closed. Taking it from the
        # templates keeps it true for whatever shapes were loaded.
        from . import grid
        self._counters = [set() for _ in self.letters]
        for index, shape in zip(owner, variants):
            self._counters[index].add(grid.counters(shape.astype(np.uint8)))


    # -- persistence ------------------------------------------------------
    @staticmethod
    def _build() -> tuple[dict[str, list[np.ndarray]], str]:
        """Templates and a key describing where they came from.

        Pictures of real tiles win when they are available, because a reference
        glyph only has to be the right shape and a font is not the shape of a tile.
        Falling back to a rendered font matters just as much: it is what makes the
        server start with no assets to install at all.

        More than one folder may be given, and whether adding a second set helps is
        a question for measurement rather than principle - a set that describes your
        tiles badly gives every letter another way to be matched wrongly.
        """
        folders = [f.strip() for f in settings.template_tiles.split(",") if f.strip()]
        gathered: dict[str, list[np.ndarray]] = {}
        used: list[str] = []
        for folder in folders:
            found = build_from_tiles(folder)
            if len(found) < settings.template_tiles_min_letters:
                continue
            used.append(f"{Path(folder).name}({len(found)})")
            for ch, mask in found.items():
                gathered.setdefault(ch, []).append(mask)

        if settings.template_font_fallback or not gathered:
            rendered, font = build_from_font()
            for ch, mask in rendered.items():
                gathered.setdefault(ch, []).append(mask)
            used.append(f"font:{Path(font).name}")

        return gathered, "+".join(used)

    @classmethod
    def load(cls, path: str) -> "TemplateBank":
        """Read a bank saved by `save`, whatever it was built from.

        Used for `settings.template_bank`, which exists because the best reference
        glyphs come from a tile set's own artwork and that artwork lives on the
        machine that has the tile app installed - not in a container. Build the bank
        once with `scripts/build_templates.py`, ship the file, point this at it.
        """
        data = np.load(path, allow_pickle=False)
        templates: dict[str, list[np.ndarray]] = {}
        for key in sorted(data.files):
            letter = key.split("_", 1)[0]
            if letter in LETTERS:
                templates.setdefault(letter, []).append(data[key])
        if not templates:
            raise ValueError(f"{path} holds no letter templates")
        return cls(templates, str(data["source"]) if "source" in data.files else path)

    @classmethod
    def load_or_build(cls, cache: str | None = None) -> "TemplateBank":
        if settings.template_bank:
            # An explicit bank is taken as given. It is not checked against what
            # this machine would build, because the whole point is that it could
            # not build it.
            return cls.load(settings.template_bank)

        cache = cache or settings.template_cache
        templates, origin = cls._build()
        want = f"{origin}:{settings.glyph_size}:{settings.glyph_pad}"

        if cache and os.path.exists(cache):
            try:
                cached = cls.load(cache)
                if cached.source == want:
                    return cached
            except Exception:
                pass

        bank = cls(templates, want)
        if cache:
            try:
                bank.save(cache)
            except OSError:
                # A read-only or unwritable cache path must not stop the server;
                # templates are cheap enough to rebuild on each cold start.
                pass
        return bank

    def save(self, path: str) -> None:
        # Keyed by letter *and* variant number, because a letter may have several
        # reference shapes; keying by letter alone silently kept only one of them.
        seen: dict[str, int] = {}
        arrays: dict[str, np.ndarray] = {}
        for index, mask in enumerate(self.stack):
            letter = self.letters[self._owner[index]]
            n = seen.get(letter, 0)
            seen[letter] = n + 1
            arrays[f"{letter}_{n}"] = mask.astype(np.uint8)

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez_compressed(path, source=np.array(self.source), **arrays)

    # -- matching ---------------------------------------------------------
    def _shifts(self, glyph: np.ndarray) -> np.ndarray:
        """All translated copies of a glyph, as a (k*k, size, size) bool array."""
        k = settings.match_shift
        size = glyph.shape[0]
        offsets = range(-k, k + 1)
        out = np.zeros(((2 * k + 1) ** 2, size, size), dtype=bool)
        g = glyph.astype(bool)
        i = 0
        for dy in offsets:
            for dx in offsets:
                ys0, ys1 = max(0, dy), min(size, size + dy)
                xs0, xs1 = max(0, dx), min(size, size + dx)
                yd0, yd1 = max(0, -dy), min(size, size - dy)
                xd0, xd1 = max(0, -dx), min(size, size - dx)
                out[i, yd0:yd1, xd0:xd1] = g[ys0:ys1, xs0:xs1]
                i += 1
        return out

    def counter_bonus(self, observed: int) -> np.ndarray:
        """Per-letter reward for enclosing as many holes as the glyph being read.

        A reward for agreement rather than a penalty for disagreement, so a glyph
        whose strokes broke badly enough to lose a counter is not actively pushed
        away from the right answer - it simply gets no help.
        """
        return np.array(
            [settings.counter_bonus if observed in seen else 0.0
             for seen in self._counters], dtype=np.float32,
        )

    def match(self, glyph: np.ndarray, bonus: np.ndarray | None = None) -> Match:
        """Best entry for a normalised glyph, by overlap against every variant.

        Overlap treats every pixel alike, which is right for letters that differ
        over their whole shape and is this classifier's weak point for a pair that
        differs in one small feature: 'D' against 'O', 'C' against 'O', 'S' against
        'B', 'Q' against 'O'. What separates each of those pairs is the *shape* of
        one feature - and the shape of a feature is precisely what differs between
        one typeface and another.

        So the first fix for them is not a cleverer comparison, it is a reference
        glyph of the right shape; see `build_from_tiles`. The second is `bonus`, an
        additive per-entry score the caller may supply from evidence outside the
        glyph - in practice the tile's printed point value, which distinguishes
        every one of those pairs. It is added rather than used to filter, so
        evidence that is wrong can only overturn a near-tie.
        """
        if glyph is None or not glyph.any():
            return Match(None, 0.0, 0.0, [])

        shifts = self._shifts(glyph)                       # (K, S, S)
        flat_t = self.stack.reshape(self.stack.shape[0], -1)   # (V, S*S)
        flat_q = shifts.reshape(shifts.shape[0], -1)       # (K, S*S)

        # inter[k, v] and union[k, v] in two matrix products.
        inter = flat_q.astype(np.float32) @ flat_t.T.astype(np.float32)
        q_counts = flat_q.sum(axis=1).astype(np.float32)[:, None]
        union = q_counts + self._counts[None, :].astype(np.float32) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)

        # A letter scores as its best variant, so several reference shapes per
        # letter compete rather than dilute.
        n_letters = len(self.letters)
        if len(self._owner) != n_letters:
            best = np.zeros((iou.shape[0], n_letters), dtype=np.float32)
            np.maximum.at(best.T, self._owner, iou.T)
            iou = best

        per_letter = iou.max(axis=0)                       # best over shifts
        # The overlap a letter achieves on its own, kept apart from any outside
        # evidence: `min_match_iou` is a statement about the glyph, and a bonus must
        # not be able to talk a shapeless blob into being a letter.
        shape_only = per_letter.copy()
        if bonus is not None:
            per_letter = per_letter + bonus

        order = np.argsort(-per_letter)
        ranked = [(self.letters[i], float(per_letter[i])) for i in order[:5]]

        best_iou = float(shape_only[order[0]])
        if shape_only.max() < settings.min_match_iou:
            return Match(None, 0.0, float(shape_only.max()), ranked)

        # Confidence comes from the *shapes*, with the winner as outside evidence
        # ranked it. Outside evidence may say which letter this is; it may never say
        # how sure we are, because the two are not the same question and conflating
        # them breaks the one property the board state is trusted on.
        #
        # The failure is easy to miss. Knowing a glyph encloses a hole pushes 'C',
        # 'G' and twenty others down, so the probability mass piles onto the handful
        # of letters that enclose one - and the winner looks certain while the
        # question that actually mattered, 'O' or 'Q', is untouched. A test caught a
        # 'Q' misread as 'O' arriving with no flag on it at all. Scoring the softmax
        # on shape alone keeps the reported confidence equal to how far the shapes
        # really separate, which for that pair is barely at all.
        t = max(settings.conf_temperature, 1e-3)
        z = (shape_only - shape_only.max()) / t
        p = np.exp(z)
        p /= p.sum()
        return Match(self.letters[order[0]], float(p[order[0]]), best_iou, ranked)


def read_value(values: TemplateBank | None, glyph: np.ndarray | None) -> Match:
    """Read a tile's printed point value from the marks beside its letter."""
    if values is None or glyph is None:
        return Match(None, 0.0, 0.0, [])
    return values.match(glyph)


def digit_count_bonus(letters: list[str], marks: int) -> np.ndarray | None:
    """Additive evidence from *how many* digits the tile's value has, not which.

    The cheapest reliable thing the point value says. Reading the digit itself was
    tried and measured worse than ignoring it: at this resolution the digit is a
    handful of pixels, and a misread one is confident evidence for the wrong letter.
    Counting the marks needs no recognition at all - two marks means a two-digit
    value, and in English only 'Q' and 'Z' are worth ten.

    It earns its place on the one confusion that dominates. 'Q' is an 'O' with a small
    tail, which overlap cannot see and hole-counting cannot either, since both enclose
    exactly one; over 43 boards a person checked cell by cell, 'O' was read where 'Q'
    belonged 22 times, the largest single error left. The count is orthogonal to both.
    """
    if marks < 2:
        # One mark is the ordinary case and says almost nothing - most letters have a
        # single-digit value - so no evidence is offered rather than a bonus for
        # twenty-four letters at once.
        return None
    two_digit = np.array([POINTS.get(ch, 0) >= 10 for ch in letters], dtype=np.float32)
    return settings.digit_count_bonus * two_digit


def value_bonus(letters: list[str], value: str | None, iou: float) -> np.ndarray | None:
    """Additive per-letter evidence from a point value that has been read.

    Only offered when the digit matched a reference well enough to be worth
    believing; and small, so it settles a near-tie and cannot overturn a clear read.
    A digit eight pixels tall is not reliable enough to be given a veto.
    """
    if value is None or iou < settings.value_min_iou:
        return None
    want = int(value)
    agrees = np.array([POINTS.get(ch) == want for ch in letters], dtype=np.float32)
    return settings.value_agreement_bonus * agrees


_bank: TemplateBank | None = None
_values: TemplateBank | None = None
_values_built = False


def reset() -> None:
    """Drop the cached banks, so a settings change takes effect."""
    global _bank, _values, _values_built
    _bank = None
    _values = None
    _values_built = False


def bank() -> TemplateBank:
    global _bank
    if _bank is None:
        _bank = TemplateBank.load_or_build()
    return _bank


def values() -> TemplateBank | None:
    """Digit references for the printed point value, or None if switched off."""
    global _values, _values_built
    if _values_built:
        return _values
    _values_built = True
    if not settings.use_point_value:
        return None

    gathered: dict[str, list[np.ndarray]] = {}
    used: list[str] = []
    for folder in (f.strip() for f in settings.template_tiles.split(",")):
        if not folder:
            continue
        found = build_values_from_tiles(folder)
        if found:
            used.append(Path(folder).name)
            for key, masks in found.items():
                gathered.setdefault(key, []).extend(masks)

    if not gathered:
        gathered = build_values_from_font()
        used.append("font")

    # Every value must be represented, or a tile carrying a missing one would be
    # pushed away from its own letter by the bonus going to everything else.
    if any(v not in gathered for v in VALUES):
        return None
    _values = TemplateBank(gathered, "values:" + "+".join(used), alphabet=VALUES)
    return _values
