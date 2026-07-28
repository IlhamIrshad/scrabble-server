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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import settings
from .grid import normalize_glyph

LETTERS = [chr(ord("A") + i) for i in range(26)]


@dataclass
class Match:
    letter: str | None
    confidence: float
    best_iou: float
    ranked: list[tuple[str, float]]      # (letter, iou), descending, top 5


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
    """Render one reference glyph per letter. Returns (templates, font used)."""
    canvas = 256
    font, used = resolve_font(font_path)
    templates = {ch: normalize_glyph(_render_letter(ch, font, canvas)) for ch in LETTERS}
    return templates, used


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
    def __init__(self, templates: dict[str, np.ndarray], source: str):
        self.source = source
        self.letters = [ch for ch in LETTERS if ch in templates]
        self.stack = np.stack([templates[ch].astype(bool) for ch in self.letters])
        self._counts = self.stack.reshape(len(self.letters), -1).sum(axis=1)

    # -- persistence ------------------------------------------------------
    @classmethod
    def load_or_build(cls, cache: str | None = None) -> "TemplateBank":
        cache = cache or settings.template_cache
        templates, used = build_from_font()
        want = f"font:{used}:{settings.glyph_size}:{settings.glyph_pad}"

        if cache and os.path.exists(cache):
            try:
                data = np.load(cache, allow_pickle=False)
                if str(data["source"]) == want:
                    tpl = {k: data[k] for k in data.files if k in LETTERS}
                    if tpl:
                        return cls(tpl, want)
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
        arrays = {ch: self.stack[i].astype(np.uint8) for i, ch in enumerate(self.letters)}
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

    def match(self, glyph: np.ndarray) -> Match:
        if glyph is None or not glyph.any():
            return Match(None, 0.0, 0.0, [])

        shifts = self._shifts(glyph)                       # (K, S, S)
        n_letters = len(self.letters)
        flat_t = self.stack.reshape(n_letters, -1)         # (L, S*S)
        flat_q = shifts.reshape(shifts.shape[0], -1)       # (K, S*S)

        # inter[k, l] and union[k, l] in two matrix products.
        inter = flat_q.astype(np.float32) @ flat_t.T.astype(np.float32)
        q_counts = flat_q.sum(axis=1).astype(np.float32)[:, None]
        union = q_counts + self._counts[None, :].astype(np.float32) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0)

        per_letter = iou.max(axis=0)                       # best over shifts
        order = np.argsort(-per_letter)
        ranked = [(self.letters[i], float(per_letter[i])) for i in order[:5]]

        best_iou = float(per_letter[order[0]])
        if best_iou < settings.min_match_iou:
            return Match(None, 0.0, best_iou, ranked)

        t = max(settings.conf_temperature, 1e-3)
        z = (per_letter - per_letter.max()) / t
        p = np.exp(z)
        p /= p.sum()
        return Match(self.letters[order[0]], float(p[order[0]]), best_iou, ranked)


_bank: TemplateBank | None = None


def bank() -> TemplateBank:
    global _bank
    if _bank is None:
        _bank = TemplateBank.load_or_build()
    return _bank
