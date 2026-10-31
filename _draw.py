"""Tiny PIL drawing kit shared by make_readme_assets.py and make_demo_video.py.

Everything is drawn at 2x and downsampled for crisp edges. Colors match specdec.style.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SURFACE = "#fcfcfb"
PLANE = "#f4f3ef"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # n-gram
ORANGE = "#eb6834"    # draft model
AQUA = "#1baf7a"      # suffix
GOOD = "#0ca30c"      # accepted (status: good) - always paired with a check icon
BAD = "#d03b3b"       # rejected (status: critical) - always paired with a cross icon
GOOD_TINT = "#e6f5e6"
BAD_TINT = "#fbe6e6"
BLUE_TINT = "#e3eefb"
ORANGE_TINT = "#fdeee7"
NEUTRAL_TINT = "#f0efec"
TERM_BG = "#141413"
TERM_FG = "#e8e6df"
TERM_DIM = "#8f8d85"
TERM_GREEN = "#7fd18b"
TERM_BLUE = "#7fb2f0"
TERM_ORANGE = "#f0a57f"

_FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/Library/Fonts", str(Path.home() / "Library/Fonts"),
              "/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts"]
_WEIGHTS = {"regular": ["Inter-Regular.otf", "Inter-Regular.ttf", "DejaVuSans.ttf", "arial.ttf"],
            "medium": ["Inter-Medium.otf", "Inter-Medium.ttf", "DejaVuSans.ttf", "arial.ttf"],
            "semibold": ["Inter-SemiBold.otf", "Inter-SemiBold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
            "bold": ["InterDisplay-Bold.otf", "Inter-Bold.otf", "Inter-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf"],
            "mono": ["DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf"],
            "mono_bold": ["DejaVuSansMono-Bold.ttf", "Menlo.ttc", "consolab.ttf"]}


@lru_cache(maxsize=None)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    for name in _WEIGHTS[weight]:
        for d in _FONT_DIRS:
            p = Path(d) / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


class Canvas:
    """Draw in logical px; internally rendered at `scale`x then downsampled."""

    def __init__(self, w: int, h: int, bg: str = SURFACE, scale: int = 2):
        self.w, self.h, self.s = w, h, scale
        self.img = Image.new("RGB", (w * scale, h * scale), bg)
        self.d = ImageDraw.Draw(self.img)

    def _p(self, *xy):
        return [v * self.s for v in xy]

    def font(self, size, weight="regular"):
        return font(int(size * self.s), weight)

    def text(self, x, y, s, size=16, weight="regular", fill=INK, anchor="la"):
        self.d.text(self._p(x, y), s, font=self.font(size, weight), fill=fill, anchor=anchor)

    def text_w(self, s, size=16, weight="regular") -> float:
        return self.d.textlength(s, font=self.font(size, weight)) / self.s

    def rect(self, x, y, w, h, r=8, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self._p(x, y, x + w, y + h), radius=r * self.s, fill=fill,
                                 outline=outline, width=max(1, int(width * self.s)))

    def dashed_rect(self, x, y, w, h, color=MUTED, width=1.5, dash=6, gap=4, r=8, fill=None):
        if fill:
            self.rect(x, y, w, h, r=r, fill=fill)
        segs = [((x + r, y), (x + w - r, y)), ((x + r, y + h), (x + w - r, y + h)),
                ((x, y + r), (x, y + h - r)), ((x + w, y + r), (x + w, y + h - r))]
        for (x0, y0), (x1, y1) in segs:
            length = abs(x1 - x0) + abs(y1 - y0)
            pos = 0.0
            while pos < length:
                end = min(pos + dash, length)
                t0, t1 = pos / length, end / length
                self.line(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1,
                          color=color, width=width)
                pos += dash + gap
        for cx, cy, a0 in ((x + r, y + r, 180), (x + w - r, y + r, 270), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90)):
            self.d.arc(self._p(cx - r, cy - r, cx + r, cy + r), a0, a0 + 90, fill=color, width=int(width * self.s))

    def line(self, x0, y0, x1, y1, color=INK, width=2):
        self.d.line(self._p(x0, y0, x1, y1), fill=color, width=max(1, int(width * self.s)))

    def circle(self, cx, cy, r, fill=None, outline=None, width=1):
        self.d.ellipse(self._p(cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline,
                       width=max(1, int(width * self.s)))

    def arrow(self, x0, y0, x1, y1, color=INK_2, width=2, head=8):
        import math
        self.line(x0, y0, x1, y1, color, width)
        ang = math.atan2(y1 - y0, x1 - x0)
        pts = [(x1, y1), (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45)),
               (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))]
        self.d.polygon([v * self.s for p in pts for v in p], fill=color)

    def check(self, cx, cy, size=8, color=GOOD, width=2.4):
        self.d.line(self._p(cx - size * 0.9, cy, cx - size * 0.25, cy + size * 0.65, cx + size, cy - size * 0.75),
                    fill=color, width=int(width * self.s), joint="curve")

    def cross(self, cx, cy, size=7, color=BAD, width=2.4):
        self.line(cx - size, cy - size, cx + size, cy + size, color, width)
        self.line(cx - size, cy + size, cx + size, cy - size, color, width)

    def finish(self, out_scale: int = 1) -> Image.Image:
        size = (self.w * out_scale, self.h * out_scale)
        return self.img if size == self.img.size else self.img.resize(size, Image.LANCZOS)


TOKEN_STYLES = {
    # state: (fill, outline, text color, dashed)
    "prompt": (NEUTRAL_TINT, None, INK_2, False),
    "target": (INK, None, "#ffffff", False),
    "draft": (SURFACE, MUTED, INK, True),
    "accepted": (GOOD_TINT, GOOD, INK, False),
    "rejected": (BAD_TINT, BAD, BAD, False),
    "ghost": (SURFACE, GRID, MUTED, True),
}


def token(c: Canvas, x, y, s, state="target", size=18, h=40, pad=12, icon=True) -> float:
    """Draw one token chip; returns its width. Accepted/rejected chips carry an icon, not just color."""
    fill, outline, fg, dashed = TOKEN_STYLES[state]
    s = s.strip()  # BPE tokens often carry a leading space; drop it for display
    tw = c.text_w(s, size, "mono")
    extra = 22 if icon and state in ("accepted", "rejected") else 0
    w = tw + 2 * pad + extra
    if dashed:
        c.dashed_rect(x, y, w, h, color=outline, fill=fill, r=7)
    else:
        c.rect(x, y, w, h, r=7, fill=fill, outline=outline, width=1.5 if outline else 1)
    c.text(x + pad, y + h / 2, s, size, "mono", fg, anchor="lm")
    if state == "rejected":
        c.line(x + pad - 2, y + h / 2, x + pad + tw + 2, y + h / 2, BAD, 1.6)
    if extra:
        ix = x + pad + tw + 13
        (c.check if state == "accepted" else c.cross)(ix, y + h / 2, size=6)
    return w
