"""Render the README demo video (MP4) and a short GIF of the decoding race. No GPU needed.

    python scripts/make_demo_video.py            # -> docs/assets/demo.mp4, demo-race.gif, demo-poster.png

Scenes
  1 title
  2 plain vs speculative decoding race  (illustration of the mechanism, not a measurement)
  3 the knobs: speculative_config for n-gram and draft-model
  4 terminal: REAL output of the CPU commands, captured live when this script runs
  5 what the repo does + what theory predicts
  6 how to run it
Needs ffmpeg on PATH.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _draw import (AXIS, BLUE, BLUE_TINT, GOOD_TINT, GRID, INK, INK_2, MUTED, ORANGE,  # noqa: E402
                   PLANE, SURFACE, TERM_BG, TERM_BLUE, TERM_DIM, TERM_FG, TERM_GREEN, TERM_ORANGE, Canvas,
                   token)

OUT = ROOT / "docs" / "assets"
W, H, FPS = 1280, 720, 30
OUT_SIZE = (1920, 1080)
REPO = "github.com/AhmedMujtaba999/specdec-vllm"

TERM: dict = {}  # command -> captured output lines (filled in main, shared with workers)


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def seg(t: float, a: float, b: float) -> float:
    """0 before a, 1 after b, eased in between."""
    return ease((t - a) / (b - a)) if b > a else float(t >= a)


def paste(c: Canvas, img: Image.Image, x, y, w):
    h = img.height * w / img.width
    im = img.convert("RGB").resize((int(w * c.s), int(h * c.s)), Image.LANCZOS)
    c.img.paste(im, (int(x * c.s), int(y * c.s)))
    return h


# --------------------------------------------------------------------------- scene 1: title
def scene_title(c: Canvas, t: float, dur: float):
    c.rect(0, 0, W / 2, 6, r=0, fill=BLUE)
    c.rect(W / 2, 0, W / 2, 6, r=0, fill=ORANGE)
    a = seg(t, 0.1, 0.6)
    if a:
        c.text(96, 250, "LLM INFERENCE  ·  vLLM  ·  BENCHMARK", 15, "semibold", MUTED)
    if seg(t, 0.3, 0.9):
        c.text(96, 280, "Speculative decoding with vLLM", 56, "bold", INK)
    if seg(t, 0.7, 1.3):
        c.text(96, 366, "Baseline vs n-gram vs draft-model · speedup, acceptance, correctness", 22, "regular", INK_2)
    items = [("tax", "prompt"), (" =", "accepted"), (" subtotal", "accepted"), (" *", "accepted"),
             (" discount", "rejected"), (" tax_rate", "target")]
    x = 96
    for i, (s, state) in enumerate(items):
        if t < 1.4 + i * 0.22:
            break
        x += token(c, x, 460, s, state, size=20, h=46) + 10
    if t > 1.4 + len(items) * 0.22 + 0.2:
        c.text(x + 10, 483, "→  4 tokens from 1 target pass", 22, "semibold", INK, anchor="lm")


# --------------------------------------------------------------------------- scene 2: race
P = 0.75     # seconds per target forward pass (same clock for both lanes)
START = 1.2  # lanes start after the intro
OUTPUT = ["def", " compute_invoice_total", "(items", ",", " tax_rate", ",", " discount", "=None", "):",
          " subtotal", " =", " 0", " for"]
# Each step = ONE target forward pass. n-gram drafts are copied from the prompt.
STEPS = [
    {"draft": ["def", " calc", "(items", ","], "accept": 1, "extra": " compute_invoice_total",
     "hl": [(0, "def calc(items,")]},
    {"draft": [], "accept": 0, "extra": "(items", "hl": []},     # no n-gram match for the new name
    {"draft": [",", " tax_rate", ",", " discount"], "accept": 4, "extra": "=None",
     "hl": [(0, ", tax_rate, discount")]},
    {"draft": ["):", " subtotal", " =", " 0"], "accept": 4, "extra": " for",
     "hl": [(0, "):"), (1, "subtotal = 0")]},
]
PROMPT_CODE = ["def calc(items, tax_rate, discount=None):", "    subtotal = 0", "    for item in items:"]
X0, X1 = 64, 1216


def flow(c, items, y0, row_h=50, size=17, h=40):
    """Lay out chips left-to-right, wrapping. items: (text, state, alpha_ok)."""
    x, y = X0, y0
    for s, state in items:
        w = c.text_w(s.strip(), size, "mono") + 24 + (22 if state in ("accepted", "rejected") else 0)
        if x + w > X1:
            x, y = X0, y + row_h
        token(c, x, y, s, state, size=size, h=h)
        x += w + 7
    return x, y


def scene_race(c: Canvas, t: float, dur: float):
    c.text(48, 30, "Same output. Fewer target passes.", 28, "bold", INK)
    c.text(48, 72, "Illustration of the mechanism · n-gram speculation, γ = 4 · one tick = one target forward pass",
           14, "regular", MUTED)

    # prompt panel, with the span the draft is copying highlighted
    step_now = int((t - START) // P) if t >= START else -1
    c.rect(48, 102, 1184, 92, r=12, fill=PLANE)
    c.text(68, 116, "PROMPT (excerpt)", 11.5, "semibold", MUTED)
    c.text(68, 138, "Rename calc → compute_invoice_total.", 14, "regular", INK_2)
    c.text(68, 160, "Return the full updated code.", 14, "regular", INK_2)
    cx = 470
    hls = STEPS[step_now]["hl"] if 0 <= step_now < len(STEPS) else []
    for i, line in enumerate(PROMPT_CODE):
        ly = 116 + i * 23
        for li, sub in hls:
            if li == i and sub in line:
                k = line.index(sub)
                hx = cx + c.text_w(line[:k], 14.5, "mono")
                c.rect(hx - 3, ly - 2, c.text_w(sub, 14.5, "mono") + 6, 22, r=5, fill=BLUE_TINT)
        c.text(cx, ly, line, 14.5, "mono", INK)
    if hls:
        c.text(1212, 116, "n-gram copies from here", 12, "semibold", BLUE, anchor="ra")

    # lane A: plain decoding
    done_a = 0 if t < START else min(len(OUTPUT), int((t - START) / P + 0.4))  # a pass lands at 0.6P, same as lane B
    c.text(48, 214, "Plain decoding", 17, "semibold", INK)
    c.text(1232, 214, f"target passes: {done_a}", 17, "semibold", INK, anchor="ra")
    items = [(s, "target") for s in OUTPUT[:done_a]]
    x, y = flow(c, items, 246)
    if START <= t and done_a < len(OUTPUT):
        nxt = OUTPUT[done_a]
        w = c.text_w(nxt.strip(), 17, "mono") + 24
        if x + w > X1:
            x, y = X0, y + 50
        pulse = 0.5 + 0.5 * math.sin((t - START) / P * 2 * math.pi)
        c.dashed_rect(x, y, w, 40, color=GRID if pulse < 0.5 else AXIS, r=7)
        c.text(x + w / 2, y + 20, "…", 16, "semibold", MUTED, anchor="mm")

    c.line(48, 352, 1232, 352, GRID, 1)

    # lane B: speculative decoding
    c.text(48, 370, "Speculative decoding", 17, "semibold", INK)
    c.text(230, 372, "n-gram · γ = 4", 14, "regular", MUTED)
    committed, passes, live, note = [], 0, [], ""
    for k, st in enumerate(STEPS):
        t0 = START + k * P
        if t < t0:
            break
        ph = (t - t0) / P
        accepted = st["draft"][:st["accept"]]
        if ph >= 1:  # step finished: commit accepted drafts (green) + target token (dark)
            committed += [(s, "accepted_done") for s in accepted] + [(st["extra"], "target")]
            passes += 1
            continue
        if not st["draft"]:
            note = "no match in the prompt → plain target step"
            if ph >= 0.6:
                live = [(st["extra"], "target")]
                passes += 1
            continue
        if ph < 0.6:
            n = min(len(st["draft"]), int(ph / 0.06) + 1)
            live = [(s, "draft") for s in st["draft"][:n]]
            note = "draft: copy from the prompt" if ph < 0.25 else "target verifies all 4 in ONE pass…"
        else:
            live = [(s, "accepted") for s in accepted]
            live += [(s, "rejected") for s in st["draft"][st["accept"]:st["accept"] + 1]]
            live += [(st["extra"], "target")]
            passes += 1
            note = (f"{len(accepted)} accepted + 1 from the target = {len(accepted) + 1} tokens"
                    if len(accepted) < len(st["draft"]) else "all 4 accepted + 1 bonus token = 5 tokens")
    total_b = sum(1 for _ in committed)
    if total_b >= len(OUTPUT):
        note = "done: 13 tokens. Plain decoding is still going…" if done_a < len(OUTPUT) else ""
    c.text(1232, 370, f"target passes: {passes}", 17, "semibold", INK, anchor="ra")
    if note:
        c.text(48, 400, note, 13.5, "medium", BLUE if "copy" in note else INK_2)
    # accepted_done = a draft token that was accepted earlier: green chip, no icon
    x, y = X0, 428
    for s, st_ in committed + live:
        state = {"accepted_done": "accepted"}.get(st_, st_)
        w = c.text_w(s.strip(), 17, "mono") + 24 + (22 if st_ in ("accepted", "rejected") else 0)
        if x + w > X1:
            x, y = X0, y + 50
        token(c, x, y, s, state, size=17, h=40, icon=st_ != "accepted_done")
        x += w + 7

    # summary
    end_a = START + len(OUTPUT) * P
    s = seg(t, end_a + 0.2, end_a + 0.8)
    if s:
        y = 556
        for i, (big, small, tint) in enumerate([(f"{len(OUTPUT)} passes", "plain decoding", PLANE),
                                                (f"{len(STEPS)} passes", "speculative decoding", BLUE_TINT)]):
            x = 48 + i * 300
            c.rect(x, y, 280, 104, r=14, fill=tint)
            c.text(x + 22, y + 22, big, 32, "bold", INK)
            c.text(x + 22, y + 68, small, 15, "regular", INK_2)
        c.rect(668, y, 564, 104, r=14, fill=GOOD_TINT)
        c.check(700, y + 38, size=10, width=3.2)
        c.text(724, y + 38, "identical output text", 22, "semibold", INK, anchor="lm")
        c.text(700, y + 74, "greedy decoding is lossless; bench.py checks it on every run", 14, "regular", INK_2,
               anchor="lm")


# --------------------------------------------------------------------------- scene 3: knobs
CODE = [
    "from vllm import LLM",
    "",
    "# n-gram (prompt lookup): no extra model",
    'LLM(model="Qwen/Qwen2.5-3B-Instruct",',
    "    speculative_config={",
    '        "method": "ngram",',
    '        "num_speculative_tokens": 5,',
    '        "prompt_lookup_min": 2,',
    '        "prompt_lookup_max": 4})',
    "",
    "# draft model: a small same-family model proposes",
    'LLM(model="Qwen/Qwen2.5-3B-Instruct",',
    "    speculative_config={",
    '        "method": "draft_model",',
    '        "model": "Qwen/Qwen2.5-0.5B-Instruct",',
    '        "num_speculative_tokens": 5})',
]


def code_line(c, x, y, line, size=15):
    """Minimal syntax colors: comments dim, strings green, numbers orange."""
    if line.lstrip().startswith("#"):
        c.text(x, y, line, size, "mono", TERM_DIM)
        return
    parts, buf, in_str = [], "", False
    for ch in line:
        if ch == '"':
            if in_str:
                parts.append((buf + ch, TERM_GREEN))
                buf, in_str = "", False
            else:
                parts.append((buf, TERM_FG))
                buf, in_str = ch, True
        else:
            buf += ch
    parts.append((buf, TERM_GREEN if in_str else TERM_FG))
    for text, color in parts:
        if color == TERM_FG and any(ch.isdigit() for ch in text) and "Qwen" not in text:
            for i, ch in enumerate(text):
                c.text(x + c.text_w(text[:i], size, "mono"), y, ch, size, "mono",
                       TERM_ORANGE if ch.isdigit() else TERM_FG)
        else:
            c.text(x, y, text, size, "mono", color)
        x += c.text_w(text, size, "mono")


def scene_knobs(c: Canvas, t: float, dur: float):
    c.text(48, 30, "The knobs you actually set", 28, "bold", INK)
    c.text(48, 72, "speculative_config is a plain dict on vllm.LLM (or --speculative-config on vllm serve)",
           14, "regular", MUTED)
    c.rect(48, 110, 660, 560, r=14, fill=TERM_BG)
    total = sum(len(l) for l in CODE)
    shown = int(total * seg(t, 0.2, 4.0)) if t < 4.0 else total
    y = 138
    for line in CODE:
        if shown <= 0:
            break
        part = line[:shown]
        code_line(c, 76, y, part)
        shown -= len(line)
        y += 30
    knobs = [("1", "num_speculative_tokens", "γ: how many tokens to guess per step"),
             ("2", "prompt_lookup_min / max", "n-gram window matched against the prompt"),
             ("3", "method + model", "draft_model needs a same-tokenizer small model"),
             ("4", "batch size (your traffic)", "bench.py measures bs=1 AND batched: the win shrinks as batches grow")]
    for i, (n, name, desc) in enumerate(knobs):
        if t < 1.0 + i * 0.9:
            break
        y = 130 + i * 128
        c.circle(756, y + 16, 15, fill=INK)
        c.text(756, y + 16, n, 14, "semibold", "#ffffff", anchor="mm")
        c.text(786, y + 4, name, 17, "mono_bold", INK)
        words, line, ly = desc.split(), "", y + 36
        for w_ in words:
            if c.text_w(line + " " + w_, 15) > 420:
                c.text(786, ly, line.strip(), 15, "regular", INK_2)
                line, ly = "", ly + 22
            line += " " + w_
        c.text(786, ly, line.strip(), 15, "regular", INK_2)


# --------------------------------------------------------------------------- scene 4: terminal
CMDS = [["pytest", "-q"],
        ["python", "-m", "specdec.theory", "--alpha", "0.8", "--c", "0.15"],
        ["python", "bench.py", "--mode", "ngram", "--gamma", "5", "--dry-run"]]


def capture_terminal() -> dict:
    out = {}
    for cmd in CMDS:
        real = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] if cmd[0] == "pytest" else \
            [sys.executable] + cmd[1:]
        r = subprocess.run(real, cwd=ROOT, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).rstrip("\n").splitlines()
        out[" ".join(cmd)] = [ln for ln in lines if not ln.startswith("===") or "passed" in ln]
    return out


def term_color(line: str) -> tuple[str, str]:
    if "passed" in line:
        return TERM_GREEN, "mono_bold"
    if line.startswith("best gamma"):
        return TERM_BLUE, "mono_bold"
    if line.startswith("SKIPPED"):
        return TERM_DIM, "mono"
    return TERM_FG, "mono"


def scene_terminal(c: Canvas, t: float, dur: float):
    c.rect(32, 24, 1216, 672, r=14, fill=TERM_BG)
    for i in range(3):
        c.circle(60 + i * 20, 48, 6, fill="#3a3936")
    c.text(640, 48, "specdec-vllm — CPU only, no GPU needed", 13, "medium", TERM_DIM, anchor="mm")
    cmds = list(TERM.keys())
    # timeline: (cmd index, start time); third command clears the screen
    starts = [0.4, 3.6, 8.6]
    screens = [[0, 1], [2]]
    screen = screens[1] if t >= starts[2] else screens[0]
    y, size, lh = 84, 15, 22
    for idx in screen:
        st = starts[idx]
        if t < st:
            break
        cmd = cmds[idx]
        typed = cmd[:int((t - st) / 0.035)]
        c.text(56, y, "$", size, "mono_bold", TERM_GREEN)
        c.text(74, y, typed, size, "mono", TERM_FG)
        y += lh
        out_t = st + len(cmd) * 0.035 + 0.25
        if t >= out_t:
            n = int((t - out_t) / 0.06) + 1
            for line in TERM[cmd][:n]:
                col, wt = term_color(line)
                c.text(56, y, line, size, wt, col)
                y += lh
        y += lh * 0.6
    # caret
    if int(t * 2) % 2 == 0:
        c.rect(56, y, 9, 17, r=1, fill=TERM_FG)


# --------------------------------------------------------------------------- scene 5: repo + theory
_IMGS: dict = {}


def _img(name):
    if name not in _IMGS:
        _IMGS[name] = Image.open(OUT / name)
    return _IMGS[name]


def scene_overview(c: Canvas, t: float, dur: float):
    half = dur / 2
    if t < half:
        paste(c, _img("pipeline.png"), 48, 40, 1184)
    else:
        c.text(48, 40, "What theory predicts (before you measure)", 28, "bold", INK)
        c.text(48, 82, "bench.py + report.py then put your real GPU numbers next to these curves",
               15, "regular", INK_2)
        paste(c, _img("theory-speedup.png"), 48, 130, 1184)


# --------------------------------------------------------------------------- scene 6: run it
def scene_run(c: Canvas, t: float, dur: float):
    c.rect(0, 0, W / 2, 6, r=0, fill=BLUE)
    c.rect(W / 2, 0, W / 2, 6, r=0, fill=ORANGE)
    c.text(96, 150, "Run it", 48, "bold", INK)
    c.rect(96, 240, 1088, 230, r=16, fill=TERM_BG)
    lines = [("$", "pip install -r requirements.txt", "# NVIDIA GPU + vLLM"),
             ("$", "python bench.py --mode all", "# 7 configs -> results/*.json"),
             ("$", "python report.py results --assets docs/assets", "# report.md + charts"),
             ("$", "pytest", "# correctness + sanity checks")]
    for i, (p, cmd, cm) in enumerate(lines):
        if t < 0.3 + i * 0.35:
            break
        y = 270 + i * 46
        c.text(130, y, p, 18, "mono_bold", TERM_GREEN)
        c.text(156, y, cmd, 18, "mono", TERM_FG)
        c.text(156 + c.text_w(cmd + "  ", 18, "mono"), y, cm, 18, "mono", TERM_DIM)
    if t > 1.8:
        c.text(96, 520, REPO, 22, "semibold", INK)
        c.text(96, 560, "Tests run on CPU in CI · the benchmark needs an NVIDIA GPU (e.g. a Colab L4)", 16,
               "regular", INK_2)


SCENES = [(4.2, scene_title), (START + len(OUTPUT) * P + 3.6, scene_race), (7.5, scene_knobs),
          (12.5, scene_terminal), (8.0, scene_overview), (5.0, scene_run)]
FADE = 0.35


def total_duration():
    return sum(d for d, _ in SCENES)


def render(t: float) -> bytes:
    acc = 0.0
    for dur, fn in SCENES:
        if t < acc + dur or fn is SCENES[-1][1]:
            lt = t - acc
            c = Canvas(W, H, scale=2)
            fn(c, lt, dur)
            img = c.img.resize(OUT_SIZE, Image.LANCZOS)
            a = min(1.0, lt / FADE, max(0.0, (dur - lt) / FADE))
            if a < 1:
                img = Image.blend(Image.new("RGB", OUT_SIZE, SURFACE), img, ease(a))
            return img.tobytes()
        acc += dur
    raise ValueError(t)


def _init(term):
    TERM.update(term)


def ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    OUT.mkdir(parents=True, exist_ok=True)
    term = capture_terminal()
    TERM.update(term)
    n = int(total_duration() * FPS)
    mp4 = OUT / "demo.mp4"
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{OUT_SIZE[0]}x{OUT_SIZE[1]}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    with Pool(initializer=_init, initargs=(term,)) as pool:
        for i, frame in enumerate(pool.imap(render, [i / FPS for i in range(n)], chunksize=8)):
            proc.stdin.write(frame)
            if i % 150 == 0:
                print(f"frame {i}/{n}", flush=True)
    proc.stdin.close()
    proc.wait()
    print("wrote", mp4)

    # poster: the race scene's final summary
    race_end = SCENES[0][0] + SCENES[1][0] - 0.6
    Image.frombytes("RGB", OUT_SIZE, render(race_end)).save(OUT / "demo-poster.png", optimize=True)
    print("wrote", OUT / "demo-poster.png")

    # GIF of the race only (plays inline in a GitHub README)
    gif = OUT / "demo-race.gif"
    ss, dur = SCENES[0][0], SCENES[1][0]
    vf = "fps=12,scale=960:-1:flags=lanczos"
    pal = OUT / "_palette.png"
    ffmpeg(["-ss", str(ss), "-t", str(dur), "-i", str(mp4), "-vf", f"{vf},palettegen=max_colors=96:stats_mode=diff",
            str(pal)])
    ffmpeg(["-ss", str(ss), "-t", str(dur), "-i", str(mp4), "-i", str(pal), "-lavfi",
            f"{vf}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle", str(gif)])
    pal.unlink()
    print("wrote", gif)


if __name__ == "__main__":
    main()
