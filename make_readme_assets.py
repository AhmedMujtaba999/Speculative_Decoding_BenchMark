"""Regenerate the README images in docs/assets/ (no GPU needed).

    python scripts/make_readme_assets.py

banner.png            hero image
how-it-works.png      draft -> verify -> accept/reject, vs plain decoding
pipeline.png          how the repo's pieces fit together
theory-speedup.png    expected speedup vs acceptance (closed-form, Leviathan et al. 2023)
optimal-gamma.png     best gamma for each (alpha, draft cost) pair

Benchmark result charts are NOT made here - they come from your own GPU run:
    python bench.py --mode all && python report.py results --assets docs/assets
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from _draw import (AXIS, BLUE, BLUE_TINT, GRID, INK, INK_2, MUTED, NEUTRAL_TINT, ORANGE,  # noqa: E402
                   ORANGE_TINT, PLANE, SURFACE, Canvas, token)
from specdec import style, theory  # noqa: E402

OUT = ROOT / "docs" / "assets"
RENDER, EXPORT = 4, 2  # draw at 4x, save at 2x (sharp on retina, light on bandwidth)

# The running example: the model is rewriting code from the prompt, so the next tokens
# can be copied from the prompt (n-gram) or guessed by a small model (draft).
PREV = "tax"
DRAFT = [" =", " subtotal", " *", " discount"]
FIX = " tax_rate"


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, optimize=True)
    print("wrote", OUT / name)


def chips(c, x, y, items, size=17, h=38, gap=8):
    for s, state in items:
        x += token(c, x, y, s, state, size=size, h=h) + gap
    return x


# --------------------------------------------------------------------------- banner
def banner():
    c = Canvas(1280, 372, scale=RENDER)
    c.rect(0, 0, 1280, 6, r=0, fill=BLUE)
    c.rect(640, 0, 640, 6, r=0, fill=ORANGE)
    c.text(64, 58, "LLM INFERENCE  ·  vLLM  ·  BENCHMARK", 13, "semibold", MUTED)
    c.text(64, 84, "Speculative decoding with vLLM", 50, "bold", INK)
    c.text(64, 156, "Baseline vs n-gram vs draft-model speculation. Measures speedup,", 19, "regular", INK_2)
    c.text(64, 184, "acceptance rate and greedy-output correctness on real prompts.", 19, "regular", INK_2)

    # formula card
    x0, y0, w, h = 860, 60, 356, 158
    c.rect(x0, y0, w, h, r=14, fill=PLANE)
    c.text(x0 + 22, y0 + 22, "EXPECTED SPEEDUP", 12, "semibold", MUTED)
    c.text(x0 + 22, y0 + 52, "(1 − α", 22, "semibold", INK)
    sx = x0 + 22 + c.text_w("(1 − α", 22, "semibold") + 1
    c.text(sx, y0 + 46, "γ+1", 13, "semibold", INK)
    c.text(sx + c.text_w("γ+1", 13, "semibold") + 2, y0 + 52, ")", 22, "semibold", INK)
    c.line(x0 + 22, y0 + 88, x0 + w - 22, y0 + 88, INK_2, 1.5)
    c.text(x0 + 22, y0 + 96, "(1 − α) · (γ·c + 1)", 22, "semibold", INK)
    c.text(x0 + 22, y0 + 132, "α acceptance  ·  γ draft length  ·  c draft cost", 12.5, "regular", INK_2)

    # token strip: one target pass yields four tokens
    y = 262
    c.text(64, y - 30, "ONE TARGET FORWARD PASS", 12, "semibold", MUTED)
    x = chips(c, 64, y, [(PREV, "prompt")] + [(t, "accepted") for t in DRAFT[:3]]
              + [(DRAFT[3], "rejected"), (FIX, "target")], size=18, h=42, gap=9)
    c.text(x + 14, y + 21, "→  4 tokens instead of 1", 19, "semibold", INK, anchor="lm")
    c.text(64, y + 66, "3 drafted tokens accepted  +  1 token from the target at the first mismatch. "
           "Greedy output is identical to plain decoding.", 14, "regular", INK_2)
    save(c.finish(EXPORT), "banner.png")


# --------------------------------------------------------------------------- how it works
def how_it_works():
    c = Canvas(1280, 750, scale=RENDER)
    c.text(48, 40, "How speculative decoding works", 30, "bold", INK)
    c.text(48, 84, "Decode is memory-bound: one target pass costs about the same for 1 token or 5. "
           "So guess cheaply, then verify in bulk.", 16, "regular", INK_2)

    # A. plain decoding
    y = 140
    c.text(48, y, "Plain decoding", 18, "semibold", INK)
    c.text(48, y + 26, "1 target forward pass → 1 token", 14, "regular", INK_2)
    x = 520
    x += token(c, x, y - 2, PREV, "prompt", size=17, h=38) + 8
    for i, t in enumerate([" =", " subtotal", " *", FIX]):
        w = token(c, x, y - 2, t, "target", size=17, h=38)
        c.text(x + w / 2, y + 46, f"pass {i + 1}", 12, "medium", MUTED, anchor="ma")
        x += w + 10
    c.text(x + 12, y + 17, "4 passes → 4 tokens", 15, "semibold", INK_2, anchor="lm")

    c.line(48, 232, 1232, 232, GRID, 1)

    # B. speculative decoding
    y = 258
    c.text(48, y, "Speculative decoding", 18, "semibold", INK)
    c.text(48, y + 26, "γ = 4  (num_speculative_tokens)", 14, "regular", INK_2)

    rows = [
        ("1", "Draft proposes γ tokens", "n-gram: copy from the prompt  ·  or a small draft model",
         [(PREV, "prompt")] + [(t, "draft") for t in DRAFT]),
        ("2", "Target verifies all of them in ONE pass", "keep drafts up to the first mismatch",
         [(PREV, "prompt")] + [(t, "accepted") for t in DRAFT[:3]] + [(DRAFT[3], "rejected")]),
        ("3", "Target adds its own token", "the correction (or a bonus token if all γ were accepted)",
         [(PREV, "prompt")] + [(t, "accepted") for t in DRAFT[:3]] + [(FIX, "target")]),
    ]
    y = 318
    for num, title, sub, items in rows:
        c.circle(62, y + 19, 14, fill=INK)
        c.text(62, y + 19, num, 14, "semibold", "#ffffff", anchor="mm")
        c.text(88, y + 6, title, 15, "semibold", INK)
        c.text(88, y + 28, sub, 12.5, "regular", INK_2)
        chips(c, 520, y, items, size=17, h=38)
        y += 70
    c.rect(520 - 6, y - 4, 570, 44, r=10, fill=PLANE)
    c.text(534, y + 18, "1 target pass → 4 new tokens", 17, "semibold", INK, anchor="lm")
    c.text(810, y + 18, "same text as plain greedy decoding", 14, "regular", INK_2, anchor="lm")

    # knobs
    y = 590
    cards = [("α", "acceptance", "How often the target agrees with the next drafted token. "
              "Set by the workload, not by you.", NEUTRAL_TINT),
             ("γ", "num_speculative_tokens", "How far to guess ahead. Bigger γ = more upside, "
              "more wasted work on misses.", BLUE_TINT),
             ("c", "draft cost ÷ target cost", "n-gram lookup costs almost nothing (c ≈ 0); "
              "a small same-family model costs roughly 0.1–0.3.", ORANGE_TINT)]
    w = (1232 - 48 - 2 * 20) / 3
    for i, (sym, name, desc, tint) in enumerate(cards):
        x = 48 + i * (w + 20)
        c.rect(x, y, w, 130, r=14, fill=tint)
        c.text(x + 22, y + 18, sym, 30, "bold", INK)
        c.text(x + 62, y + 28, name, 15, "semibold", INK)
        words, line, ly = desc.split(), "", y + 66
        for word in words:
            if c.text_w(line + " " + word, 13.5) > w - 44:
                c.text(x + 22, ly, line.strip(), 13.5, "regular", INK_2)
                line, ly = "", ly + 21
            line += " " + word
        c.text(x + 22, ly, line.strip(), 13.5, "regular", INK_2)
    save(c.finish(EXPORT), "how-it-works.png")


# --------------------------------------------------------------------------- pipeline
def pipeline():
    c = Canvas(1280, 584, scale=RENDER)
    c.text(48, 40, "What the repo does", 30, "bold", INK)
    c.text(48, 84, "One command per stage. Every stage except the vLLM run is testable on a laptop.",
           16, "regular", INK_2)

    boxes = [
        ("specdec/prompts.py", ["16 prompts", "4 categories:", "code_edit · extract", "rag_qa · open_ended"], PLANE),
        ("bench.py --mode all", ["1 subprocess / config", "baseline", "ngram γ = 3, 5, 8", "draft γ = 3, 5, 8"], PLANE),
        ("vllm.LLM(...)", ["speculative_config", "batch → throughput", "bs=1 → latency", "  + per category"], BLUE_TINT),
        ("results/*.json", ["tok/s · acceptance", "per-position rates", "outputs · KV blocks", "GPU · vLLM version"], PLANE),
        ("report.py", ["report.md", "dashboard + 4 charts", "--assets docs/assets", ""], PLANE),
    ]
    gap, top, bh = 34, 150, 176
    bw = (1232 - 48 - gap * 4) / 5
    for i, (head, lines, tint) in enumerate(boxes):
        x = 48 + i * (bw + gap)
        c.rect(x, top, bw, bh, r=14, fill=tint)
        c.text(x + 18, top + 20, head, 14.5, "mono_bold", INK)
        c.line(x + 18, top + 50, x + bw - 18, top + 50, AXIS, 1)
        for j, ln in enumerate(lines):
            c.text(x + 18, top + 64 + j * 24, ln, 14, "regular", INK_2)
        if i < 4:
            c.arrow(x + bw + 6, top + bh / 2, x + bw + gap - 6, top + bh / 2, INK_2, 2, 9)

    # metrics callout under the vLLM box
    x3 = 48 + 2 * (bw + gap)
    c.line(x3 + bw / 2, top + bh, x3 + bw / 2, top + bh + 26, AXIS, 1.5)
    l1, l2 = "llm.get_metrics()  →  vllm:spec_decode_num_drafts,", "num_draft_tokens, num_accepted_tokens[_per_pos]"
    cw = max(c.text_w(l1, 12.5, "mono"), c.text_w(l2, 12.5, "mono")) + 36
    cx0 = x3 + bw / 2 - cw / 2
    c.rect(cx0, top + bh + 26, cw, 62, r=10, fill=SURFACE, outline=GRID, width=1.5)
    c.text(cx0 + 18, top + bh + 40, l1, 12.5, "mono", INK_2)
    c.text(cx0 + 18, top + bh + 62, l2, 12.5, "mono", INK_2)

    # tests strip
    y = 452
    c.rect(48, y, 1184, 112, r=14, fill=PLANE)
    c.text(70, y + 20, "pytest", 15, "mono_bold", INK)
    c.text(140, y + 21, "CPU (runs in CI)", 13, "semibold", MUTED)
    x = 70
    for name in ("test_theory", "test_metrics", "test_prompts", "test_bench_cli", "test_report", "test_pipeline"):
        x += token(c, x, y + 52, name, "prompt", size=13.5, h=34, pad=10) + 8
    c.text(x + 18, y + 20, "GPU results", 13, "semibold", MUTED)
    token(c, x + 18, y + 52, "test_results", "target", size=13.5, h=34, pad=10)
    c.text(x + 162, y + 69, "greedy match · sane counters", 13, "regular", INK_2, anchor="lm")
    save(c.finish(EXPORT), "pipeline.png")


# --------------------------------------------------------------------------- theory charts
BLUES = ["#86b6ef", "#2a78d6", "#184f95"]      # ordinal: gamma 3 -> 8 (light -> dark)
ORANGES = ["#f4a582", "#eb6834", "#a8401a"]


def theory_speedup():
    style.apply()
    alphas = [a / 200 for a in range(0, 200)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (title, cost, ramp) in zip(axes, [("n-gram lookup  (c ≈ 0)", 0.0, BLUES),
                                              ("small draft model  (c = 0.25)", 0.25, ORANGES)]):
        ax.axhline(1.0, color=AXIS, lw=1.2)
        ax.text(0.99, 0.93, "break-even (1x)", color=style.MUTED, fontsize=9, ha="right", va="top",
                transform=ax.get_yaxis_transform())
        for g, col in zip((3, 5, 8), ramp):
            ys = [theory.expected_speedup(a, g, cost) for a in alphas]
            ax.plot(alphas, ys, color=col, lw=2)
            ax.annotate(f"γ={g}", (alphas[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9.5, color=style.INK_2)
        ax.set_title(title)
        ax.set_xlabel("α  (per-token acceptance probability)")
        ax.set_xlim(0, 1.08)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel("expected speedup (x)")
    axes[0].set_ylim(0, 9.8)
    axes[1].set_ylim(0, 3.4)
    fig.suptitle(r"Expected speedup $=\dfrac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$"
                 "     idealized: memory-bound decode, independent per-token acceptance",
                 x=0.01, ha="left", fontsize=10.5, color=style.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "theory-speedup.png", dpi=170)
    plt.close(fig)
    print("wrote", OUT / "theory-speedup.png")


def optimal_gamma():
    style.apply()
    alphas = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    costs = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    best = [[theory.optimal_gamma(a, cst, max_gamma=12) for a in alphas] for cst in costs]
    data = [[s for _, s in row] for row in best]
    cmap = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower", vmin=1, vmax=max(map(max, data)))
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(range(len(alphas)), [f"{a:.2f}".rstrip("0") for a in alphas])
    ax.set_yticks(range(len(costs)), [f"{c:g}" for c in costs])
    ax.set_xlabel("α  (per-token acceptance probability)")
    ax.set_ylabel("c  (draft cost ÷ target cost)")
    vmax = max(map(max, data))
    for i, row in enumerate(best):
        for j, (g, s) in enumerate(row):
            dark = s > 1 + 0.45 * (vmax - 1)
            ax.text(j, i + 0.13, f"γ={g}", ha="center", va="center", fontsize=10, fontweight="semibold",
                    color="#ffffff" if dark else INK)
            ax.text(j, i - 0.2, f"{s:.1f}x", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if dark else INK_2)
    ax.set_title("Best γ (max 12) and its expected speedup")
    fig.text(0.01, 0.01, "Idealized: ignores verification cost at large γ and batch sizes. Real n-gram α is capped "
             "by how much of the output exists in the prompt.", fontsize=8.5, color=style.MUTED)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("expected speedup (x)", color=style.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "optimal-gamma.png", dpi=170)
    plt.close(fig)
    print("wrote", OUT / "optimal-gamma.png")


if __name__ == "__main__":
    banner()
    how_it_works()
    pipeline()
    theory_speedup()
    optimal_gamma()
