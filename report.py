"""Turn results/*.json into a markdown report + charts.

    python report.py                         # reads results/, writes results/report.md + PNGs
    python report.py results --assets docs/assets   # also copy charts for the README

Outputs
    report.md                  summary table, per-category tables, highlights, failed configs
    report.png                 2x2 dashboard (speedup, tokens/step vs gamma, category heatmap, per-position)
    speedup.png, category_heatmap.png, acceptance_length.png, position_acceptance.png  (single panels)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from specdec import style, theory
from specdec.prompts import CATEGORIES

MODE_ORDER = {"baseline": 0, "ngram": 1, "draft": 2, "suffix": 3}


# --------------------------------------------------------------------------- loading
def _normalize(r: dict) -> dict:
    """Upgrade schema-1 rows (flat fields only) to the schema-2 shape."""
    if "batch" not in r and "latency" not in r and "tokens_per_s" in r:
        r["batch"] = {"tokens_per_s": r["tokens_per_s"], "elapsed_s": r.get("elapsed_s"),
                      "output_tokens": r.get("output_tokens"),
                      "spec": {"acceptance_rate": r.get("acceptance_rate"),
                               "mean_acceptance_length": r.get("tokens_per_step"),
                               "per_position_rate": []}}
        r["batch_texts"] = r.get("texts", [])
    return r


def _texts(r: dict) -> tuple[str, list[str]]:
    if r.get("latency_texts"):
        return "latency", r["latency_texts"]
    return "batch", r.get("batch_texts") or r.get("texts") or []


def _ratio(a, b):
    return a / b if a is not None and b else None


def load(results_dir="results") -> tuple[list[dict], list[dict]]:
    """Return (ok_rows, failed_rows) with derived fields filled in."""
    raw = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*_g*.json"))]
    failed = [r for r in raw if r.get("status") == "failed"]
    rows = [_normalize(r) for r in raw if r.get("status", "ok") == "ok"]
    rows.sort(key=lambda r: (MODE_ORDER.get(r["mode"], 9), r["gamma"]))
    base = next((r for r in rows if r["mode"] == "baseline"), None)

    for r in rows:
        r["label"] = "baseline" if r["mode"] == "baseline" else f'{r["mode"]} γ={r["gamma"]}'
        b, l = r.get("batch"), r.get("latency")
        bb, bl = (base or {}).get("batch"), (base or {}).get("latency")
        r["batch_tps"] = b["tokens_per_s"] if b else None
        r["latency_tps"] = l["tokens_per_s"] if l else None
        r["batch_speedup"] = _ratio(r["batch_tps"], bb["tokens_per_s"] if bb else None)
        r["latency_speedup"] = _ratio(r["latency_tps"], bl["tokens_per_s"] if bl else None)

        spec = (l or b or {}).get("spec", {})
        r["acceptance_rate"] = spec.get("acceptance_rate")
        r["tokens_per_step"] = spec.get("mean_acceptance_length")
        r["per_position_rate"] = spec.get("per_position_rate") or []
        r["alpha_est"] = None
        if r["tokens_per_step"] and r["gamma"]:
            tps = min(max(r["tokens_per_step"], 1.0), r["gamma"] + 1)
            r["alpha_est"] = theory.alpha_from_tokens_per_step(tps, r["gamma"])

        r["match"] = None
        if base:
            kind, mine = _texts(r)
            theirs = base.get(f"{kind}_texts") or _texts(base)[1]
            if mine and theirs:
                r["match"] = sum(a == b for a, b in zip(mine, theirs)) / len(theirs)

        r["category_speedup"], r["category_accept"] = {}, {}
        if l and bl and "by_category" in l and "by_category" in bl:
            for cat, stats in l["by_category"].items():
                ref = bl["by_category"].get(cat)
                r["category_speedup"][cat] = _ratio(stats["tokens_per_s"], ref["tokens_per_s"] if ref else None)
                r["category_accept"][cat] = stats.get("acceptance_rate")
    return rows, failed


# --------------------------------------------------------------------------- markdown
def fmt(x, spec, suffix=""):
    return "-" if x is None else format(x, spec) + suffix


def _categories(rows) -> list[str]:
    seen = {c for r in rows for c in r["category_speedup"]}
    return [c for c in CATEGORIES if c in seen] + sorted(seen - set(CATEGORIES))


def summary_table(rows) -> list[str]:
    lines = ["| config | batch tok/s | batch speedup | bs=1 tok/s | bs=1 speedup | accept rate "
             "| tokens/step | est. α | KV blocks | greedy match |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f'| {r["label"]} | {fmt(r["batch_tps"], ".1f")} | {fmt(r["batch_speedup"], ".2f", "x")} '
            f'| {fmt(r["latency_tps"], ".1f")} | {fmt(r["latency_speedup"], ".2f", "x")} '
            f'| {fmt(r["acceptance_rate"], ".2f")} | {fmt(r["tokens_per_step"], ".2f")} '
            f'| {fmt(r["alpha_est"], ".2f")} | {fmt(r.get("kv_cache_blocks"), "d")} '
            f'| {fmt(r["match"], ".0%")} |')
    return lines


def category_tables(rows) -> list[str]:
    cats = _categories(rows)
    spec_rows = [r for r in rows if r["mode"] != "baseline" and r["category_speedup"]]
    if not cats or not spec_rows:
        return []
    head = "| config | " + " | ".join(cats) + " |"
    sep = "|---|" + "---:|" * len(cats)
    out = ["## Speedup by prompt category (batch size 1)", "", head, sep]
    for r in spec_rows:
        out.append(f'| {r["label"]} | ' + " | ".join(fmt(r["category_speedup"].get(c), ".2f", "x") for c in cats) + " |")
    out += ["", "## Acceptance rate by prompt category", "", head, sep]
    for r in spec_rows:
        out.append(f'| {r["label"]} | ' + " | ".join(fmt(r["category_accept"].get(c), ".2f") for c in cats) + " |")
    return out + [""]


def highlights(rows) -> list[str]:
    spec = [r for r in rows if r["mode"] != "baseline"]
    out = []
    for key, name in (("latency_speedup", "batch size 1"), ("batch_speedup", "batched")):
        cands = [r for r in spec if r[key] is not None]
        if cands:
            best = max(cands, key=lambda r: r[key])
            out.append(f'- Best config at {name}: **{best["label"]}** ({best[key]:.2f}x vs baseline)')
    for cat in _categories(rows):
        cands = [r for r in spec if r["category_speedup"].get(cat) is not None]
        if cands:
            best = max(cands, key=lambda r: r["category_speedup"][cat])
            out.append(f'- `{cat}`: best is **{best["label"]}** ({best["category_speedup"][cat]:.2f}x)')
    low = [r for r in spec if r["match"] is not None and r["match"] < 0.9]
    if low:
        out.append("- Greedy-match below 90% for: " + ", ".join(r["label"] for r in low)
                   + " (check dtype; small bf16 drift is expected, large drift is a bug)")
    return out


def environment_line(rows) -> str:
    r = rows[0]
    env = r.get("env", {})
    bits = [f'target `{r.get("model", "?")}`']
    drafts = {x.get("draft_model") for x in rows if x.get("draft_model")}
    if drafts:
        bits.append("draft `" + ", ".join(sorted(drafts)) + "`")
    for k, name in (("gpu", "GPU"), ("vllm", "vLLM"), ("torch", "torch")):
        if env.get(k):
            bits.append(f"{name} {env[k]}")
    if r.get("max_tokens"):
        bits.append(f'max_tokens {r["max_tokens"]}')
    if r.get("categories"):
        bits.append(f'{len(r["categories"])} prompts')
    return " · ".join(bits)


# --------------------------------------------------------------------------- charts
def _color(r):
    return style.METHOD_COLORS.get(r["mode"], style.INK_2)


def plot_speedup(ax, rows):
    """Dumbbell: hollow = batched, filled = batch size 1; color = method."""
    spec = [r for r in rows if r["mode"] != "baseline"][::-1]
    ys = range(len(spec))
    for y, r in zip(ys, spec):
        vals = [v for v in (r["batch_speedup"], r["latency_speedup"]) if v is not None]
        if len(vals) == 2:
            ax.plot(vals, [y, y], color=style.GRID, lw=3, zorder=1, solid_capstyle="round")
        if r["batch_speedup"] is not None:
            ax.scatter(r["batch_speedup"], y, s=70, facecolor=style.SURFACE, edgecolor=_color(r), lw=2, zorder=3)
        if r["latency_speedup"] is not None:
            ax.scatter(r["latency_speedup"], y, s=70, color=_color(r), edgecolor=style.SURFACE, lw=2, zorder=4)
            ax.annotate(f'{r["latency_speedup"]:.2f}x', (r["latency_speedup"], y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=style.INK_2)
    ax.axvline(1.0, color=style.AXIS, lw=1, zorder=0)
    ax.set_yticks(list(ys), [r["label"] for r in spec])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("speedup vs baseline (tokens/s ratio)")
    ax.set_title("Speedup: batch size 1 vs batched")
    handles = [Line2D([], [], marker="o", ls="", color=style.METHOD_COLORS[m], label=style.METHOD_NAMES[m], ms=8)
               for m in ("ngram", "draft", "suffix") if any(r["mode"] == m for r in spec)]
    handles += [Line2D([], [], marker="o", ls="", color=style.INK_2, label="batch size 1", ms=8),
                Line2D([], [], marker="o", ls="", mfc=style.SURFACE, mec=style.INK_2, mew=2, label="batched", ms=8)]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    lo = min([1.0] + [v for r in spec for v in (r["batch_speedup"], r["latency_speedup"]) if v])
    hi = max([1.0] + [v for r in spec for v in (r["batch_speedup"], r["latency_speedup"]) if v])
    ax.set_xlim(min(0.8, lo - 0.1), hi + 0.35)


def plot_acceptance_length(ax, rows):
    """Measured tokens per target pass vs gamma, against the gamma+1 ceiling."""
    gammas = sorted({r["gamma"] for r in rows if r["gamma"]})
    if gammas:
        g = [0] + gammas
        ax.plot(g, [x + 1 for x in g], color=style.AXIS, lw=1.5, label="ceiling (γ+1)")
    for mode in ("ngram", "draft", "suffix"):
        pts = [(r["gamma"], r["tokens_per_step"]) for r in rows if r["mode"] == mode and r["tokens_per_step"]]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=style.METHOD_COLORS[mode], marker="o", ms=7, mec=style.SURFACE, mew=2,
                    label=style.METHOD_NAMES[mode])
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, -2), textcoords="offset points",
                        fontsize=9, color=style.INK_2)
    ax.set_xlabel("γ (num_speculative_tokens)")
    ax.set_ylabel("tokens per target forward pass")
    ax.set_title("Mean acceptance length")
    if gammas:
        ax.set_xticks(gammas)
        ax.set_xlim(gammas[0] - 0.5, gammas[-1] + 1)
    ax.set_ylim(bottom=1)
    ax.legend(loc="upper left", fontsize=8.5)


def plot_category_heatmap(ax, rows, fig=None):
    cats = _categories(rows)
    spec = [r for r in rows if r["mode"] != "baseline" and r["category_speedup"]]
    if not cats or not spec:
        ax.text(0.5, 0.5, "no batch-size-1 results\n(run with --workload latency or both)",
                ha="center", va="center", color=style.MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    data = [[r["category_speedup"].get(c) or float("nan") for c in cats] for r in spec]
    flat = [v for row in data for v in row if v == v]
    lo, hi = min(flat + [0.9]), max(flat + [1.1])
    im = ax.imshow(data, cmap=style.SPEEDUP_CMAP, norm=TwoSlopeNorm(vcenter=1.0, vmin=min(lo, 0.99), vmax=max(hi, 1.01)),
                   aspect="auto")
    ax.set_xticks(range(len(cats)), cats)
    ax.set_yticks(range(len(spec)), [r["label"] for r in spec])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            if v == v:
                dark = v > 1 + 0.6 * (hi - 1) or v < 1 - 0.6 * (1 - lo)
                ax.text(j, i, f"{v:.2f}x", ha="center", va="center", fontsize=9,
                        color="#ffffff" if dark else style.INK)
    ax.set_title("Speedup by prompt category (batch size 1)")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, color=style.MUTED)


def plot_position_acceptance(ax, rows):
    """P(draft position i accepted). Dashed = i.i.d. theory alpha^(i+1) at the fitted alpha."""
    shown = False
    for mode in ("ngram", "draft", "suffix"):
        cands = [r for r in rows if r["mode"] == mode and r["per_position_rate"]]
        if not cands:
            continue
        r = max(cands, key=lambda x: x["gamma"])
        pos = list(range(1, len(r["per_position_rate"]) + 1))
        ax.plot(pos, r["per_position_rate"], color=_color(r), marker="o", ms=7, mec=style.SURFACE, mew=2,
                label=f'{style.METHOD_NAMES[mode]} γ={r["gamma"]} (measured)')
        if r["alpha_est"]:
            ax.plot(pos, theory.expected_position_acceptance(r["alpha_est"], len(pos)), color=_color(r),
                    lw=1.2, ls=(0, (4, 3)), alpha=0.8, label=f'theory α={r["alpha_est"]:.2f}')
        shown = True
    if not shown:
        ax.text(0.5, 0.5, "no per-position counters\n(needs a vLLM build that exports them)",
                ha="center", va="center", color=style.MUTED, transform=ax.transAxes)
    ax.set_ylim(0, 1)
    ax.set_xlabel("draft position")
    ax.set_ylabel("acceptance probability")
    ax.set_title("Acceptance by draft position")
    if shown:
        ax.legend(loc="upper right", fontsize=8)


PANELS = {
    "speedup.png": plot_speedup,
    "acceptance_length.png": plot_acceptance_length,
    "category_heatmap.png": plot_category_heatmap,
    "position_acceptance.png": plot_position_acceptance,
}


def make_charts(rows, out: Path) -> list[Path]:
    style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    plot_speedup(axes[0, 0], rows)
    plot_acceptance_length(axes[0, 1], rows)
    plot_category_heatmap(axes[1, 0], rows, fig)
    plot_position_acceptance(axes[1, 1], rows)
    fig.suptitle("Speculative decoding in vLLM: " + environment_line(rows).replace("`", ""),
                 x=0.01, ha="left", fontsize=10, color=style.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    paths = [out / "report.png"]
    fig.savefig(paths[0], dpi=150)
    plt.close(fig)
    for name, fn in PANELS.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        fn(ax, rows, fig) if fn is plot_category_heatmap else fn(ax, rows)
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        paths.append(out / name)
    return paths


# --------------------------------------------------------------------------- main
def make_report(results_dir="results", assets: str | None = None):
    rows, failed = load(results_dir)
    if not rows:
        raise SystemExit("no results found - run bench.py first")
    out = Path(results_dir)
    table = summary_table(rows)
    md = ["# Speculative decoding benchmark report", "", environment_line(rows), "",
          "## Summary", "", *table, "",
          "_tok/s = output tokens / wall time. bs=1 = one request at a time. tokens/step = 1 + accepted/drafts "
          "(vLLM's mean acceptance length). est. α = per-token acceptance implied by tokens/step._", "",
          *category_tables(rows)]
    hl = highlights(rows)
    if hl:
        md += ["## Highlights", "", *hl, ""]
    if failed:
        md += ["## Failed or unsupported configs", ""]
        md += [f'- `{f["mode"]} γ={f["gamma"]}`: {f.get("error", "unknown error")}' for f in failed] + [""]
    md += ["![dashboard](report.png)", ""]
    (out / "report.md").write_text("\n".join(md))

    paths = make_charts(rows, out)
    if assets:
        dst = Path(assets)
        dst.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy(p, dst / f"results_{p.name}")
    print("\n".join(table))
    if failed:
        print(f"\n{len(failed)} config(s) failed - see report.md")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build report.md + charts from bench.py results")
    ap.add_argument("results_dir", nargs="?", default="results")
    ap.add_argument("--assets", help="also copy charts here as results_*.png (e.g. docs/assets)")
    a = ap.parse_args()
    make_report(a.results_dir, a.assets)
