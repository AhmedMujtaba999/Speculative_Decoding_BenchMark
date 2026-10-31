"""Read vLLM's speculative decoding counters and turn them into acceptance stats.

vLLM (V1 engine) exposes cumulative counters via `LLM.get_metrics()` and Prometheus:

    vllm:spec_decode_num_drafts                  steps where a draft was proposed
    vllm:spec_decode_num_draft_tokens            draft tokens proposed
    vllm:spec_decode_num_accepted_tokens         draft tokens the target accepted
    vllm:spec_decode_num_accepted_tokens_per_pos accepted count at draft position 0..gamma-1

Counters are cumulative, so we snapshot before and after a workload and diff them.
Prometheus scrapes add a `_total` suffix; both spellings are accepted here.
Everything is pure Python so it is unit-testable without a GPU.
"""
from __future__ import annotations

_PREFIX = "vllm:spec_decode_"
_SCALARS = {"num_drafts": "drafts", "num_draft_tokens": "draft_tokens",
            "num_accepted_tokens": "accepted_tokens"}
_PER_POS = "num_accepted_tokens_per_pos"


def _short(name: str) -> str | None:
    if not name.startswith(_PREFIX):
        return None
    short = name[len(_PREFIX):]
    return short[:-6] if short.endswith("_total") else short


def snapshot(metrics) -> dict:
    """Collapse an iterable of vLLM metric objects into plain numbers.

    Each object needs `.name` and either `.value` (counter) or `.values` (vector).
    Several engines (data parallel) report the same name; their counts are summed.
    """
    snap: dict = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0, "per_pos": []}
    for m in metrics:
        short = _short(getattr(m, "name", ""))
        if short in _SCALARS:
            snap[_SCALARS[short]] += getattr(m, "value", 0) or 0
        elif short == _PER_POS:
            vals = list(getattr(m, "values", None) or [])
            if not vals and hasattr(m, "value"):  # scalar-per-position style (labelled)
                vals = [m.value]
            pos = snap["per_pos"]
            pos.extend([0] * (len(vals) - len(pos)))
            for i, v in enumerate(vals):
                pos[i] += v
    return snap


_KEYS = ("drafts", "draft_tokens", "accepted_tokens")


def zero() -> dict:
    return {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0, "per_pos": []}


def _combine(a: dict, b: dict, sign: int) -> dict:
    out = {k: a.get(k, 0) + sign * b.get(k, 0) for k in _KEYS}
    pa, pb = list(a.get("per_pos", [])), list(b.get("per_pos", []))
    n = max(len(pa), len(pb))
    pa, pb = pa + [0] * (n - len(pa)), pb + [0] * (n - len(pb))
    out["per_pos"] = [x + sign * y for x, y in zip(pa, pb)]
    return out


def diff(before: dict, after: dict) -> dict:
    """after - before, element-wise (per_pos padded to the longer list)."""
    return _combine(after, before, -1)


def add(a: dict, b: dict) -> dict:
    """a + b, element-wise - used to accumulate deltas across requests or repeats."""
    return _combine(a, b, +1)


def summarize(delta: dict) -> dict:
    """Derived stats from a counter delta. Values are None when nothing was drafted.

    acceptance_rate         accepted / drafted tokens            (0..1)
    mean_acceptance_length  1 + accepted / drafts = tokens per target pass (1..gamma+1)
    per_position_rate       accepted_at_pos_i / drafts           (decays with i)
    """
    drafts, drafted, accepted = delta.get("drafts", 0), delta.get("draft_tokens", 0), delta.get("accepted_tokens", 0)
    return {
        "drafts": drafts,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
        "mean_acceptance_length": 1 + accepted / drafts if drafts else None,
        "per_position_rate": [p / drafts for p in delta.get("per_pos", [])] if drafts else [],
    }


def read(llm) -> dict:
    """Best-effort snapshot from a live vllm.LLM. Never raises."""
    try:
        return snapshot(llm.get_metrics())
    except Exception as e:  # metrics API differs across vLLM versions
        return {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0, "per_pos": [], "error": repr(e)}
