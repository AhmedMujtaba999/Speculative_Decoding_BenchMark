from types import SimpleNamespace as M

import pytest

from specdec import metrics


def _metrics(drafts, drafted, accepted, per_pos, suffix=""):
    return [
        M(name=f"vllm:spec_decode_num_drafts{suffix}", value=drafts),
        M(name=f"vllm:spec_decode_num_draft_tokens{suffix}", value=drafted),
        M(name=f"vllm:spec_decode_num_accepted_tokens{suffix}", value=accepted),
        M(name=f"vllm:spec_decode_num_accepted_tokens_per_pos{suffix}", values=per_pos),
        M(name="vllm:num_requests_running", value=3),  # unrelated metric is ignored
    ]


def test_snapshot_reads_counters_and_vector():
    s = metrics.snapshot(_metrics(10, 50, 30, [9, 8, 6, 4, 3]))
    assert s == {"drafts": 10, "draft_tokens": 50, "accepted_tokens": 30, "per_pos": [9, 8, 6, 4, 3]}


def test_snapshot_accepts_prometheus_total_suffix():
    s = metrics.snapshot(_metrics(1, 5, 2, [1, 1, 0, 0, 0], suffix="_total"))
    assert s["drafts"] == 1 and s["accepted_tokens"] == 2


def test_snapshot_sums_engines():
    s = metrics.snapshot(_metrics(1, 3, 2, [1, 1, 0]) + _metrics(2, 6, 3, [2, 1, 0]))
    assert s["drafts"] == 3 and s["per_pos"] == [3, 2, 0]


def test_diff_and_summarize():
    before = metrics.snapshot(_metrics(10, 50, 30, [9, 8, 6, 4, 3]))
    after = metrics.snapshot(_metrics(30, 150, 90, [27, 24, 18, 12, 9]))
    d = metrics.diff(before, after)
    assert d == {"drafts": 20, "draft_tokens": 100, "accepted_tokens": 60, "per_pos": [18, 16, 12, 8, 6]}
    s = metrics.summarize(d)
    assert s["acceptance_rate"] == pytest.approx(0.6)
    assert s["mean_acceptance_length"] == pytest.approx(4.0)  # 1 + 60/20
    assert s["per_position_rate"] == pytest.approx([0.9, 0.8, 0.6, 0.4, 0.3])
    # accepted tokens == sum of per-position counts (consistency check)
    assert sum(d["per_pos"]) == d["accepted_tokens"]


def test_add_accumulates():
    a = {"drafts": 1, "draft_tokens": 3, "accepted_tokens": 2, "per_pos": [1, 1]}
    b = {"drafts": 2, "draft_tokens": 6, "accepted_tokens": 3, "per_pos": [2, 1, 0]}
    assert metrics.add(a, b) == {"drafts": 3, "draft_tokens": 9, "accepted_tokens": 5, "per_pos": [3, 2, 0]}
    assert metrics.add(metrics.zero(), a) == a


def test_baseline_has_no_spec_stats():
    s = metrics.summarize(metrics.zero())
    assert s["acceptance_rate"] is None and s["mean_acceptance_length"] is None and s["per_position_rate"] == []


def test_read_never_raises():
    class Broken:
        def get_metrics(self):
            raise RuntimeError("stat logging disabled")
    s = metrics.read(Broken())
    assert s["drafts"] == 0 and "RuntimeError" in s["error"]
