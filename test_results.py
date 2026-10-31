"""Checks on REAL benchmark output in results/ (skipped until you run bench.py on a GPU).

Point at another folder with:  SPECDEC_RESULTS=path/to/results pytest tests/test_results.py
"""
import os
from pathlib import Path

import pytest

import report

RESULTS = Path(os.environ.get("SPECDEC_RESULTS", "results"))
pytestmark = pytest.mark.skipif(not list(RESULTS.glob("*_g*.json")),
                                reason=f"no benchmark results in {RESULTS}/ - run bench.py on a GPU first")


@pytest.fixture(scope="module")
def rows():
    ok, _ = report.load(RESULTS)
    return ok


def _spec(rows):
    return [r for r in rows if r["mode"] != "baseline"]


def test_has_baseline(rows):
    assert any(r["mode"] == "baseline" for r in rows), "speedups need a baseline run"


def test_greedy_outputs_match_baseline(rows):
    # Speculative decoding is lossless; with greedy sampling outputs should match.
    # Allow a little bf16/batch-shape drift (vLLM docs: floating point can flip a near-tie).
    for r in _spec(rows):
        if r["match"] is not None:
            assert r["match"] >= 0.75, f'{r["label"]}: only {r["match"]:.0%} of outputs match baseline'


def test_acceptance_stats_are_sane(rows):
    for r in _spec(rows):
        if r["acceptance_rate"] is None:
            continue
        assert 0.0 <= r["acceptance_rate"] <= 1.0
        assert 1.0 <= r["tokens_per_step"] <= r["gamma"] + 1 + 1e-6
        if r["per_position_rate"]:
            assert len(r["per_position_rate"]) == r["gamma"]
            # position i+1 can only be accepted if position i was
            p = r["per_position_rate"]
            assert all(a + 1e-9 >= b for a, b in zip(p, p[1:])), f'{r["label"]}: per-position not decreasing'


def test_ngram_prefers_copy_heavy_prompts(rows):
    """Hypothesis from the README - reported as xfail (not failure) if your run disagrees."""
    ng = [r for r in rows if r["mode"] == "ngram" and r["category_accept"]]
    if not ng:
        pytest.skip("no per-category n-gram data")
    r = max(ng, key=lambda x: x["gamma"])
    copy = r["category_accept"].get("code_edit")
    free = r["category_accept"].get("open_ended")
    if copy is None or free is None:
        pytest.skip("missing categories")
    if not copy > free:
        pytest.xfail(f"n-gram acceptance code_edit={copy:.2f} <= open_ended={free:.2f} on this run")
