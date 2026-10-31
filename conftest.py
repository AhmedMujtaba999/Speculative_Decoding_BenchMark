"""Shared fixtures. `fake_results` writes SYNTHETIC schema-2 result files (for testing
report.py on CPU) - these numbers are made up and never shipped as benchmark results."""
import json

import pytest

from specdec.prompts import CATEGORIES

TEXTS = [f"answer {i}" for i in range(8)]


def make_result(mode, gamma, batch_tps, lat_tps, cat_tps=None, accept=None, tps_step=None,
                per_pos=None, texts=None, kv=1000):
    cats = [CATEGORIES[i % 4] for i in range(8)]
    spec = {"acceptance_rate": accept, "mean_acceptance_length": tps_step,
            "per_position_rate": per_pos or []}
    cat_tps = cat_tps or {c: lat_tps for c in CATEGORIES}
    return {
        "schema": 2, "status": "ok", "mode": mode, "gamma": gamma,
        "model": "fake/target", "draft_model": "fake/draft" if mode == "draft" else None,
        "max_tokens": 64, "env": {"gpu": "FakeGPU", "vllm": "0.0-test"},
        "kv_cache_blocks": kv, "categories": cats,
        "batch": {"tokens_per_s": batch_tps, "spec": spec},
        "latency": {"tokens_per_s": lat_tps, "spec": spec,
                    "by_category": {c: {"tokens_per_s": v, "acceptance_rate": accept,
                                        "mean_acceptance_length": tps_step}
                                    for c, v in cat_tps.items()}},
        "latency_texts": texts or TEXTS, "batch_texts": texts or TEXTS,
        "tokens_per_s": batch_tps, "acceptance_rate": accept, "tokens_per_step": tps_step,
        "texts": texts or TEXTS,
    }


@pytest.fixture
def fake_results(tmp_path):
    rows = [
        make_result("baseline", 0, 1000.0, 50.0),
        make_result("ngram", 3, 1100.0, 80.0,
                    cat_tps={"code_edit": 120.0, "extract": 100.0, "rag_qa": 70.0, "open_ended": 52.0},
                    accept=0.6, tps_step=2.8, per_pos=[0.8, 0.6, 0.4]),
        make_result("draft", 3, 900.0, 75.0, accept=0.7, tps_step=3.1, per_pos=[0.9, 0.7, 0.5],
                    texts=TEXTS[:-1] + ["different"], kv=700),
    ]
    for r in rows:
        (tmp_path / f'{r["mode"]}_g{r["gamma"]}.json').write_text(json.dumps(r))
    (tmp_path / "suffix_g5.json").write_text(json.dumps(
        {"schema": 2, "status": "failed", "mode": "suffix", "gamma": 5,
         "error": "ModuleNotFoundError('arctic_inference')"}))
    return tmp_path
