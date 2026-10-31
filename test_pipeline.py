"""End-to-end bench.py -> results/*.json -> report.py on CPU, with a FAKE vllm module.

The fake mimics the parts of vllm.LLM that bench.py touches (chat, get_metrics, engine config),
so the whole pipeline - workloads, counter diffs, per-category stats, JSON, report - is exercised
without a GPU. Numbers it produces are synthetic.
"""
import json
import sys
import types
from types import SimpleNamespace as NS

import pytest

import bench
import report


class FakeLLM:
    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        FakeLLM.last_kwargs = kwargs
        self.spec = kwargs.get("speculative_config")
        self.c = {"drafts": 0, "draft_tokens": 0, "accepted": 0, "per_pos": [0] * (self.spec or {}).get(
            "num_speculative_tokens", 0)}
        self.llm_engine = NS(vllm_config=NS(cache_config=NS(num_gpu_blocks=900 if self.spec else 1000)))

    def chat(self, messages, sp, use_tqdm=True):
        outs = []
        for m in messages:
            text = m[0]["content"]
            outs.append(NS(outputs=[NS(text=f"answer to {hash(text) % 997}", token_ids=list(range(20)))]))
            if self.spec:
                g = self.spec["num_speculative_tokens"]
                copyable = "```" in text or "JSON" in text
                acc_per_draft = g - 1 if copyable else 1  # code/extract prompts accept more
                self.c["drafts"] += 5
                self.c["draft_tokens"] += 5 * g
                self.c["accepted"] += 5 * acc_per_draft
                for i in range(acc_per_draft):
                    self.c["per_pos"][i] += 5
        return outs

    def get_metrics(self):
        if not FakeLLM.last_kwargs.get("disable_log_stats") is False:
            return []  # like vLLM: no stats unless logging is enabled
        return [NS(name="vllm:spec_decode_num_drafts", value=self.c["drafts"]),
                NS(name="vllm:spec_decode_num_draft_tokens", value=self.c["draft_tokens"]),
                NS(name="vllm:spec_decode_num_accepted_tokens", value=self.c["accepted"]),
                NS(name="vllm:spec_decode_num_accepted_tokens_per_pos", values=list(self.c["per_pos"]))]


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    mod.SamplingParams = lambda **kw: kw
    mod.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


def _run(tmp_path, *extra):
    args = bench.parse_args(["--num-prompts", "8", "--repeats", "2", "--out-dir", str(tmp_path), *extra])
    assert bench.main_one(args) == 0


def test_llm_gets_stats_logging_enabled(fake_vllm, tmp_path):
    _run(tmp_path, "--mode", "ngram", "--gamma", "4")
    kw = FakeLLM.last_kwargs
    assert kw["disable_log_stats"] is False
    assert kw["speculative_config"]["method"] == "ngram" and kw["seed"] == 0


def test_result_json_contents(fake_vllm, tmp_path):
    _run(tmp_path, "--mode", "ngram", "--gamma", "4")
    r = json.loads((tmp_path / "ngram_g4.json").read_text())
    assert r["status"] == "ok" and r["schema"] == 2 and r["kv_cache_blocks"] == 900
    assert r["batch"]["output_tokens"] == 160 and len(r["batch"]["elapsed_all_s"]) == 2
    lat = r["latency"]
    assert len(lat["requests"]) == 8 and set(lat["by_category"]) == {"code_edit", "extract", "rag_qa", "open_ended"}
    # warmup and the batch workload must not leak into the latency counters
    s = lat["spec"]
    assert s["drafts"] == 8 * 5
    assert 1 <= s["mean_acceptance_length"] <= 5 and len(s["per_position_rate"]) == 4
    assert lat["by_category"]["code_edit"]["acceptance_rate"] > lat["by_category"]["open_ended"]["acceptance_rate"]


def test_full_pipeline_report(fake_vllm, tmp_path):
    _run(tmp_path, "--mode", "baseline")
    _run(tmp_path, "--mode", "ngram", "--gamma", "4")
    _run(tmp_path, "--mode", "draft", "--gamma", "3")
    rows = report.make_report(tmp_path)
    assert [r["label"] for r in rows] == ["baseline", "ngram γ=4", "draft γ=3"]
    assert all(r["match"] == 1.0 for r in rows)
    assert rows[1]["acceptance_rate"] is not None and rows[0]["acceptance_rate"] is None
    assert (tmp_path / "report.md").exists() and (tmp_path / "report.png").exists()


def test_unsupported_method_is_recorded_not_fatal(fake_vllm, tmp_path, monkeypatch):
    class Rejects(FakeLLM):
        def __init__(self, **kw):
            if kw.get("speculative_config", {}).get("method") == "draft_model":
                raise NotImplementedError("draft_model is not supported in this build")
            super().__init__(**kw)
    monkeypatch.setattr(fake_vllm, "LLM", Rejects)
    args = bench.parse_args(["--mode", "draft", "--gamma", "3", "--out-dir", str(tmp_path)])
    assert bench.main_one(args) == 1
    r = json.loads((tmp_path / "draft_g3.json").read_text())
    assert r["status"] == "failed" and "not supported" in r["error"]
