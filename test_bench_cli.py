"""bench.py logic that does not need a GPU (vllm is imported lazily inside run_one)."""
import json

import pytest

import bench


def test_spec_configs():
    assert bench.spec_config("baseline", 5) is None
    assert bench.spec_config("ngram", 5) == {"method": "ngram", "num_speculative_tokens": 5,
                                             "prompt_lookup_min": 2, "prompt_lookup_max": 4}
    assert bench.spec_config("draft", 3, "Qwen/Qwen2.5-0.5B-Instruct") == {
        "method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 3}
    assert bench.spec_config("suffix", 8)["method"] == "suffix"


@pytest.mark.parametrize("bad", [("ngram", 0, None), ("draft", 3, None), ("nope", 3, None)])
def test_spec_config_errors(bad):
    with pytest.raises(ValueError):
        bench.spec_config(*bad)


def test_plan():
    plan = bench.build_plan(["ngram", "draft"], [3, 5, 8])
    assert plan[0] == ("baseline", 0) and len(plan) == 7
    assert ("draft", 8) in plan


def test_parse_args_lists():
    a = bench.parse_args(["--methods", "ngram,suffix", "--gammas", "2,4"])
    assert a.methods == ["ngram", "suffix"] and a.gammas == [2, 4]
    with pytest.raises(SystemExit):
        bench.parse_args(["--methods", "eagle"])


def test_dry_run_all_runs_nothing(tmp_path, capsys):
    a = bench.parse_args(["--mode", "all", "--dry-run", "--out-dir", str(tmp_path)])
    bench.run_all(a)
    out = capsys.readouterr().out
    assert out.count(">>") == 7 and "dry run: 7 configs" in out
    assert not list(tmp_path.iterdir())


def test_dry_run_single_prints_config(capsys):
    bench.main_one(bench.parse_args(["--mode", "ngram", "--gamma", "4", "--dry-run"]))
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["speculative_config"]["num_speculative_tokens"] == 4


def test_failure_file_is_readable_by_report(tmp_path):
    p = bench.write_failure("draft", 5, tmp_path, "NotImplementedError('draft_model')")
    assert p.name == "draft_g5.json" and json.loads(p.read_text())["status"] == "failed"
