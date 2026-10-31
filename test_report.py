"""report.py on synthetic results (see conftest.fake_results) - CPU only."""
import json

import pytest

import report


def test_load_derives_fields(fake_results):
    rows, failed = report.load(fake_results)
    assert [r["label"] for r in rows] == ["baseline", "ngram γ=3", "draft γ=3"]
    assert len(failed) == 1 and failed[0]["mode"] == "suffix"
    ng = rows[1]
    assert ng["batch_speedup"] == pytest.approx(1.1)
    assert ng["latency_speedup"] == pytest.approx(1.6)
    assert ng["category_speedup"]["code_edit"] == pytest.approx(2.4)
    assert ng["category_speedup"]["open_ended"] == pytest.approx(1.04)
    assert ng["match"] == 1.0
    assert rows[2]["match"] == pytest.approx(7 / 8)
    assert 0 < ng["alpha_est"] < 1


def test_baseline_has_no_acceptance(fake_results):
    base = report.load(fake_results)[0][0]
    assert base["acceptance_rate"] is None and base["alpha_est"] is None


def test_make_report_writes_files(fake_results, tmp_path):
    assets = tmp_path / "assets"
    report.make_report(fake_results, assets=str(assets))
    md = (fake_results / "report.md").read_text()
    assert "| ngram γ=3 |" in md and "1.60x" in md
    assert "Speedup by prompt category" in md
    assert "Failed or unsupported configs" in md and "arctic_inference" in md
    assert "Greedy-match below 90%" in md and "draft γ=3" in md
    for name in ("report.png", "speedup.png", "category_heatmap.png",
                 "acceptance_length.png", "position_acceptance.png"):
        assert (fake_results / name).stat().st_size > 10_000
        assert (assets / f"results_{name}").exists()


def test_schema1_results_still_load(tmp_path):
    old = {"mode": "baseline", "gamma": 0, "tokens_per_s": 100.0, "acceptance_rate": None,
           "tokens_per_step": None, "texts": ["a", "b"], "categories": ["code_edit", "extract"]}
    new = dict(old, mode="ngram", gamma=5, tokens_per_s=130.0, acceptance_rate=0.5, tokens_per_step=3.5)
    for r in (old, new):
        (tmp_path / f'{r["mode"]}_g{r["gamma"]}.json').write_text(json.dumps(r))
    rows, _ = report.load(tmp_path)
    assert rows[1]["batch_speedup"] == pytest.approx(1.3) and rows[1]["latency_speedup"] is None
    report.make_report(tmp_path)  # charts must cope with missing latency data
    assert (tmp_path / "report.png").exists()


def test_no_results_exits(tmp_path):
    with pytest.raises(SystemExit):
        report.make_report(tmp_path)
