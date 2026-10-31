import math

import pytest

from specdec import theory


def test_tokens_per_step_known_value():
    # (1 - 0.8^6) / 0.2 = 3.68928
    assert theory.expected_tokens_per_step(0.8, 5) == pytest.approx(3.68928)


@pytest.mark.parametrize("gamma", [1, 3, 8])
def test_tokens_per_step_bounds(gamma):
    assert theory.expected_tokens_per_step(0.0, gamma) == pytest.approx(1.0)  # target always adds 1
    assert theory.expected_tokens_per_step(1.0, gamma) == pytest.approx(gamma + 1)  # all accepted + bonus
    assert theory.expected_tokens_per_step(0.999999, gamma) == pytest.approx(gamma + 1, rel=1e-4)


def test_gamma_zero_is_plain_decoding():
    assert theory.expected_speedup(0.9, 0, c=0.5) == pytest.approx(1.0)


def test_speedup_monotone_in_alpha():
    vals = [theory.expected_speedup(a / 10, 5, c=0.1) for a in range(11)]
    assert vals == sorted(vals)


def test_bad_draft_makes_things_slower():
    # alpha=0 -> 1 token/step but we paid for gamma draft passes
    assert theory.expected_speedup(0.0, 5, c=0.2) == pytest.approx(1 / 2.0)


def test_acceptance_rate_is_not_alpha():
    rate = theory.expected_acceptance_rate(0.8, 5)
    assert rate < 0.8  # acceptance stops at first miss, later positions rarely reached
    assert rate == pytest.approx((3.68928 - 1) / 5)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.75, 0.95])
@pytest.mark.parametrize("gamma", [2, 5, 8])
def test_inverse_roundtrip(alpha, gamma):
    tps = theory.expected_tokens_per_step(alpha, gamma)
    assert theory.alpha_from_tokens_per_step(tps, gamma) == pytest.approx(alpha, abs=1e-6)
    rate = theory.expected_acceptance_rate(alpha, gamma)
    assert theory.alpha_from_acceptance_rate(rate, gamma) == pytest.approx(alpha, abs=1e-6)


def test_position_curve_decays():
    curve = theory.expected_position_acceptance(0.7, 4)
    assert curve == pytest.approx([0.7, 0.49, 0.343, 0.2401])


def test_optimal_gamma():
    # free draft + good acceptance -> speculate as far as allowed
    assert theory.optimal_gamma(0.9, 0.0, max_gamma=10)[0] == 10
    # expensive draft -> short drafts win
    g, s = theory.optimal_gamma(0.6, 0.3, max_gamma=10)
    assert 1 <= g <= 3 and s > 1
    # brute-force check
    best = max(range(1, 11), key=lambda k: theory.expected_speedup(0.6, k, 0.3))
    assert g == best


def test_input_validation():
    with pytest.raises(ValueError):
        theory.expected_tokens_per_step(1.2, 3)
    with pytest.raises(ValueError):
        theory.expected_speedup(0.5, 3, c=-1)
    with pytest.raises(ValueError):
        theory.alpha_from_tokens_per_step(9.0, 3)


def test_cli_runs(capsys):
    theory._main(["--alpha", "0.8", "--c", "0.15", "--max-gamma", "6"])
    out = capsys.readouterr().out
    assert "best gamma" in out and not math.isnan(float(out.split("->")[-1].split("x")[0]))
