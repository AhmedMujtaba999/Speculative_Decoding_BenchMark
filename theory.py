"""Closed-form model of speculative decoding (Leviathan, Kalman & Matias, 2023).

Notation used everywhere in this repo:

    alpha  per-token acceptance probability: chance the target agrees with the next
           drafted token (assumed independent per position - a simplification)
    gamma  number of speculative tokens drafted per step (vLLM: num_speculative_tokens)
    c      cost of ONE draft forward pass relative to ONE target forward pass
           (~0 for n-gram lookup, ~0.1-0.3 for a small same-family draft model)

One "step" = draft gamma tokens, then ONE target forward pass that scores all of them.
Tokens are kept up to the first disagreement, and the target always contributes one
token of its own (the correction, or a bonus token if everything was accepted), so a
step yields between 1 and gamma + 1 tokens.

Run as a script for a quick table:
    python -m specdec.theory --alpha 0.8 --c 0.15
"""
from __future__ import annotations

import argparse


def _check(alpha: float, gamma: int) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")


def expected_tokens_per_step(alpha: float, gamma: int) -> float:
    """E[tokens per target forward pass] = (1 - alpha^(gamma+1)) / (1 - alpha).

    vLLM calls the measured version of this the *mean acceptance length*
    (= 1 + accepted_tokens / num_drafts).
    """
    _check(alpha, gamma)
    if alpha == 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, gamma: int, c: float = 0.0) -> float:
    """Expected wall-clock speedup vs plain decoding: tokens_per_step / (gamma*c + 1).

    Assumes decode is memory-bound, so verifying gamma+1 tokens costs about the same
    as generating 1 token. That holds at small batch sizes; at large batch sizes the
    GPU becomes compute-bound and real speedups shrink (this is what bench.py's
    `latency` vs `batch` workloads show).
    """
    if c < 0:
        raise ValueError("c must be >= 0")
    return expected_tokens_per_step(alpha, gamma) / (gamma * c + 1.0)


def expected_acceptance_rate(alpha: float, gamma: int) -> float:
    """What vLLM reports as acceptance rate: accepted_tokens / draft_tokens.

    Note this is NOT alpha. Because acceptance stops at the first miss, later
    positions are rarely reached, so the rate is always <= alpha.
    """
    if gamma == 0:
        return 0.0
    return (expected_tokens_per_step(alpha, gamma) - 1.0) / gamma


def expected_position_acceptance(alpha: float, gamma: int) -> list[float]:
    """P(draft position i is accepted) = alpha^(i+1): the per-position curve decays."""
    _check(alpha, gamma)
    return [alpha ** (i + 1) for i in range(gamma)]


def _bisect(fn, target: float, lo: float = 0.0, hi: float = 1.0, iters: int = 80) -> float:
    for _ in range(iters):
        mid = (lo + hi) / 2
        if fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def alpha_from_tokens_per_step(tokens_per_step: float, gamma: int) -> float:
    """Invert expected_tokens_per_step: recover alpha from a measured mean acceptance length."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 1.0 <= tokens_per_step <= gamma + 1:
        raise ValueError(f"tokens_per_step must be in [1, {gamma + 1}]")
    return _bisect(lambda a: expected_tokens_per_step(a, gamma), tokens_per_step)


def alpha_from_acceptance_rate(rate: float, gamma: int) -> float:
    """Invert expected_acceptance_rate: recover alpha from vLLM's acceptance rate."""
    return alpha_from_tokens_per_step(1.0 + rate * gamma, gamma)


def optimal_gamma(alpha: float, c: float, max_gamma: int = 16) -> tuple[int, float]:
    """Best gamma in 1..max_gamma for this alpha and draft cost. Returns (gamma, speedup)."""
    best = max(range(1, max_gamma + 1), key=lambda g: expected_speedup(alpha, g, c))
    return best, expected_speedup(alpha, best, c)


def _main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Expected speculative decoding speedup table")
    ap.add_argument("--alpha", type=float, default=0.8, help="per-token acceptance prob")
    ap.add_argument("--c", type=float, default=0.0, help="draft cost / target cost (n-gram ~0)")
    ap.add_argument("--max-gamma", type=int, default=10)
    a = ap.parse_args(argv)

    print(f"alpha={a.alpha}  c={a.c}")
    print(f"{'gamma':>5} | {'tokens/step':>11} | {'accept rate':>11} | {'speedup':>7}")
    print("-" * 44)
    for g in range(1, a.max_gamma + 1):
        print(f"{g:>5} | {expected_tokens_per_step(a.alpha, g):>11.2f} | "
              f"{expected_acceptance_rate(a.alpha, g):>11.2f} | "
              f"{expected_speedup(a.alpha, g, a.c):>6.2f}x")
    g, s = optimal_gamma(a.alpha, a.c, a.max_gamma)
    print(f"best gamma = {g}  ->  {s:.2f}x expected speedup")


if __name__ == "__main__":
    _main()
