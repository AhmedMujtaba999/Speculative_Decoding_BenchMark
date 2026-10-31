"""Benchmark vLLM with and without speculative decoding.

Single run : python bench.py --mode ngram --gamma 5
Everything : python bench.py --mode all                 (baseline + ngram/draft x gamma 3,5,8)
Preview    : python bench.py --mode all --dry-run       (prints the plan, needs no GPU)

Each config runs in its own subprocess so GPU memory is fully released between runs.
Two workloads are measured per config (pick with --workload):

  batch    all prompts submitted at once -> throughput (tokens/s) under batching
  latency  one prompt at a time (batch size 1) -> per-request speed, per-category stats

Speculative decoding is a *latency* optimization for memory-bound decode, so expect the
biggest wins in `latency` and smaller (sometimes negative) wins in `batch`.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from specdec import metrics as specm
from specdec.prompts import build_prompts

TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"
MODES = ("baseline", "ngram", "draft", "suffix")
SCHEMA = 2


# --------------------------------------------------------------------------- config
def spec_config(mode: str, gamma: int, draft_model: str | None = None,
                lookup_min: int = 2, lookup_max: int = 4) -> dict | None:
    """The `speculative_config` dict handed to vllm.LLM (None = speculation off)."""
    if mode == "baseline":
        return None
    if gamma < 1:
        raise ValueError("gamma must be >= 1 for speculative modes")
    if mode == "ngram":   # prompt-lookup: copy the continuation of a matching n-gram
        return {"method": "ngram", "num_speculative_tokens": gamma,
                "prompt_lookup_min": lookup_min, "prompt_lookup_max": lookup_max}
    if mode == "draft":   # small same-family model proposes, big model verifies
        if not draft_model:
            raise ValueError("draft mode needs --draft-model")
        return {"method": "draft_model", "model": draft_model, "num_speculative_tokens": gamma}
    if mode == "suffix":  # suffix-tree lookup over prompt + past outputs (pip install arctic-inference)
        return {"method": "suffix", "num_speculative_tokens": gamma}
    raise ValueError(f"unknown mode {mode!r}")


def build_plan(methods: list[str], gammas: list[int]) -> list[tuple[str, int]]:
    """baseline first, then every (method, gamma) pair."""
    return [("baseline", 0)] + [(m, g) for m in methods for g in gammas]


def result_path(out_dir: str | Path, mode: str, gamma: int) -> Path:
    return Path(out_dir) / f"{mode}_g{gamma}.json"


# --------------------------------------------------------------------------- helpers
def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        import vllm
        info["vllm"] = vllm.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return info


def kv_cache_blocks(llm) -> int | None:
    """KV-cache capacity after weights load. A draft model's weights + cache eat into it."""
    for path in ("llm_engine.vllm_config.cache_config.num_gpu_blocks",
                 "llm_engine.cache_config.num_gpu_blocks"):
        obj = llm
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if obj:
                return int(obj)
        except Exception:
            continue
    return None


def _chat(llm, messages, sp):
    t0 = time.perf_counter()
    outs = llm.chat(messages, sp, use_tqdm=False)
    return outs, time.perf_counter() - t0


# --------------------------------------------------------------------------- workloads
def run_batch(llm, messages, sp, repeats: int) -> tuple[dict, list[str]]:
    """All prompts at once; median wall time over `repeats`."""
    times, total = [], specm.zero()
    outs = None
    for _ in range(repeats):
        before = specm.read(llm)
        outs, dt = _chat(llm, messages, sp)
        total = specm.add(total, specm.diff(before, specm.read(llm)))
        times.append(dt)
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    elapsed = statistics.median(times)
    return {
        "num_prompts": len(messages), "output_tokens": n_tokens,
        "elapsed_s": elapsed, "elapsed_all_s": times,
        "tokens_per_s": n_tokens / elapsed,
        "spec": specm.summarize(total),
    }, [o.outputs[0].text for o in outs]


def run_latency(llm, messages, cats, sp) -> tuple[dict, list[str]]:
    """One request at a time (batch size 1): the regime speculative decoding targets."""
    rows, texts, total = [], [], specm.zero()
    by_cat: dict[str, dict] = {}
    for msg, cat in zip(messages, cats):
        before = specm.read(llm)
        outs, dt = _chat(llm, [msg], sp)
        delta = specm.diff(before, specm.read(llm))
        n = len(outs[0].outputs[0].token_ids)
        rows.append({"category": cat, "output_tokens": n, "elapsed_s": dt,
                     "tokens_per_s": n / dt, "spec": specm.summarize(delta)})
        texts.append(outs[0].outputs[0].text)
        total = specm.add(total, delta)
        c = by_cat.setdefault(cat, {"tokens": 0, "time": 0.0, "delta": specm.zero()})
        c["tokens"] += n
        c["time"] += dt
        c["delta"] = specm.add(c["delta"], delta)

    tok = sum(r["output_tokens"] for r in rows)
    secs = sum(r["elapsed_s"] for r in rows)
    return {
        "output_tokens": tok, "elapsed_s": secs,
        "tokens_per_s": tok / secs, "ms_per_token": 1000 * secs / tok,
        "spec": specm.summarize(total),
        "by_category": {
            cat: {"tokens_per_s": c["tokens"] / c["time"], **{
                k: v for k, v in specm.summarize(c["delta"]).items()
                if k in ("acceptance_rate", "mean_acceptance_length")}}
            for cat, c in by_cat.items()},
        "requests": rows,
    }, texts


# --------------------------------------------------------------------------- runners
def run_one(args) -> dict:
    from vllm import LLM, SamplingParams  # lazy: --dry-run / --mode all need no GPU

    cfg = spec_config(args.mode, args.gamma, args.draft_model, args.lookup_min, args.lookup_max)
    # disable_log_stats=False: the offline LLM turns stat logging off by default, and without it
    # llm.get_metrics() has no spec-decode counters -> acceptance would silently come back empty.
    kwargs = dict(model=args.model, gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_model_len, seed=0, dtype=args.dtype,
                  disable_log_stats=False)
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    if cfg:
        kwargs["speculative_config"] = cfg

    t_load = time.perf_counter()
    llm = LLM(**kwargs)
    load_s = time.perf_counter() - t_load

    cats, prompts = zip(*build_prompts(args.num_prompts))
    messages = [[{"role": "user", "content": p}] for p in prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)  # greedy: output must match baseline
    llm.chat(messages[:2], sp, use_tqdm=False)  # warmup: CUDA graphs, torch.compile, allocator

    gamma = args.gamma if cfg else 0
    result = {
        "schema": SCHEMA, "status": "ok", "mode": args.mode, "gamma": gamma,
        "model": args.model, "draft_model": args.draft_model if args.mode == "draft" else None,
        "speculative_config": cfg, "max_tokens": args.max_tokens,
        "env": env_info(), "load_s": load_s, "kv_cache_blocks": kv_cache_blocks(llm),
        "categories": list(cats),
    }
    if args.workload in ("batch", "both"):
        result["batch"], result["batch_texts"] = run_batch(llm, messages, sp, args.repeats)
    if args.workload in ("latency", "both"):
        result["latency"], result["latency_texts"] = run_latency(llm, messages, cats, sp)

    # flat fields kept for quick inspection / backward compatibility with schema 1
    main = result.get("batch") or result["latency"]
    result["tokens_per_s"] = main["tokens_per_s"]
    result["acceptance_rate"] = main["spec"]["acceptance_rate"]
    result["tokens_per_step"] = main["spec"]["mean_acceptance_length"]
    result["texts"] = result.get("latency_texts") or result["batch_texts"]
    return result


def write_result(result: dict, out_dir: str) -> Path:
    path = result_path(out_dir, result["mode"], result["gamma"])
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2))
    return path


def write_failure(mode: str, gamma: int, out_dir: str, error: str, tb: str = "") -> Path:
    return write_result({"schema": SCHEMA, "status": "failed", "mode": mode, "gamma": gamma,
                         "error": error, "traceback": tb, "env": env_info()}, out_dir)


def main_one(args) -> int:
    gamma = args.gamma if args.mode != "baseline" else 0
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "gamma": gamma,
                          "speculative_config": spec_config(args.mode, args.gamma, args.draft_model,
                                                            args.lookup_min, args.lookup_max)}, indent=2))
        return 0
    try:
        result = run_one(args)
    except Exception as e:  # e.g. this vLLM build does not support the method
        path = write_failure(args.mode, gamma, args.out_dir, repr(e), traceback.format_exc())
        print(f"[{args.mode} gamma={gamma}] FAILED: {e!r}  -> {path}", file=sys.stderr)
        return 1
    path = write_result(result, args.out_dir)
    b, l = result.get("batch"), result.get("latency")
    msg = f"[{args.mode} gamma={gamma}]"
    if b:
        msg += f" batch {b['tokens_per_s']:.1f} tok/s"
    if l:
        msg += f" | bs=1 {l['tokens_per_s']:.1f} tok/s"
    if result["acceptance_rate"] is not None:
        msg += f" | accept {result['acceptance_rate']:.2f} | {result['tokens_per_step']:.2f} tok/step"
    print(f"{msg}  -> {path}")
    return 0


def run_all(args) -> int:
    plan = build_plan(args.methods, args.gammas)
    failed = 0
    for mode, g in plan:
        cmd = [sys.executable, __file__, "--mode", mode, "--gamma", str(g or 1),
               "--model", args.model, "--draft-model", args.draft_model,
               "--num-prompts", str(args.num_prompts), "--max-tokens", str(args.max_tokens),
               "--gpu-util", str(args.gpu_util), "--max-model-len", str(args.max_model_len),
               "--dtype", args.dtype, "--workload", args.workload, "--repeats", str(args.repeats),
               "--lookup-min", str(args.lookup_min), "--lookup-max", str(args.lookup_max),
               "--out-dir", args.out_dir]
        if args.enforce_eager:
            cmd.append("--enforce-eager")
        print(">>", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failed += 1
            if not result_path(args.out_dir, mode, g).exists():  # crashed before writing (OOM kill, segfault)
                write_failure(mode, g, args.out_dir, f"subprocess exited with code {rc}")
    print(f"done: {len(plan) - failed}/{len(plan)} configs succeeded" if not args.dry_run
          else f"dry run: {len(plan)} configs")
    return 0


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=[*MODES, "all"], default="all")
    ap.add_argument("--model", default=TARGET, help="target model")
    ap.add_argument("--draft-model", default=DRAFT, help="draft model for --mode draft (same tokenizer family)")
    ap.add_argument("--gamma", type=int, default=5, help="num_speculative_tokens for a single run")
    ap.add_argument("--methods", type=lambda s: [m.strip() for m in s.split(",")], default=["ngram", "draft"],
                    help="speculative methods for --mode all (default: ngram,draft; add suffix if installed)")
    ap.add_argument("--gammas", type=_int_list, default=[3, 5, 8], help="gammas for --mode all")
    ap.add_argument("--lookup-min", type=int, default=2, help="n-gram: prompt_lookup_min")
    ap.add_argument("--lookup-max", type=int, default=4, help="n-gram: prompt_lookup_max")
    ap.add_argument("--workload", choices=["both", "batch", "latency"], default="both")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats of the batch workload (median)")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-util", type=float, default=0.8)
    ap.add_argument("--dtype", default="auto", help="auto | bfloat16 | half (use half on T4/V100)")
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA graphs (faster start, slower decode)")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--dry-run", action="store_true", help="print configs/commands, run nothing")
    a = ap.parse_args(argv)
    for m in a.methods:
        if m not in MODES or m == "baseline":
            ap.error(f"--methods: unknown speculative method {m!r}")
    return a


if __name__ == "__main__":
    a = parse_args()
    sys.exit(run_all(a) if a.mode == "all" else main_one(a))
