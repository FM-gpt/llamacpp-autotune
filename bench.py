#!/usr/bin/env python
"""
bench.py - the fixed measurement harness for llamacpp-autotune.

This is the read-only "harness" (analogous to autoresearch's prepare.py). It does
NOT decide what to try - it just measures ONE knob config faithfully and prints a
clean summary. The optimizer loop (you, driven by program.md) chooses configs,
reads these numbers, and decides keep/discard.

What it measures for a given config:
  - generation throughput  (tg, tok/s)  mean +/- stddev   <- the objective
  - prompt throughput      (pp, tok/s)  mean +/- stddev   <- reported secondary
  - peak VRAM (MiB)        sampled from nvidia-smi during the run
  - perplexity             ONLY when a lossy knob changes (KV-quant etc.); else skipped

It is a thin wrapper over `llama-bench -o json` (which already repeats -r times and
reports mean/stddev) plus a VRAM sampler thread and an optional llama-perplexity gate.

Usage:
  python bench.py                          # baseline (incumbent good config)
  python bench.py --fa off                 # one knob changed
  python bench.py --ctk q8_0 --ctv q8_0    # triggers the perplexity gate
  python bench.py --batch 512 --ubatch 128 --desc "small batch"

Nothing here writes to results.tsv - logging keep/discard is the loop's job
(see program.md). This script only measures and prints.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

# ----------------------------------------------------------------------------
# FIXED CONSTANTS - the "budget". Do not vary these between experiments, or the
# numbers stop being comparable (this is the fixed-workload discipline).
# ----------------------------------------------------------------------------
CUDA_DIR    = r"D:\local\llamacpp-cuda"
LLAMA_BENCH = os.path.join(CUDA_DIR, "llama-bench.exe")
LLAMA_PPL   = os.path.join(CUDA_DIR, "llama-perplexity.exe")
MODEL       = r"D:\local\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf"

# Representative workload, held constant across ALL experiments.
N_PROMPT = 512     # prompt tokens (prompt-processing test)
N_GEN    = 256     # generated tokens (generation test) <- objective lives here
N_DEPTH  = 2048    # tokens already in context before the test (models a loaded
                   # chat). Non-zero so KV-cache size / quant / VRAM budget matter
                   # for the 14B. Keep it FIXED across experiments.
REPS     = 5       # llama-bench repetitions -> gives mean +/- stddev

# Constraints / gate.
VRAM_BUDGET_GB = 11.0   # hard ceiling on the 12 GB card (headroom for desktop)
PPL_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "wikitext-2-raw", "wiki.test.raw")
PPL_NGL = 99            # offload for the perplexity pass (quality, not speed)
PPL_CTX = 2048          # ppl context = our working depth, so KV-quant error is exercised
PPL_CHUNKS = 20         # cap chunks for a fast gate (~40k tokens). We only need a
                        # STABLE RELATIVE ppl (quant vs f16), not the full-corpus number.

# Knob defaults = a sane incumbent baseline (full GPU offload + flash attention).
# Establish the actual baseline numbers for the current model with `--baseline`.
DEFAULTS = {
    "ngl": "99", "fa": "on", "batch": "2048", "ubatch": "512",
    "threads": "8", "poll": "50", "ctk": "f16", "ctv": "f16",
    "split_mode": "layer", "mmap": "1", "nkvo": "0",
}

# Which knobs can change model OUTPUT (and thus need a perplexity check when
# they deviate from baseline). Everything else is bit-exact for our purposes.
QUALITY_KNOBS = {"ctk", "ctv"}


def build_bench_cmd(cfg, reps=REPS):
    """Assemble the llama-bench command for one config."""
    return [
        LLAMA_BENCH,
        "-m", MODEL,
        "-p", str(N_PROMPT),
        "-n", str(N_GEN),
        "-d", str(N_DEPTH),
        "-r", str(reps),
        "-ngl", cfg["ngl"],
        "-fa", cfg["fa"],
        "-b", cfg["batch"],
        "-ub", cfg["ubatch"],
        "-t", cfg["threads"],
        "--poll", cfg["poll"],
        "-ctk", cfg["ctk"],
        "-ctv", cfg["ctv"],
        "-sm", cfg["split_mode"],
        "-mmp", cfg["mmap"],
        "-nkvo", cfg["nkvo"],
        "-o", "json",
    ]


def gpu_used_mib():
    """Single nvidia-smi sample of used VRAM in MiB, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    """Polls nvidia-smi while the benchmark runs; records the peak used MiB."""
    def __init__(self, interval=0.15):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = gpu_used_mib()
            if v is not None and v > self.peak:
                self.peak = v
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()


def run_bench(cfg):
    """Run llama-bench for cfg while sampling VRAM. Returns (results, peak_mib)."""
    cmd = build_bench_cmd(cfg)
    idle = gpu_used_mib()
    sampler = VramSampler()
    sampler.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    finally:
        sampler.stop()
        sampler.join(timeout=2)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr + "\n")
        raise SystemExit(f"llama-bench failed (exit {proc.returncode}). "
                         f"Likely an invalid knob combination - treat as a crash.")
    results = json.loads(proc.stdout)
    return results, sampler.peak, idle


def pick_rows(results):
    """Split llama-bench json rows into the pp (prompt) and tg (gen) tests."""
    pp = tg = None
    for r in results:
        if r["n_gen"] == 0 and r["n_prompt"] > 0:
            pp = r
        elif r["n_gen"] > 0:
            tg = r
    return pp, tg


def run_perplexity(cfg):
    """Run llama-perplexity for a lossy config. Returns float PPL or None."""
    if not os.path.exists(PPL_CORPUS):
        sys.stderr.write(
            f"[ppl] corpus not found at {PPL_CORPUS} - skipping quality gate. "
            f"Run setup to fetch wikitext, or set PPL_CORPUS.\n")
        return None
    cmd = [LLAMA_PPL, "-m", MODEL, "-f", PPL_CORPUS,
           "-ngl", str(PPL_NGL), "-fa", "on",
           "-c", str(PPL_CTX), "--chunks", str(PPL_CHUNKS),
           "-ctk", cfg["ctk"], "-ctv", cfg["ctv"]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    ppl = None
    for line in (proc.stdout + proc.stderr).splitlines():
        # llama-perplexity prints e.g. "Final estimate: PPL = 7.1234 +/- 0.04567"
        if "Final estimate" in line and "PPL" in line:
            try:
                ppl = float(line.split("PPL")[1].split("=")[1].split()[0])
            except Exception:
                pass
    return ppl


def config_id(cfg):
    canon = ";".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return hashlib.sha1(canon.encode()).hexdigest()[:7]


def parse_args():
    ap = argparse.ArgumentParser(description="Measure one llama.cpp knob config.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v, help=f"(default: {v})")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--desc", default="", help="short description for your log")
    ap.add_argument("--baseline", action="store_true",
                    help="force all knobs to defaults (ignores other flags)")
    ap.add_argument("--check-ppl", action="store_true",
                    help="force the perplexity pass even for bit-exact knobs")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULTS)
    if not args.baseline:
        for k in DEFAULTS:
            cfg[k] = getattr(args, k)

    cid = config_id(cfg)
    knob_str = " ".join(f"{k}={cfg[k]}" for k in
                        ["ngl", "fa", "batch", "ubatch", "threads",
                         "poll", "ctk", "ctv", "split_mode", "mmap", "nkvo"])

    print(f"[bench] config {cid}: {knob_str}", file=sys.stderr)
    results, peak_mib, idle_mib = run_bench(cfg)
    pp, tg = pick_rows(results)

    # Quality gate: only when a lossy knob deviates from baseline (or forced).
    lossy = any(cfg[k] != DEFAULTS[k] for k in QUALITY_KNOBS)
    ppl = None
    ppl_note = "skipped (bit-exact knobs)"
    if lossy or args.check_ppl:
        print("[bench] lossy knob changed -> running perplexity gate...",
              file=sys.stderr)
        ppl = run_perplexity(cfg)
        ppl_note = f"{ppl:.4f}" if ppl is not None else "unavailable (no corpus)"

    peak_gb = peak_mib / 1024.0 if peak_mib else 0.0
    delta_mib = (peak_mib - idle_mib) if (peak_mib and idle_mib) else None
    within = peak_gb <= VRAM_BUDGET_GB

    # Summary block (mirrors autoresearch's printed metric block).
    print("---")
    print(f"config_id:        {cid}")
    print(f"config:           {knob_str}")
    print(f"gen_tok_s:        {tg['avg_ts']:.2f} +/- {tg['stddev_ts']:.2f}")
    print(f"prompt_tok_s:     {pp['avg_ts']:.2f} +/- {pp['stddev_ts']:.2f}")
    print(f"peak_vram_mb:     {peak_mib}"
          + (f"  (delta {delta_mib} over idle)" if delta_mib is not None else ""))
    print(f"peak_vram_gb:     {peak_gb:.2f}")
    print(f"within_budget:    {'yes' if within else 'NO - over %.1f GB' % VRAM_BUDGET_GB}")
    print(f"perplexity:       {ppl_note}")
    print(f"workload:         p={N_PROMPT} n={N_GEN} d={N_DEPTH} reps={args.reps}")
    print("---")

    # Ready-to-paste TSV row for results.tsv. The LOOP sets status (keep/discard).
    ppl_col = f"{ppl:.4f}" if ppl is not None else "-"
    desc = args.desc or "(describe this experiment)"
    print(f"tsv_row:\t{cid}\t{tg['avg_ts']:.2f}\t{pp['avg_ts']:.2f}"
          f"\t{peak_gb:.1f}\t{ppl_col}\t<keep|discard>\t{desc}")


if __name__ == "__main__":
    main()
