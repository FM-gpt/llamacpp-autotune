#!/usr/bin/env python
"""
bench.py -- fixed measurement harness for llamacpp-autotune.

This is the read-only "harness" (analogous to autoresearch's prepare.py). It does
NOT decide what to try -- it just measures ONE knob config faithfully and prints a
clean summary. The optimizer loop (you, driven by program.md) chooses configs,
reads these numbers, and decides keep/discard.

What it measures for a given config:
  - generation throughput  (tg, tok/s)  mean +/- stddev   <- the primary metric
  - prompt throughput      (pp, tok/s)  mean +/- stddev   <- reported secondary
  - peak VRAM (MiB)        sampled from nvidia-smi during the run
  - avg SM clock (MHz)     throttle watchdog: if it drifts run-to-run, tok/s
                           numbers are not comparable (power/thermal capping)
  - perplexity             ONLY when a lossy knob changes (KV-quant etc.); else skipped

It wraps `llama-bench -o json` (which already repeats -r times and reports
mean/stddev) plus a VRAM+clock sampler thread and an optional llama-perplexity gate.

Usage:
  python bench.py                          # baseline (all defaults)
  python bench.py --fa off                 # one knob changed
  python bench.py --ctk q8_0 --ctv q8_0   # triggers the perplexity gate
  python bench.py --batch 512 --desc "small batch"

Configuration: edit config.toml or set env vars (LLAMA_BIN_DIR, LLAMA_MODEL, etc.).
Run `python setup.py` first to verify your setup.

Nothing here writes to results.tsv -- logging keep/discard is the loop's job.
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

import autotune_config as C

# -- knob defaults (sane incumbent: full GPU offload + flash attention) --------
DEFAULTS = {
    "ngl":        "99",
    "fa":         "on",
    "batch":      "2048",
    "ubatch":     "512",
    "threads":    "8",
    "poll":       "50",
    "ctk":        "f16",
    "ctv":        "f16",
    "split_mode": "layer",
    "mmap":       "1",
    "nkvo":       "0",
}

# Knobs that can affect model output -- these trigger the perplexity quality gate.
QUALITY_KNOBS = {"ctk", "ctv"}


# -- GPU sampling --------------------------------------------------------------

def gpu_sample():
    """Return (used_mib, sm_clock_mhz) or (None, None) if nvidia-smi unavailable."""
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,clocks.sm",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10)
        a, b = out.strip().splitlines()[0].split(",")
        return int(a), int(b)
    except Exception:
        return None, None


def gpu_used_mib():
    return gpu_sample()[0]


class VramSampler(threading.Thread):
    """Polls nvidia-smi in the background; records peak VRAM and mean SM clock.

    The mean SM clock is the throttle watchdog. On power-capped consumer GPUs
    (e.g. RTX 3060 at 170W), heavy configs can drop below the locked-clock ceiling
    mid-run. If the mean clock differs significantly between configs, the tok/s
    numbers are not on the same basis and cannot be directly compared.

    Mitigation: lock clocks with `nvidia-smi -lgc <MHz>` (needs admin/root) before
    running a sweep. The clock column makes every run self-auditing.
    """
    def __init__(self, interval=0.15):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.clocks = []
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            v, c = gpu_sample()
            if v is not None and v > self.peak:
                self.peak = v
            if c is not None:
                self.clocks.append(c)
            time.sleep(self.interval)

    @property
    def mean_clock(self):
        return round(sum(self.clocks) / len(self.clocks)) if self.clocks else None

    def stop(self):
        self._stop_event.set()


# -- benchmark runner ----------------------------------------------------------

def build_bench_cmd(cfg, reps):
    return [
        C.LLAMA_BENCH,
        "-m",     C.MODEL,
        "-p",     str(C.N_PROMPT),
        "-n",     str(C.N_GEN),
        "-d",     str(C.N_DEPTH),
        "-r",     str(reps),
        "-ngl",   cfg["ngl"],
        "-fa",    cfg["fa"],
        "-b",     cfg["batch"],
        "-ub",    cfg["ubatch"],
        "-t",     cfg["threads"],
        "--poll", cfg["poll"],
        "-ctk",   cfg["ctk"],
        "-ctv",   cfg["ctv"],
        "-sm",    cfg["split_mode"],
        "-mmp",   cfg["mmap"],
        "-nkvo",  cfg["nkvo"],
        "-o",     "json",
    ]


def run_bench(cfg, reps):
    cmd = build_bench_cmd(cfg, reps)
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
        raise SystemExit(
            f"llama-bench failed (exit {proc.returncode}). "
            "Likely an invalid knob combination -- treat as a crash.")
    results = json.loads(proc.stdout)
    return results, sampler.peak, idle, sampler.mean_clock


def pick_rows(results):
    """Split llama-bench JSON output into the pp (prompt) and tg (gen) rows."""
    pp = tg = None
    for r in results:
        if r["n_gen"] == 0 and r["n_prompt"] > 0:
            pp = r
        elif r["n_gen"] > 0:
            tg = r
    return pp, tg


# -- perplexity quality gate --------------------------------------------------─

def run_perplexity(cfg):
    """Run llama-perplexity for a lossy config. Returns float PPL or None."""
    if not os.path.exists(C.PPL_CORPUS):
        sys.stderr.write(
            f"[ppl] corpus not found at {C.PPL_CORPUS}\n"
            f"      Run `python setup.py` to fetch wikitext-2, or set the path.\n")
        return None
    cmd = [C.LLAMA_PPL, "-m", C.MODEL, "-f", C.PPL_CORPUS,
           "-ngl", "99", "-fa", "on",
           "-c", str(C.PPL_CTX), "--chunks", str(C.PPL_CHUNKS),
           "-ctk", cfg["ctk"], "-ctv", cfg["ctv"]]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    ppl = None
    for line in (proc.stdout + proc.stderr).splitlines():
        if "Final estimate" in line and "PPL" in line:
            try:
                ppl = float(line.split("PPL")[1].split("=")[1].split()[0])
            except Exception:
                pass
    return ppl


# -- config ID (stable hash of sorted knob=value pairs) ----------------------─

def config_id(cfg):
    canon = ";".join(f"{k}={cfg[k]}" for k in sorted(cfg))
    return hashlib.sha1(canon.encode()).hexdigest()[:7]


# -- CLI ----------------------------------------------------------------------─

def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure one llama.cpp knob config and print a summary block.")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", default=v, metavar=v,
                        help=f"(default: {v})")
    ap.add_argument("--reps",       type=int, default=C.REPS,
                    help=f"benchmark repetitions (default: {C.REPS})")
    ap.add_argument("--desc",       default="",
                    help="short description for your results.tsv entry")
    ap.add_argument("--baseline",   action="store_true",
                    help="ignore other flags -- measure the default incumbent config")
    ap.add_argument("--check-ppl",  action="store_true",
                    help="force the perplexity pass even for bit-exact knobs")
    return ap.parse_args()


def main():
    errors = C.validate()
    if errors:
        print("[bench] configuration errors -- run `python setup.py` to fix:\n"
              + "\n".join(errors), file=sys.stderr)
        sys.exit(1)

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
    results, peak_mib, idle_mib, sm_clock = run_bench(cfg, args.reps)
    pp, tg = pick_rows(results)

    lossy = any(cfg[k] != DEFAULTS[k] for k in QUALITY_KNOBS)
    ppl = None
    ppl_note = "skipped (bit-exact knobs)"
    if lossy or args.check_ppl:
        print("[bench] lossy knob changed -> running perplexity gate...", file=sys.stderr)
        ppl = run_perplexity(cfg)
        if ppl is not None:
            delta_pct = (ppl / _baseline_ppl() - 1) * 100 if _baseline_ppl() else None
            gate = (f"PASS (+{delta_pct:.2f}%)" if delta_pct is not None and
                    delta_pct <= C.PPL_THRESHOLD else
                    f"FAIL (+{delta_pct:.2f}% > {C.PPL_THRESHOLD}% threshold)"
                    if delta_pct is not None else "no baseline ppl to compare")
            ppl_note = f"{ppl:.4f}  [{gate}]"
        else:
            ppl_note = "unavailable (corpus missing -- run setup.py)"

    peak_gb = peak_mib / 1024.0 if peak_mib else 0.0
    delta_mib = (peak_mib - idle_mib) if (peak_mib and idle_mib) else None
    within = peak_gb <= C.VRAM_BUDGET_GB

    print("---")
    print(f"config_id:        {cid}")
    print(f"config:           {knob_str}")
    print(f"gen_tok_s:        {tg['avg_ts']:.2f} +/- {tg['stddev_ts']:.2f}")
    print(f"prompt_tok_s:     {pp['avg_ts']:.2f} +/- {pp['stddev_ts']:.2f}")
    print(f"peak_vram_mb:     {peak_mib}"
          + (f"  (delta {delta_mib} over idle)" if delta_mib is not None else ""))
    print(f"peak_vram_gb:     {peak_gb:.2f}")
    print(f"within_budget:    {'yes' if within else 'NO -- over %.1f GB' % C.VRAM_BUDGET_GB}")
    print(f"avg_sm_clock_mhz: {sm_clock}  "
          "(stable = comparable; drifting = power-throttled, numbers unreliable)")
    print(f"perplexity:       {ppl_note}")
    print(f"workload:         p={C.N_PROMPT} n={C.N_GEN} d={C.N_DEPTH} reps={args.reps}")
    print("---")

    ppl_col = f"{ppl:.4f}" if ppl is not None else "-"
    desc = args.desc or "(describe this experiment)"
    print(f"tsv_row:\t{cid}\t{tg['avg_ts']:.2f}\t{pp['avg_ts']:.2f}"
          f"\t{peak_gb:.1f}\t{sm_clock}\t{ppl_col}\t<keep|discard>\t{desc}")


def _baseline_ppl():
    """Read the f16 reference ppl from program.md if it was recorded there."""
    import re
    md = os.path.join(os.path.dirname(__file__), "program.md")
    try:
        with open(md) as f:
            m = re.search(r"f16 reference ppl\s*=\s*([0-9.]+)", f.read())
            return float(m.group(1)) if m else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
