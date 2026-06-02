#!/usr/bin/env python
"""
bench_server.py -- speculative-decoding harness for llamacpp-autotune (Phase 2).

llama-bench cannot measure speculative decoding. This script drives `llama-server`
with a target + optional draft model and reads real-request generation throughput
from the server's response timings. It also reports draft-token acceptance rate and
peak VRAM including both models.

Spec decoding is lossless at temperature 0 (the target verifies every drafted
token), so the output is identical with or without a draft -- only the speed changes.
That makes this a clean apples-to-apples throughput comparison.

Tunable knobs (draft side):
  --draft <path|none>   draft model path, or "none" for no speculation
  --n-max N             max tokens drafted per step   (default: 3)
  --p-min P             min draft token probability   (default: 0.0)
  --ngld N              draft GPU layers              (default: 99 = full offload)

Sampling knobs (affects spec acceptance rate -- see experiments/_sampling-and-speed.md):
  --temp, --top-k, --top-p, --min-p

Usage:
  python bench_server.py --draft none          # baseline, no spec
  python bench_server.py --n-max 8             # tune draft length
  python bench_server.py --n-max 8 --temp 0.7  # check acceptance at higher temp

Configuration: edit config.toml or set LLAMA_BIN_DIR, LLAMA_MODEL, LLAMA_DRAFT_MODEL.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

import autotune_config as C
from bench import VramSampler, gpu_used_mib, config_id

# Fixed prompt: representative technical prose. Moderate predictability gives an
# honest middle-of-the-road acceptance rate -- not best-case (code) nor worst-case
# (random text). Keep it fixed across experiments so acceptance rates are comparable.
PROMPT = (
    "Explain, step by step and in detail, how a modern CPU executes a single "
    "machine instruction, from fetch to retire. Cover pipelining, the role of "
    "registers and cache, and what can stall the pipeline."
)

LOGFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")


def base_url():
    return f"http://127.0.0.1:{C.SERVER_PORT}"


def wait_health(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url() + "/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def post_completion(samplers):
    """One generation request. Returns (timings_dict, full_response)."""
    payload = {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": C.SERVER_MAXTOK,
        "seed": 0,
    }
    payload.update(samplers)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url() + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp.get("timings", {}), resp


def build_server_cmd(cfg):
    cmd = [C.LLAMA_SERVER, "-m", C.MODEL,
           "-ngl", "99", "-fa", "on",
           "-c", str(C.SERVER_CTX),
           "--host", "127.0.0.1", "--port", str(C.SERVER_PORT)]
    if cfg["draft"] != "none":
        cmd += ["-md",                  cfg["draft"],
                "-ngld",                cfg["ngld"],
                "--spec-draft-n-max",   cfg["n_max"],
                "--spec-draft-p-min",   cfg["p_min"]]
    return cmd


def acceptance(timings):
    """Extract draft acceptance rate from server timings, if reported."""
    drafted  = timings.get("draft_n")         or timings.get("n_draft")
    accepted = (timings.get("draft_n_accepted")
                or timings.get("n_draft_accepted")
                or timings.get("n_accepted"))
    if drafted and accepted is not None:
        return accepted / drafted, drafted, accepted
    return None, None, None


def parse_args():
    ap = argparse.ArgumentParser(
        description="Measure one speculative-decoding config via llama-server.")
    ap.add_argument("--draft",  default=C.DRAFT_MODEL,
                    help="draft model path, or 'none' for no-spec baseline")
    ap.add_argument("--n-max",  dest="n_max",  default="3",
                    help="max tokens drafted per step (default: 3)")
    ap.add_argument("--p-min",  dest="p_min",  default="0.0",
                    help="min draft token probability (default: 0.0)")
    ap.add_argument("--ngld",   default="99",
                    help="draft model GPU layers (default: 99 = full offload)")
    ap.add_argument("--reps",   type=int,      default=C.REPS)
    ap.add_argument("--temp",   default="0",
                    help="sampling temperature (default: 0 = greedy, lossless spec)")
    ap.add_argument("--top-k",  dest="top_k",  default=None)
    ap.add_argument("--top-p",  dest="top_p",  default=None)
    ap.add_argument("--min-p",  dest="min_p",  default=None)
    ap.add_argument("--desc",   default="")
    return ap.parse_args()


def main():
    errors = C.validate(require_draft=False)
    if errors:
        print("[bench_server] configuration errors -- run `python setup.py`:\n"
              + "\n".join(errors), file=sys.stderr)
        sys.exit(1)

    args = parse_args()
    cfg  = {"draft": args.draft or "none",
            "n_max": args.n_max, "p_min": args.p_min, "ngld": args.ngld}
    spec = cfg["draft"] != "none"

    samplers = {"temperature": float(args.temp)}
    if args.top_k is not None: samplers["top_k"] = int(args.top_k)
    if args.top_p is not None: samplers["top_p"] = float(args.top_p)
    if args.min_p is not None: samplers["min_p"] = float(args.min_p)

    smp  = "temp=%s" % args.temp + "".join(
        f" {k}={getattr(args, k)}" for k in ("top_k", "top_p", "min_p")
        if getattr(args, k) is not None)
    base = (f"spec n_max={cfg['n_max']} p_min={cfg['p_min']} "
            f"draft={os.path.basename(cfg['draft'])}" if spec else "no-spec baseline")
    label = f"{base} [{smp}]"
    cid   = config_id(cfg)

    print(f"[server] launching: {label}", file=sys.stderr)
    idle = gpu_used_mib()
    logf = open(LOGFILE, "w")
    proc = subprocess.Popen(build_server_cmd(cfg), stdout=logf, stderr=logf)
    sampler = VramSampler()
    try:
        if not wait_health():
            proc.terminate()
            raise SystemExit(
                "Server did not become healthy -- check server.log.\n"
                "Common causes: bad flag name for this build, or draft/target "
                "vocabulary mismatch.")
        post_completion(samplers)  # warmup -- discard
        sampler.start()
        tgs, acc_rate = [], None
        for _ in range(args.reps):
            timings, _ = post_completion(samplers)
            tgs.append(timings.get("predicted_per_second", 0.0))
            r, _, _ = acceptance(timings)
            if r is not None:
                acc_rate = r
    finally:
        sampler.stop()
        sampler.join(timeout=2)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()

    tg_mean = statistics.mean(tgs)
    tg_sd   = statistics.stdev(tgs) if len(tgs) > 1 else 0.0
    peak_mib = sampler.peak
    peak_gb  = peak_mib / 1024.0 if peak_mib else 0.0
    delta    = (peak_mib - idle) if (peak_mib and idle) else None

    print("---")
    print(f"config_id:        {cid}")
    print(f"config:           {label}")
    print(f"gen_tok_s:        {tg_mean:.2f} +/- {tg_sd:.2f}")
    print(f"draft_accept:     "
          + (f"{acc_rate*100:.1f}%" if acc_rate is not None else "n/a (not reported by server)"))
    print(f"peak_vram_mb:     {peak_mib}"
          + (f"  (delta {delta} over idle)" if delta is not None else ""))
    print(f"peak_vram_gb:     {peak_gb:.2f}")
    print(f"within_budget:    "
          + ("yes" if peak_gb <= C.VRAM_BUDGET_GB
             else f"NO -- over {C.VRAM_BUDGET_GB:.1f} GB"))
    print(f"avg_sm_clock_mhz: {sampler.mean_clock}  "
          "(stable = comparable; drifting = power-throttled, numbers unreliable)")
    print(f"workload:         server /v1 max_tokens={C.SERVER_MAXTOK} "
          f"ctx={C.SERVER_CTX} reps={args.reps}")
    print("---")
    desc = args.desc or label
    print(f"tsv_row:\t{cid}\t{tg_mean:.2f}\t-\t{peak_gb:.1f}"
          f"\t{sampler.mean_clock}\t-\t<keep|discard>\t{desc}")


if __name__ == "__main__":
    main()
