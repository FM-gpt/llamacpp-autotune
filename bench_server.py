#!/usr/bin/env python
"""
bench_server.py - speculative-decoding harness for llamacpp-autotune (Phase 2).

llama-bench can't measure speculative decoding, so this drives `llama-server` with
a target + draft model and reads real-request generation throughput from the server
timings. Spec decoding is LOSSLESS at temperature 0 (the target verifies every drafted
token), so the output is identical with or without a draft - only the speed changes.
That makes this a clean apples-to-apples speed comparison.

Tuned knobs (draft side):
  --draft <path|none>   draft model (none = plain baseline, no speculation)
  --n-max N             tokens drafted per step  (--spec-draft-n-max, default 3)
  --p-min P             min draft probability     (--spec-draft-p-min, default 0.0)
  --ngld N              draft GPU layers          (-ngld, default 99 = all on GPU)

Usage:
  python bench_server.py --draft none            # server baseline (no spec)
  python bench_server.py --n-max 8 --desc "draft 8"

Measures generation tok/s (mean +/- stddev over --reps requests) + draft acceptance
(if the server reports it) + peak VRAM (target + draft). Prints a summary block and a
ready-to-paste results.tsv row. Always stops the server it launched.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

from bench import VramSampler, gpu_used_mib, config_id  # reuse the harness bits

CUDA_DIR     = r"D:\local\llamacpp-cuda"
LLAMA_SERVER = os.path.join(CUDA_DIR, "llama-server.exe")
TARGET = r"D:\local\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
DRAFT  = r"D:\local\models\Qwen2.5-0.5B-Instruct-Q8_0.gguf"
HOST, PORT = "127.0.0.1", 8081
CTX        = 4096
REPS       = 5
MAX_TOKENS = 256
LOGFILE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

# Fixed greedy prompt. Representative technical prose - moderate predictability, so the
# acceptance rate (and thus speedup) is an honest middle-of-the-road number, not a
# best-case (code) or worst-case (random) outlier.
PROMPT = ("Explain, step by step and in detail, how a modern CPU executes a single "
          "machine instruction, from fetch to retire. Cover pipelining, the role of "
          "registers and cache, and what can stall the pipeline.")


def base_url():
    return f"http://{HOST}:{PORT}"


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


def post_completion():
    """One greedy generation; returns the server 'timings' dict."""
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(base_url() + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return resp.get("timings", {}), resp


def build_server_cmd(cfg):
    cmd = [LLAMA_SERVER, "-m", TARGET, "-ngl", "99", "-fa", "on",
           "-c", str(CTX), "--host", HOST, "--port", str(PORT)]
    if cfg["draft"] != "none":
        cmd += ["-md", cfg["draft"],
                "-ngld", cfg["ngld"],
                "--spec-draft-n-max", cfg["n_max"],
                "--spec-draft-p-min", cfg["p_min"]]
    return cmd


def acceptance(timings):
    """Extract draft acceptance rate from timings if the server reports it."""
    drafted = timings.get("draft_n") or timings.get("n_draft")
    accepted = (timings.get("draft_n_accepted") or timings.get("n_draft_accepted")
                or timings.get("n_accepted"))
    if drafted and accepted is not None:
        return accepted / drafted, drafted, accepted
    return None, None, None


def parse_args():
    ap = argparse.ArgumentParser(description="Measure one spec-decoding config.")
    ap.add_argument("--draft", default=DRAFT,
                    help="draft model path, or 'none' for the no-spec baseline")
    ap.add_argument("--n-max", dest="n_max", default="3")
    ap.add_argument("--p-min", dest="p_min", default="0.0")
    ap.add_argument("--ngld", default="99")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--desc", default="")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = {"draft": args.draft, "n_max": args.n_max,
           "p_min": args.p_min, "ngld": args.ngld}
    spec = cfg["draft"] != "none"
    label = (f"spec n_max={cfg['n_max']} p_min={cfg['p_min']} ngld={cfg['ngld']} "
             f"draft={os.path.basename(cfg['draft'])}" if spec else "no-spec baseline")
    cid = config_id(cfg)

    print(f"[server] launching: {label}", file=sys.stderr)
    idle = gpu_used_mib()
    logf = open(LOGFILE, "w")
    proc = subprocess.Popen(build_server_cmd(cfg), stdout=logf, stderr=logf)
    sampler = VramSampler()
    try:
        if not wait_health():
            proc.terminate()
            raise SystemExit("server did not become healthy - check server.log "
                             "(likely a bad flag or incompatible draft/target vocab).")
        post_completion()  # warmup (discard)
        sampler.start()
        tgs, acc_rate = [], None
        for i in range(args.reps):
            timings, _ = post_completion()
            tgs.append(timings.get("predicted_per_second", 0.0))
            r, dn, da = acceptance(timings)
            if r is not None:
                acc_rate = r
    finally:
        sampler.stop(); sampler.join(timeout=2)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()

    tg_mean = statistics.mean(tgs)
    tg_sd = statistics.stdev(tgs) if len(tgs) > 1 else 0.0
    peak_mib = sampler.peak
    peak_gb = peak_mib / 1024.0 if peak_mib else 0.0
    delta = (peak_mib - idle) if (peak_mib and idle) else None

    print("---")
    print(f"config_id:        {cid}")
    print(f"config:           {label}")
    print(f"gen_tok_s:        {tg_mean:.2f} +/- {tg_sd:.2f}")
    print(f"draft_accept:     {('%.1f%%' % (acc_rate*100)) if acc_rate is not None else 'n/a (not reported)'}")
    print(f"peak_vram_mb:     {peak_mib}"
          + (f"  (delta {delta} over idle)" if delta is not None else ""))
    print(f"peak_vram_gb:     {peak_gb:.2f}")
    print(f"within_budget:    {'yes' if peak_gb <= 11.0 else 'NO - over 11.0 GB'}")
    print(f"avg_sm_clock_mhz: {sampler.mean_clock}  (watch for drift - throttle = uncomparable)")
    print(f"workload:         server /v1, max_tokens={MAX_TOKENS}, temp=0, reps={args.reps}")
    print("---")
    desc = args.desc or label
    print(f"tsv_row:\t{cid}\t{tg_mean:.2f}\t-\t{peak_gb:.1f}\t-\t<keep|discard>\t{desc}")


if __name__ == "__main__":
    main()
