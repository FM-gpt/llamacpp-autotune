# program.md — the autotune loop

This is an autoresearch-style autonomous optimizer. Instead of editing training
code to lower `val_bpb`, you tune **llama.cpp runtime knobs** to raise inference
**throughput** on this machine. You are the optimizer: you pick a knob config,
measure it with `bench.py`, and keep or discard based on the result.

The human edits this file and `search_space.md`. You do not edit `bench.py` or
`bench_server.py` — those are the ground-truth harness. The workload and paths
are set in `config.toml`.

## Objective

**Maximize generation throughput `gen_tok_s` (tg)** on the fixed workload, subject to:
- **Hard constraint:** `peak_vram_gb` ≤ `vram_budget_gb` from `config.toml`
  (printed as `within_budget` in every result).
- **Quality gate:** when a lossy knob changes (`ctk`/`ctv` ≠ f16), `bench.py` runs
  perplexity automatically. Keep the change only if perplexity is within **+1%**
  of the f16 baseline. Pure-speed knobs are bit-exact — no ppl check needed.
  **Record the f16 reference ppl here after running `--baseline --check-ppl`:**
  `f16 reference ppl = _______ → gate threshold = _______`

`prompt_tok_s` (pp) is always reported as a secondary signal.

## The harness

One command measures one config:

```bash
python bench.py [--knob value ...] --desc "what I changed"
# examples:
python bench.py --baseline
python bench.py --fa off --desc "flash attention off"
python bench.py --ctk q8_0 --ctv q8_0 --desc "kv cache q8"
```

The output ends in a `tsv_row:` line ready to paste into `results.tsv`.
`bench.py` does NOT write to `results.tsv` — you do that after deciding keep/discard.

See `search_space.md` for the knobs, allowed values, and recommended order.

## The keep rule (do not skip it)

Single benchmarks are noisy. `bench.py` repeats each test `reps` times (default 5)
and reports `mean +/- stddev`. A new config is only **better** if its `gen_tok_s`
mean beats the incumbent's by **more than the combined noise floor**:

```
improvement_is_real  ⟺  (tg_new − tg_inc)  >  sqrt(σ_new² + σ_inc²)
```

If the gain is inside that band → **tie → discard**. Prefer the simpler/incumbent config.
This rule is the whole point. Never keep on one lucky number.

Also discard if `within_budget` is NO, or if the quality gate fails (ppl over threshold).

### Clock hygiene (IMPORTANT on power-capped consumer GPUs)

Every result prints `avg_sm_clock_mhz`. If the clock differs significantly between
configs, the tok/s numbers are not on the same basis:

- **Best:** lock clocks with `nvidia-smi -lgc <MHz>` (needs admin/root) before a sweep.
  Reset with `nvidia-smi -rgc` when done.
- **Without lock:** add a short cooldown between runs; rely on the keep rule's stddev
  band to absorb residual noise; treat any "win" below ~1 tok/s with suspicion.

If you see a non-monotonic curve on a clearly monotonic parameter (e.g. n_max for
spec decoding) — stop, add cooldowns, and re-run. That is the throttle trap.

## The loop — RUN FOREVER until interrupted

Maintain an **incumbent** = the best config so far (start from `--baseline`).

LOOP:
1. Check `results.tsv` and the incumbent. Pick **one** knob to try next
   (one at a time — see `search_space.md` for order and expected effects).
2. Run `python bench.py --<knob> <value> ...other incumbent knobs... --desc "..."`.
3. Read the summary. Apply the keep rule.
4. Append the `tsv_row:` to `results.tsv` (fill in status: keep/discard/crash).
5. Write a short note to `experiments/<config_id>.md`: what changed, the numbers,
   and *why* you kept or discarded. Include the reasoning even for discards — that
   is what makes this a teaching artifact, not just a number dump.
6. If kept → new incumbent. If discarded → old incumbent unchanged.
7. Every ~10 experiments: `git add -A && git commit`.

**Crashes:** bad knob combos make `bench.py` exit non-zero. Log as `crash`, note
the reason, and move on.

**NEVER STOP to ask "should I keep going?"** The human may be away. If you run out
of single-knob ideas: combine near-misses, try a 2-knob coordinated change
(e.g. batch+ubatch), sweep a knob's full range, or re-read `search_space.md`.
The loop ends only when the human interrupts.
