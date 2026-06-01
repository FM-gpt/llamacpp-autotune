# program.md — the autotune loop

This is an autoresearch-style autonomous optimizer. Instead of editing training
code to lower `val_bpb`, you tune **llama.cpp runtime knobs** to raise inference
**throughput** on this machine (RTX 3060). You are the optimizer: you pick a knob
config, measure it with `bench.py`, and keep or discard based on the result.

The human edits this file and `search_space.md`. You do not edit `bench.py` or the
fixed constants in it — that is the ground-truth harness.

## Objective

**Maximize generation throughput `gen_tok_s` (tg)** on the fixed workload, subject to:
- **Hard constraint:** `peak_vram_gb` ≤ 11.0 (it's printed as `within_budget`).
- **Quality gate:** when a lossy knob changes (`ctk`/`ctv` ≠ f16), `bench.py` runs
  perplexity automatically. Keep the change only if `perplexity` is within **+1%**
  of the f16 baseline ppl. Pure-speed knobs are bit-exact — no ppl needed.
  **f16 reference ppl = 4.5807 → gate threshold = 4.626** (Qwen2.5-14B, ctx 2048,
  20 chunks). Re-measure the reference if the model or PPL settings change.

`prompt_tok_s` (pp) is reported too; it's a secondary signal, not the objective.

## The harness

One command measures one config:

```powershell
uv run bench.py [--knob value ...] --desc "what I changed"
# e.g.
uv run bench.py --baseline
uv run bench.py --fa off --desc "flash attn off"
uv run bench.py --ctk q8_0 --ctv q8_0 --desc "kv cache q8"
```

It prints a summary block ending in a `tsv_row:` line. `bench.py` does NOT write
to `results.tsv` — **you** do, after deciding keep/discard.

See `search_space.md` for the knobs, allowed values, and tiers.

## The keep rule (the golden rule — do not skip it)

Single benchmarks are noisy. `bench.py` already repeats each test 5× and prints
`mean +/- stddev`. A new config is only **better** if its `gen_tok_s` mean beats the
incumbent's mean by **more than the combined noise**:

```
improvement_is_real  ⟺  (tg_new − tg_inc)  >  sqrt(stddev_new² + stddev_inc²)
```

If the gain is inside that band, it's a **tie → discard** (prefer the simpler/incumbent
config). This rule is the whole point — it's the lesson that a 55%-looking single-shot
win evaporated to a tie under 5 runs. Never keep on a single lucky number.

Also discard if `within_budget` is `NO`, or (for lossy knobs) if the quality gate fails.

## The loop — RUN FOREVER until interrupted

Maintain an **incumbent** = the best config so far (start from `--baseline`).

LOOP:
1. Look at `results.tsv` and the incumbent. Pick **one** knob to change next
   (one knob at a time — see `search_space.md` for ideas and ordering).
2. Run `uv run bench.py --<knob> <value> ...incumbent's other knobs... --desc "..."`.
   - Start from the incumbent config and change exactly one knob.
3. Read the summary. Apply the **keep rule** above.
4. Append a row to `results.tsv` (tab-separated) with status `keep` or `discard`,
   using the printed `tsv_row:` values (fill in the status).
5. Write a one-paragraph note to `experiments/<config_id>.md`: what you changed,
   the numbers, and *why* you kept/discarded (include the revert reasoning — this
   is the teaching artifact and video companion material).
6. If kept, the new config becomes the incumbent. If discarded, keep the old incumbent.
7. Periodically (every ~10 experiments) `git add -A && git commit` the logs.

**Crashes:** an invalid knob combo makes `bench.py` exit non-zero. Log it as
`crash` in the tsv, write a short note, and move on.

**NEVER STOP to ask "should I keep going?"** The human may be away and expects you to
run indefinitely. If you run out of single-knob ideas: combine prior near-misses, try
a coordinated 2-knob change (e.g. batch+ubatch together), sweep a knob's full range,
or re-read `search_space.md` for an axis you haven't touched. The loop ends only when
the human interrupts.

## Benchmark hygiene

- Hold the workload fixed (it's baked into `bench.py` — `p=512 n=256 d=0`). Don't change it mid-run.
- GPU boost clocks and temperature drift over a long session add noise. If you have
  admin, lock clocks once (`nvidia-smi -lgc <freq>`); otherwise rely on the keep rule's
  stddev band to reject noise, and don't over-interpret sub-stddev wiggles.
- One knob per step. Resist the urge to change two things and guess which mattered.
