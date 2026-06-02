# Speculative decoding — tuning summary (Phase 2)

**Optimum found:** draft = Qwen2.5-0.5B-Instruct **Q8_0**, **`--spec-draft-n-max 8`**,
**`--spec-draft-p-min 0.0`**, draft fully on GPU. → **~36–37 tok/s vs 30 baseline,
about +20%** (locked clock, lossless at temp 0). Peak VRAM ~11.2 GB (over the 11 GB
self-budget, fine on 12 GB).

## What each knob taught us (all at locked 1800 MHz)

- **n_max (draft length):** peak at **8** (36.8), close second at 10 (35.4). Falls off
  hard past 12, and **n_max=16 is slower than no-spec at all (−22%)** — over-drafting
  wastes target verify compute on rejected tokens. The sweet spot is "draft ~8, verify
  in one batch."
- **p_min (draft confidence floor):** **0.0 is best.** Raising it trades throughput for
  acceptance — p_min=0.9 hits 97% acceptance but tg collapses to ~baseline because the
  draft barely speculates. We want max *tokens accepted per step*, not max accept rate.
- **draft quant:** **Q8 > Q4.** Q4 draft (32.0) loses to Q8 (35.7) — at 0.5B the draft
  is already cheap, so its *accuracy* (30% vs 27% acceptance) matters more than its speed.

## Caveat on the fine curve

Even with `-lgc 1800`, compute-heavy configs (low n_max) still dipped below the locked
clock (1752–1778 MHz) because the **170 W power cap overrides the clock lock**. The
robust conclusions above hold (they're at full 1799–1800 MHz); a pristine fine curve
would need locking at ~1500 MHz where power never caps. The `avg_sm_clock_mhz` readout
is what makes every run self-auditing.

## Bottom line

Spec decoding is the **only** lever that meaningfully beat the baseline — the plain
llama-bench knobs were all ties or regressions because a fully-offloaded 14B is pinned
by memory bandwidth. ~+20% generation speed, identical output, for one extra 0.5 GB
draft model.
