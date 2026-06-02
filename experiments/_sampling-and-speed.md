# Do sampling knobs affect speed?

Two regimes, both measured at locked 1800 MHz.

## Plain generation: no.
- no-spec temp 0: 30.2 tok/s
- no-spec temp 0.7: 29.2 tok/s (within noise)

Sampling (temp/top-k/top-p/min-p) runs on the logit vector *after* the forward pass —
microseconds against ~30 ms/token for a 14B. So it doesn't move tok/s. (This is also
why `llama-bench` doesn't expose samplers — it decodes greedily.) **No sampling knob
improves plain-generation speed.**

## Speculative decoding: yes — and lower temperature wins.
- spec n_max=8, temp 0:   36.8 tok/s, 30% accept   → +20% vs no-spec
- spec n_max=8, temp 0.7: 30.8 tok/s, 24% accept   → only +5%
- spec n_max=8, temp 0.7 + min_p 0.1: 31.5, 24% accept

Higher temperature flattens the target distribution, so the draft's confident guesses
get rejected more often → acceptance falls → the speedup collapses. A `min-p` floor
sharpens the distribution and recovers a little, but not to temp-0 levels.

**Takeaway:** the +20% spec-decode win is contingent on greedy / low-temperature
decoding (extraction, classification, structured output). At creative temperatures
(~0.7) speculation is barely worth it. Sampling doesn't make decoding faster — but it
*governs how much spec decoding can help.* Note: acceptance is also prompt-dependent
(code/predictable text accepts more than free prose at any temp).
