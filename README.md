# llamacpp-autotune

> Autonomously tune llama.cpp runtime knobs for the fastest local inference on *your* GPU.

An [autoresearch](https://github.com/karpathy/autoresearch)-style optimizer, but instead of editing training code to minimize a loss metric, it tunes **llama.cpp runtime flags** to maximize **inference throughput** on the exact hardware in front of it.

The key insight: "best flags" from someone else's benchmark don't transfer to your machine (different GPU, VRAM, driver, thermals). This finds *yours* — empirically, with proper noise accounting.

---

## How it works

The loop is simple:

```
pick one knob  →  measure it (bench.py)  →  keep if gain > noise  →  log  →  repeat
```

Three files:

| file | role | who edits it |
|---|---|---|
| `bench.py` | fixed measurement harness (like `prepare.py` in autoresearch) | nobody |
| `bench_server.py` | speculative-decoding harness via `llama-server` | nobody |
| `program.md` | the loop rules + keep/discard discipline | you |
| `search_space.md` | the knobs, allowed values, tiers | you |
| `config.toml` | your paths and workload settings | you |

Point an AI agent at `program.md` and let it run overnight.

---

## The noise problem (and why this project exists)

Single benchmarks lie. On a consumer GPU (RTX 3060, 4080, etc.):
- The **power cap** clocks the GPU down under heavy loads — variable amounts, run to run.
- **Between-run drift** of 1–2 tok/s can mask or invent results.

`bench.py` fixes this three ways:
1. **5 repetitions per test** → `llama-bench` reports mean ± stddev.
2. **SM-clock monitoring** → every result includes `avg_sm_clock_mhz`; if it drifts between configs, the numbers aren't on the same basis. The harness prints a warning.
3. **The keep rule**: a config is only "better" if its mean beats the incumbent by **more than the combined noise floor**: `Δ > √(σ_new² + σ_old²)`. Ties are discarded.

This rule is the whole point. Without it, you're just chasing luck.

---

## Quick start

**1. Configure**

```toml
# config.toml
[paths]
llama_bin_dir = "/path/to/llama.cpp/build/bin"   # or C:/llama.cpp/build/bin
model         = "/path/to/your-model-Q4_K_M.gguf"
draft_model   = "/path/to/small-model-Q8_0.gguf" # for bench_server.py only
```

Or use environment variables:
```bash
export LLAMA_BIN_DIR=/path/to/llama.cpp/build/bin
export LLAMA_MODEL=/path/to/model.gguf
```

**2. Verify setup**

```bash
python setup.py
```

This checks your binaries, model file, and downloads the wikitext-2 perplexity corpus.

**3. Establish baseline**

```bash
python bench.py --baseline
```

Record the output in `results.tsv`. This is your incumbent — all future experiments compare against it.

**4. Try a knob**

```bash
python bench.py --fa off --desc "flash attention off"
```

Apply the keep rule: if `gen_tok_s` beats the incumbent by more than `√(σ_new² + σ_inc²)`, it's a real improvement. Otherwise discard.

**5. Speculative decoding (Phase 2)**

```bash
python bench_server.py --draft none          # server baseline
python bench_server.py --n-max 8             # spec decoding on
```

See `search_space.md` for recommended knob order and expected effect sizes.

---

## What we found (RTX 3060 12GB, Qwen2.5-14B-Instruct Q4_K_M)

A full sweep is in `results.tsv` and `experiments/`. The short version:

**Phase 1 — llama-bench knobs:** every single one was a tie or regression. A fully-offloaded 14B is memory-bandwidth bound — you cannot flag your way past physics. The defaults (`-ngl 99 -fa on`) are the right answer.

**Phase 2 — speculative decoding:** the only lever that beats the baseline.

| config | tok/s | vs baseline |
|---|---|---|
| baseline (no spec) | 30.2 | — |
| spec, n_max=3 | 32.2 | +7% |
| **spec, n_max=8** | **36.8** | **+22%** |
| spec, n_max=16 | 23.4 | −22% |

Draft model: Qwen2.5-0.5B-Instruct Q8_0. Lossless at temperature 0.

**Key findings from the sweep:**
- `n_max=8` is the sweet spot. `n_max=16` over-drafts and becomes *slower than no spec*.
- `p_min=0.0` is best — higher values chase acceptance rate at the cost of tokens-per-step.
- Q8 draft beats Q4 draft — at 0.5B, prediction accuracy matters more than draft speed.
- **Spec-decode gain is temperature-dependent**: full +22% at temp 0, shrinks to ~+5% at temp 0.7. If your use case requires creative temperatures, speculation is barely worth it.

**The GPU throttle trap:** on a power-capped 3060, back-to-back spec runs caused variable clock drops (1752–1799 MHz at a nominal 1800 lock), producing a non-physical "valley" in the n_max curve. Fixed by adding the SM-clock watchdog and a cooldown between runs. See `experiments/d2c5e86.md` for the full story.

---

## Hardware

Tested on: RTX 3060 12GB / Ryzen 7 5800 / Windows 11, llama.cpp build b9437.

Should work on any NVIDIA GPU with `nvidia-smi` available. The SM-clock sampler degrades gracefully to `None` on AMD/Apple Silicon (VRAM sampling still works if your platform has a compatible query tool — contributions welcome).

---

## Project structure

```
config.toml          your paths and workload settings (edit this)
bench.py             llama-bench harness (do not modify)
bench_server.py      llama-server / spec-decoding harness (do not modify)
setup.py             one-time setup + verifier
program.md           loop rules and keep/discard discipline (edit to steer)
search_space.md      knob taxonomy and expected effects (edit to steer)
results.tsv          experiment log
experiments/         per-experiment notes (reasoning, revert rationale)
data/                perplexity corpus (auto-downloaded by setup.py)
```

---

## License

MIT
