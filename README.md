# llamacpp-autotune

> Autonomously tune llama.cpp runtime knobs for the fastest local inference on *your* GPU.

An [autoresearch](https://github.com/karpathy/autoresearch)-style optimizer, but the
thing being optimized isn't training code — it's the **runtime flags** of
[llama.cpp](https://github.com/ggml-org/llama.cpp). An agent picks a knob config,
measures it, and keeps or discards — looping until it converges on the best settings
for the exact model + hardware in front of it.

The numbers from one person's machine don't transfer to yours (different GPU, driver,
thermals). So rather than copy someone's "best flags," this *finds* yours empirically.

## How it works

| file | role | edited by |
|---|---|---|
| `bench.py` | fixed measurement harness — wraps `llama-bench`, reports tok/s ± stddev, peak VRAM, and a perplexity gate | nobody (it's the ground truth) |
| `program.md` | the optimizer loop + the keep/discard rule | human |
| `search_space.md` | the knobs, allowed values, and tiers | human |
| `results.tsv` | the experiment log | the loop |
| `experiments/` | per-experiment notes (what & why, including reverts) | the loop |

The metric is **generation throughput** (tok/s), under a VRAM ceiling and a perplexity
quality gate for lossy knobs (KV-cache quantization). The core discipline: `llama-bench`
repeats every test 5×, and a config is only "better" if it beats the incumbent by **more
than the combined stddev** — single-shot wins are noise.

## Requirements

- A CUDA-enabled llama.cpp build (this repo points at `D:\local\llamacpp-cuda`).
- A local GGUF model (default: `Llama-3.2-3B-Instruct-Q4_K_M`).
- Python 3.10+ and [uv](https://docs.astral.sh/uv/). `bench.py` is pure stdlib.

## Quick start (PowerShell)

```powershell
# measure the baseline config
uv run bench.py --baseline

# try one knob
uv run bench.py --fa off --desc "flash attn off"
```

Then point an agent at `program.md` and let it run the loop.

## Status

v1: speed + VRAM + KV-quant quality gate via `llama-bench` / `llama-perplexity`.
Planned: a `llama-server` harness for speculative decoding and real-request latency,
and a model/quant comparison axis.

## License

MIT
