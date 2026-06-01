# search_space.md — the tunable knobs

The knobs `bench.py` exposes, their allowed values, and how to think about each.
**Tier 1** is bit-exact (speed/VRAM only). **Tier 2** can change output → the
perplexity gate fires automatically. **Tier 3** is out of `llama-bench` scope and
needs the Phase-2 server harness.

Baseline incumbent: `ngl=99 fa=on batch=2048 ubatch=512 threads=8 poll=50
ctk=f16 ctv=f16 split_mode=layer mmap=1 nkvo=0`.

## Tier 1 — pure speed, bit-exact (no quality gate)

| knob | flag | values to try | what it does / intuition |
|---|---|---|---|
| `ngl` | `-ngl` | 99 (all), 28, 20, 0 | GPU offload layers. 99 = whole 3B on the 3060 → should win big. Lower only to study the CPU-offload cliff. |
| `fa` | `-fa` | on, off, auto | Flash attention. `on` is the expected winner here. |
| `batch` | `-b` | 2048, 1024, 512, 4096 | Logical batch (prompt chunk). Mostly affects **pp**, little effect on **tg**. |
| `ubatch` | `-ub` | 512, 256, 128, 1024 | Physical micro-batch fed to the GPU per pass. Bigger can raise pp until it saturates; watch VRAM. |
| `threads` | `-t` | 8, 6, 4, 16 | CPU threads. With everything on GPU, tg is nearly thread-insensitive; matters more if layers spill to CPU. Ryzen 7 5800 = 8 cores / 16 threads. |
| `poll` | `--poll` | 50, 0, 100 | CPU polling vs yielding while waiting on the GPU. Small, machine-specific effect. |
| `split_mode` | `-sm` | layer, none, row | Multi-GPU split strategy. Single GPU here → expect ~no effect; `none` is the honest setting. |
| `mmap` | `-mmp` | 1, 0 | Memory-map weights. Affects load/RAM more than steady-state tg. |
| `nkvo` | `-nkvo` | 0, 1 | Keep KV cache off the GPU. Usually **hurts** tg (PCIe round-trips) — included to demonstrate that. |

**Suggested order:** confirm the big levers first (`fa`, `ngl`), then `ubatch`/`batch`
for prompt throughput, then the small/skeptical knobs (`poll`, `threads`, `nkvo`, `sm`).

## Tier 2 — memory ↔ quality trade-offs (perplexity-gated)

| knob | flag | values | what it does |
|---|---|---|---|
| `ctk` | `-ctk` | f16, q8_0, q4_0 | KV-cache **key** quantization. q8_0 ≈ halves KV VRAM, tiny quality cost. q4_0 saves more but watch ppl. |
| `ctv` | `-ctv` | f16, q8_0, q4_0 | KV-cache **value** quantization. Generally pair with `-fa on`. |

The 3B at d=0 barely uses KV, so the VRAM win is small *here*; the payoff shows at large
context. Worth a couple of runs to confirm the gate works and the quality stays in budget.

## Tier 3 — later (needs the server harness, not llama-bench)

- **Speculative decoding** (`--model-draft`, `--draft-max`): a tiny draft model proposes
  tokens; only measurable via `llama-server` real-request timings. Phase 2.
- **Real-request latency / TTFT**, parallel slots, continuous batching: also server-side.

## Knobs deliberately fixed (don't tune in v1)

- Workload (`-p/-n/-d`), model, and quant are constants — changing them breaks
  comparability. Model/quant comparison (Q4 vs Q6 vs Q8) is a separate planned axis.
