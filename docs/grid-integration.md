# GRID / Hermes integration for llamacpp-autotune

This fork is maintained as the GRID model-setup autotuning utility. Upstream is `tonbistudio/llamacpp-autotune`; GRID-specific changes live on normal Git branches and should be pushed to `FM-gpt/llamacpp-autotune`.

## Purpose

Use `llamacpp-autotune` before promoting a local GGUF model/config into the GRID controller stack. The goal is not to prove a model can make polished final product. The goal is to identify stable, repeatable llama.cpp runtime settings for grunt-work, basic design iteration, and controller-backed patch/code-sketch workflows.

## Report-first rule

Every setup, bench, or server/speculative-decoding run should go through the report wrapper:

```bash
python scripts/run_with_report.py setup
python scripts/run_with_report.py bench -- --baseline
python scripts/run_with_report.py bench -- --fa off --desc "flash attention off"
python scripts/run_with_report.py server -- --draft none
```

The wrapper writes both:

- `reports/YYYYMMDD-HHMMSS-<kind>-<args>.md`
- `reports/YYYYMMDD-HHMMSS-<kind>-<args>.json`

Reports include command, return code, safe environment variables, Git commit, GPU telemetry, stdout/stderr, and the parsed `tsv_row` when present. Commit important reports with the model/config change they justify. For very noisy sweeps, commit summaries plus representative reports.

## CT128 current baseline

As of 2026-06-11, the stable GRID controller backend is:

- CT128 / `grid-llamacpp-gpu-01`
- endpoint: `http://10.10.30.128:8081/v1`
- model id: `local-model`
- active model: Qwen2.5-Coder 14B Q4_K_M
- context: 32768
- role: stable backend for `/Users/tron/grid-local-model-bakeoff`

Do not disrupt the active 8081 service for exploratory tuning. Prefer a temporary port such as 8082/8083, or stop/restart under an explicit rollback plan.

## Example environment for CT128 runs

Run these inside CT128 or adapt paths if running via `pct exec`:

```bash
export LLAMA_BIN_DIR=/srv/llamacpp/llama.cpp/build-cuda/bin
export LLAMA_MODEL=/srv/ai-models/gguf/qwen2.5-coder-14b-q4/qwen2.5-coder-14b-instruct-q4_k_m.gguf
export LLAMA_N_PROMPT=512
export LLAMA_N_GEN=256
export LLAMA_N_DEPTH=2048
export LLAMA_REPS=5
export LLAMA_VRAM_BUDGET=19.0
python scripts/run_with_report.py setup
python scripts/run_with_report.py bench -- --baseline
```

For Qwen2.5-Coder-32B Q4_K_M exploration on the RTX A4500, prior dry-run controller testing found `--n-gpu-layers 51` at 32k context used about 18.4GB VRAM and left the rest in host RAM. Use it as a starting point, not a final answer.

## Promotion checklist

Before using a tuned setup in the GRID controller:

1. Baseline report exists and is readable.
2. Candidate report exists and includes the exact command/config.
3. Candidate beats incumbent by more than the noise floor from `program.md`.
4. VRAM stays inside the configured budget with enough service headroom.
5. Long-context/controller JSON behavior has a separate dry-run report from `/Users/tron/grid-local-model-bakeoff`.
6. CT128 stable backend rollback path is documented before changing systemd services.
7. Obsidian receives a summary entry linking the repo branch/commit and reports.

## Git maintenance

```bash
git remote -v
git fetch upstream --prune
git fetch origin --prune
git checkout grid/model-setup-reporting
git rebase upstream/master
git push origin grid/model-setup-reporting
```

Keep upstream harness files close to upstream where practical. Put GRID-specific process/docs in `docs/`, `scripts/`, and report examples rather than deeply rewriting the measurement harness.
