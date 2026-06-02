"""
autotune_config.py -- loads config.toml, applies env-var overrides, and exposes
a single `CFG` object used by bench.py and bench_server.py.

Priority: env var > config.toml > built-in default.
"""
import os
import sys
import tomllib  # stdlib since Python 3.11; falls back gracefully below 3.11
from pathlib import Path

_HERE = Path(__file__).parent
_TOML = _HERE / "config.toml"

# -- load toml ----------------------------------------------------------------
def _load_toml():
    if not _TOML.exists():
        return {}
    try:
        with open(_TOML, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"[config] warning: could not parse config.toml: {e}", file=sys.stderr)
        return {}

_t = _load_toml()

def _s(section, key, default=""):
    return _t.get(section, {}).get(key, default)

def _n(section, key, default):
    return _t.get(section, {}).get(key, default)

def _p(raw):
    """Normalise a path string to the OS-native separator so subprocesses see it correctly."""
    return str(Path(raw)) if raw else ""

# -- resolve (env > toml > default) ------------------------------------------─
LLAMA_BIN_DIR   = _p(os.environ.get("LLAMA_BIN_DIR")    or _s("paths", "llama_bin_dir"))
MODEL           = _p(os.environ.get("LLAMA_MODEL")       or _s("paths", "model"))
DRAFT_MODEL     = _p(os.environ.get("LLAMA_DRAFT_MODEL") or _s("paths", "draft_model"))

N_PROMPT        = int(os.environ.get("LLAMA_N_PROMPT", _n("workload", "n_prompt", 512)))
N_GEN           = int(os.environ.get("LLAMA_N_GEN",    _n("workload", "n_gen",    256)))
N_DEPTH         = int(os.environ.get("LLAMA_N_DEPTH",  _n("workload", "n_depth",  2048)))
REPS            = int(os.environ.get("LLAMA_REPS",     _n("workload", "reps",     5)))

VRAM_BUDGET_GB  = float(os.environ.get("LLAMA_VRAM_BUDGET", _n("constraints", "vram_budget_gb", 11.0)))

PPL_THRESHOLD   = float(_n("perplexity", "ppl_threshold_pct", 1.0))
PPL_CHUNKS      = int(_n("perplexity",   "ppl_chunks",        20))
PPL_CTX         = int(_n("perplexity",   "ppl_ctx",           2048))

SERVER_PORT     = int(_n("server", "port",       8081))
SERVER_CTX      = int(_n("server", "ctx",        4096))
SERVER_MAXTOK   = int(_n("server", "max_tokens", 256))

# -- derived paths ------------------------------------------------------------─
def _bin(name):
    if LLAMA_BIN_DIR:
        for ext in ("", ".exe"):
            p = Path(LLAMA_BIN_DIR) / (name + ext)
            if p.exists():
                return str(p)
    # fall back to PATH
    import shutil
    found = shutil.which(name) or shutil.which(name + ".exe")
    if found:
        return found
    return name   # let subprocess fail with a clear message

LLAMA_BENCH = _bin("llama-bench")
LLAMA_PPL   = _bin("llama-perplexity")
LLAMA_SERVER= _bin("llama-server")

PPL_CORPUS  = str(_HERE / "data" / "wikitext-2-raw" / "wiki.test.raw")


def validate(require_draft=False):
    """Call from setup.py or the harnesses to fail early with clear messages."""
    errors = []
    import shutil
    for label, path in [("llama-bench",       LLAMA_BENCH),
                        ("llama-perplexity",   LLAMA_PPL),
                        ("llama-server",       LLAMA_SERVER)]:
        if not (Path(path).exists() or shutil.which(path)):
            errors.append(f"  {label}: not found at '{path}'")
    if not MODEL:
        errors.append("  model: not set (edit config.toml [paths] model or set LLAMA_MODEL)")
    elif not Path(MODEL).exists():
        errors.append(f"  model: file not found: {MODEL}")
    if require_draft:
        if not DRAFT_MODEL:
            errors.append("  draft_model: not set (needed for bench_server.py)")
        elif not Path(DRAFT_MODEL).exists():
            errors.append(f"  draft_model: file not found: {DRAFT_MODEL}")
    return errors
