#!/usr/bin/env python
"""
setup.py -- one-time setup and configuration verifier for llamacpp-autotune.

Run this before your first experiment:
  python setup.py

It will:
  1. Check that llama-bench, llama-perplexity, and llama-server are reachable.
  2. Check that the model file exists and is readable.
  3. Download the wikitext-2-raw perplexity corpus if it's missing.
  4. Run a quick smoke-test benchmark (2 reps, short workload) to confirm
     llama-bench runs without error on your machine.

If anything fails, it prints a clear error and exits non-zero.
"""
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# autotune_config reads config.toml and env vars
try:
    import autotune_config as C
except ImportError:
    print("ERROR: autotune_config.py not found. Run setup.py from the project root.")
    sys.exit(1)

SEP = "-" * 60


def check(label, ok, detail=""):
    status = "  OK" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    return ok


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def main():
    all_ok = True

    # -- 1. binaries ----------------------------------------------------------─
    section("1 / 4  Checking binaries")
    import shutil
    for name, path in [("llama-bench",      C.LLAMA_BENCH),
                       ("llama-perplexity", C.LLAMA_PPL),
                       ("llama-server",     C.LLAMA_SERVER)]:
        found = Path(path).exists() or bool(shutil.which(path))
        all_ok &= check(name, found,
                        f"not found at '{path}'\n"
                        f"         Set [paths] llama_bin_dir in config.toml or "
                        f"set the LLAMA_BIN_DIR env var.")

    # -- 2. model file --------------------------------------------------------─
    section("2 / 4  Checking model")
    if not C.MODEL:
        print("  [FAIL] model path is empty\n"
              "         Set [paths] model in config.toml or set LLAMA_MODEL.")
        all_ok = False
    else:
        p = Path(C.MODEL)
        ok = p.exists() and p.stat().st_size > 1_000_000
        all_ok &= check(C.MODEL, ok,
                        "file not found or too small -- check the path in config.toml")
        if ok:
            print(f"         size: {p.stat().st_size / 1e9:.2f} GB")

    # -- 3. perplexity corpus --------------------------------------------------
    section("3 / 4  Perplexity corpus (wikitext-2-raw)")
    corpus = Path(C.PPL_CORPUS)
    if corpus.exists():
        check("wiki.test.raw", True)
        print(f"         {corpus}  ({corpus.stat().st_size // 1024} KB)")
    else:
        print("  Corpus not found -- downloading wikitext-2-raw from HuggingFace...")
        data_dir = corpus.parent.parent
        data_dir.mkdir(parents=True, exist_ok=True)
        zip_path = data_dir / "wikitext-2-raw-v1.zip"
        url = ("https://huggingface.co/datasets/ggml-org/ci/resolve/main/"
               "wikitext-2-raw-v1.zip")
        try:
            print(f"  Fetching {url} ...")
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(data_dir)
            zip_path.unlink()
            found = corpus.exists()
            all_ok &= check("wiki.test.raw downloaded", found)
            if found:
                print(f"         {corpus}")
        except Exception as e:
            print(f"  [FAIL] download failed: {e}")
            print("         Download manually and place at:")
            print(f"         {corpus}")
            all_ok = False

    # -- 4. smoke test --------------------------------------------------------─
    section("4 / 4  Smoke test")
    if not all_ok:
        print("  Skipping smoke test - fix the above errors first.")
    else:
        import json as _json

        # 4a. binary health: --list-devices (no model needed, instant)
        try:
            r = subprocess.run([C.LLAMA_BENCH, "--list-devices"],
                               capture_output=True, text=True, timeout=15)
            gpu_line = next((l for l in (r.stdout + r.stderr).splitlines()
                             if "Device" in l or "CUDA" in l or "Metal" in l), "")
            all_ok &= check("llama-bench --list-devices", r.returncode == 0,
                            r.stderr[-400:] if r.returncode != 0 else "")
            if gpu_line:
                print(f"         {gpu_line.strip()}")
        except Exception as e:
            all_ok &= check("llama-bench --list-devices", False, str(e))

        # 4b. model file is readable (header check — no GPU load needed for setup)
        try:
            with open(C.MODEL, "rb") as f:
                magic = f.read(4)
            is_gguf = magic == b"GGUF"
            all_ok &= check("model is valid GGUF", is_gguf,
                            f"first 4 bytes are {magic!r}, expected b'GGUF'")
        except Exception as e:
            all_ok &= check("model is valid GGUF", False, str(e))

        # Note: we intentionally skip a full bench run here. Loading a large model
        # mid-session competes for VRAM with whatever else is running and can OOM.
        # Use `python bench.py --baseline` as the real first-run validation instead.
        print("  Skipping full bench load -- run `python bench.py --baseline` to confirm.")

    # -- summary --------------------------------------------------------------─
    print(f"\n{SEP}")
    if all_ok:
        print("Setup complete. Run your first experiment:")
        print("  python bench.py --baseline")
        print("\nThen point an agent at program.md and let the loop run.")
    else:
        print("Setup incomplete - fix the errors above, then re-run setup.py.")
    print(SEP)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
