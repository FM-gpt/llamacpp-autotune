#!/usr/bin/env python3
"""Run a llamacpp-autotune command and write a durable report.

Examples:
  python scripts/run_with_report.py bench -- --baseline
  python scripts/run_with_report.py bench -- --fa off --desc "flash attention off"
  python scripts/run_with_report.py server -- --draft none
  python scripts/run_with_report.py setup

The wrapper intentionally does not decide keep/discard. It captures evidence for
review: command, environment metadata, stdout/stderr, parsed `tsv_row`, and exit
code. Reports are written under reports/ by default.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAFE_ENV_KEYS = [
    "LLAMA_BIN_DIR",
    "LLAMA_MODEL",
    "LLAMA_DRAFT_MODEL",
    "LLAMA_N_PROMPT",
    "LLAMA_N_GEN",
    "LLAMA_N_DEPTH",
    "LLAMA_REPS",
    "LLAMA_VRAM_BUDGET",
]


def run_capture(cmd: list[str], timeout: int | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return {
            "cmd": cmd,
            "returncode": 124,
            "stdout": stdout,
            "stderr": stderr + f"\n[TIMEOUT after {timeout}s]",
            "timeout": True,
        }


def small_cmd(cmd: list[str], timeout: int = 10) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e:
        return f"unavailable: {e}"


def metadata() -> dict[str, Any]:
    nvidia = ""
    if shutil_which("nvidia-smi"):
        nvidia = small_cmd([
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,clocks.sm",
            "--format=csv,noheader,nounits",
        ])
    return {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo": small_cmd(["git", "remote", "get-url", "origin"]),
        "branch": small_cmd(["git", "branch", "--show-current"]),
        "commit": small_cmd(["git", "rev-parse", "--short", "HEAD"]),
        "git_status_short": small_cmd(["git", "status", "--short"]),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "safe_env": {k: os.environ.get(k, "") for k in SAFE_ENV_KEYS if os.environ.get(k)},
        "nvidia_smi": nvidia,
    }


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


def parse_tsv_row(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.startswith("tsv_row:"):
            return line
    return None


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-")[:80] or "run"


def fenced(label: str, body: str) -> str:
    if not body:
        body = ""
    return f"### {label}\n\n```text\n{body.rstrip()}\n```\n"


def write_report(kind: str, argv: list[str], result: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata()
    combined = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
    tsv = parse_tsv_row(combined)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = slugify(" ".join(argv)) if argv else "run"
    name = f"{stamp}-{kind}-{suffix}"
    md_path = out_dir / f"{name}.md"
    json_path = out_dir / f"{name}.json"
    payload = {
        "kind": kind,
        "args": argv,
        "command": result["cmd"],
        "returncode": result["returncode"],
        "timeout": result.get("timeout", False),
        "tsv_row": tsv,
        "metadata": meta,
        "stdout": result.get("stdout", ""),
        "stderr": result.get("stderr", ""),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cmd_display = " ".join(shlex.quote(x) for x in result["cmd"])
    md = [
        f"# llamacpp-autotune report: {kind}\n",
        f"- Created: `{meta['created_at']}`\n",
        f"- Return code: `{result['returncode']}`\n",
        f"- Timeout: `{result.get('timeout', False)}`\n",
        f"- Command: `{cmd_display}`\n",
        f"- Repo: `{meta['repo']}`\n",
        f"- Branch: `{meta['branch']}`\n",
        f"- Commit: `{meta['commit']}`\n",
        f"- JSON sidecar: `{json_path.name}`\n",
    ]
    if tsv:
        md.append(f"- Parsed TSV row: `{tsv}`\n")
    md.append("\n## Environment\n\n")
    md.append(fenced("nvidia-smi", meta.get("nvidia_smi", "")))
    md.append(fenced("safe env", json.dumps(meta.get("safe_env", {}), indent=2)))
    md.append(fenced("git status --short", meta.get("git_status_short", "")))
    md.append("\n## Output\n\n")
    md.append(fenced("stdout", result.get("stdout", "")))
    md.append(fenced("stderr", result.get("stderr", "")))
    md_path.write_text("".join(md), encoding="utf-8")
    return md_path, json_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports-dir", default="reports", help="directory for .md/.json reports")
    ap.add_argument("--timeout", type=int, default=None, help="override subprocess timeout seconds")
    ap.add_argument("kind", choices=["setup", "bench", "server"], help="command family to run")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="arguments after -- are passed through")
    ns = ap.parse_args()
    passthrough = ns.args[1:] if ns.args[:1] == ["--"] else ns.args
    script = {"setup": "setup.py", "bench": "bench.py", "server": "bench_server.py"}[ns.kind]
    timeout = ns.timeout
    if timeout is None:
        timeout = {"setup": 300, "bench": 3600, "server": 3600}[ns.kind]
    cmd = [sys.executable, str(ROOT / script), *passthrough]
    result = run_capture(cmd, timeout=timeout)
    md_path, json_path = write_report(ns.kind, passthrough, result, ROOT / ns.reports_dir)
    print(f"report_md={md_path}")
    print(f"report_json={json_path}")
    print(f"returncode={result['returncode']}")
    return int(result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
