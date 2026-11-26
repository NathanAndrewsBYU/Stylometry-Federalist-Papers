"""Provenance record written next to every result (plan section 5)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys


def _git(*args: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.run(["git", *args], cwd=here, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_manifest(config: dict, **extra) -> dict:
    pkgs = {}
    for name in ("numpy", "sklearn", "spacy", "torch", "transformers"):
        mod = sys.modules.get(name)
        if mod is not None:
            pkgs[name] = getattr(mod, "__version__", "?")
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "packages": pkgs,
        "config": config,
        "config_hash": config_hash(config),
        **extra,
    }


def write_manifest(path: str, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, default=str)
