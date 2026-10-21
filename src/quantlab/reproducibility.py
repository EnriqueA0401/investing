"""Seeding and environment fingerprinting for reproducible runs."""

from __future__ import annotations

import hashlib
import platform
import random
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

_FINGERPRINT_PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "pydantic", "pyyaml", "quantlab")


def seed_everything(seed: int) -> np.random.Generator:
    """Seed global RNGs (defensive) and return the explicit generator components should use."""
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - legacy global seeded only as a safety net
    return np.random.default_rng(seed)


def git_state(repo: Path | None = None) -> dict[str, Any]:
    cwd = repo or Path.cwd()

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def environment_fingerprint(lockfile: Path = Path("uv.lock")) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for pkg in _FINGERPRINT_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "lockfile_sha256": _file_hash(lockfile),
    }
