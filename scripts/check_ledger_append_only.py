"""Fail if research/trials.jsonl lost or changed any previously committed line.

The ledger is the multiple-testing record; rewriting it would silently
deflate the trial count used by the Deflated Sharpe Ratio and PBO.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LEDGER = Path("research/trials.jsonl")


def main(base: str = "HEAD") -> int:
    try:
        committed = subprocess.run(
            ["git", "show", f"{base}:{LEDGER.as_posix()}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return 0  # ledger not in base yet
    current = LEDGER.read_text() if LEDGER.exists() else ""
    if not current.startswith(committed):
        print(f"{LEDGER} was modified or truncated; it is append-only.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
