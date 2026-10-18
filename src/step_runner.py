"""Run one orchestrator pipeline step as a subprocess with banner, timing, fail-fast."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def run_step(
    cmd: list[str],
    label: str,
    fail_fast: bool = False,
    cwd: str | Path | None = None,
    env: dict | None = None,
) -> bool:
    """Run `cmd` and return True if it exited 0.

    A failed child never raises; with `fail_fast` this process exits with its code.
    """
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{'═' * 60}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s — {label}")
        if fail_fast:
            print("  Aborting (--fail-fast).")
            sys.exit(result.returncode)
        return False
    print(f"\n  ✓ Done in {elapsed:.0f}s — {label}")
    return True
