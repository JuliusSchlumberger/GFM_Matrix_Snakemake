"""Run one pipeline step as a subprocess, with a banner, timing and fail-fast handling.

The single definition of the step runner used by the orchestrators
(analysis/run_analysis.py, validation/run_validation.py, run_delta_floodmaps.py), which each
had their own copy.
"""

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
    """Run `cmd` in a subprocess and report the outcome.

    Prints a banner with `label` and the command, runs it, then prints "Done in Ns" or
    "FAILED (exit N)". A child's own failure never raises here: the caller decides what a
    failed step means, except that `fail_fast` exits this process with the child's code.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    label : str
        Human-readable step name for the banner and the result line.
    fail_fast : bool, default False
        On failure, ``sys.exit`` with the child's return code instead of returning False.
    cwd : str or Path, optional
        Working directory for the child (default: inherit).
    env : dict, optional
        Environment for the child (default: inherit).

    Returns
    -------
    bool
        True if the command exited 0.
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
