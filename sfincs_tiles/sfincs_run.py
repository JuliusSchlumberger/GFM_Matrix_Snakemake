"""
sfincs_run.py -- Shared SFINCS subprocess execution (Popen + threaded
stdout/stderr forwarding + timeout), used by every rule that executes the
SFINCS binary. Also holds helpers for hand-crafting a sfincs.inp that
borrows geometry from a different SFINCS model directory via relative
paths. Adapted from the basin-scale SFINCS workflow's own sfincs_run.py
(13_build_sfincs.py / 14_run_spinup.py there), which was vendored here as
julius_sfincs_code/ until cleanup Phase 2 removed it - see git history.

Paths are written by hand rather than via HydroMT's sf.config.write():
HydroMT's get_set_file_variable absolutizes file references outside the
model's own root instead of preserving a relative ../ path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path


def run_sfincs_subprocess(
    sfincs_exe: str | Path,
    cwd: str | Path,
    timeout_s: float,
    log,
    label: str = "SFINCS run",
    n_threads: int | None = None,
) -> None:
    """Run the SFINCS binary in ``cwd`` and stream its output, with a timeout.

    Launches ``sfincs_exe`` via ``subprocess.Popen`` and forwards its
    stdout/stderr line-by-line to both ``log`` and the terminal in real time.

    Parameters
    ----------
    sfincs_exe : str or Path
        Path to the SFINCS executable.
    cwd : str or Path
        Working directory SFINCS runs in; its own relative file references
        (dep, msk, bnd, rstfile, ...) resolve against this.
    timeout_s : float
        Wall-clock timeout (seconds) before the process is killed.
    log : logging.Logger
        Logger that receives SFINCS's stdout (info) / stderr (warning).
    label : str, optional
        Used only in raised error messages. Default is "SFINCS run".
    n_threads : int or None, optional
        OpenMP thread count for SFINCS (sets OMP_NUM_THREADS for the
        subprocess only). SFINCS defaults to 1 thread when unset; None or 0
        leaves the environment unchanged. Default is None.

    Raises
    ------
    RuntimeError
        If the process exceeds ``timeout_s`` or exits with a non-zero code.
    """
    sfincs_exe = Path(sfincs_exe).resolve()
    log.info(f"Running {label}: {sfincs_exe}")

    env = None
    if n_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(n_threads))
        log.info(f"{label}: OMP_NUM_THREADS={n_threads}")

    proc = subprocess.Popen(
        [str(sfincs_exe)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        env=env,
    )

    def _forward(pipe, log_fn):
        for line in pipe:
            line = line.rstrip()
            if line:
                log_fn(f"[sfincs] {line}")
                print(f"[sfincs] {line}", file=sys.stderr, flush=True)

    t_out = threading.Thread(target=_forward, args=(proc.stdout, log.info))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, log.warning))
    t_out.start()
    t_err.start()

    # proc.wait() must run before joining the reader threads, or the timeout
    # is unreachable (join() blocks until SFINCS's pipes close on exit).
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        raise RuntimeError(f"{label} exceeded {timeout_s}s timeout")

    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
