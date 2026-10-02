"""
sfincs_run.py -- Shared SFINCS subprocess execution (Popen + threaded
stdout/stderr forwarding + timeout), used by every rule that executes the
SFINCS binary. Also holds helpers for hand-crafting a sfincs.inp that
borrows geometry from a different SFINCS model directory via relative
paths, used by 13_build_sfincs.py and 14_run_spinup.py.

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


def parse_sfincs_inp(path: str | Path) -> dict[str, str]:
    """Parses a sfincs.inp file into a lowercased {key: value} dict.

    SFINCS's format is plain ``key = value`` lines; comments start with
    ``!``. Generic reader, not aware of which keys mean what.
    """
    cfg: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("!"):
                key, _, val = line.partition("=")
                cfg[key.strip().lower()] = val.strip()
    return cfg


def forward_geometry_files(
    cfg: dict[str, str],
    source_root: str | Path,
    dest_root: str | Path,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Builds ``"key = <relative path>"`` lines forwarding every non-empty
    ``*file`` entry in ``cfg`` from ``source_root`` to a new sfincs.inp at
    ``dest_root``, using a relative path computed with os.path.relpath.

    Skips keys in ``exclude`` and any ``*file`` entry whose target doesn't
    exist or is a 0-byte placeholder.
    """
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    lines = []
    for key, value in cfg.items():
        if not key.endswith("file") or key in exclude:
            continue
        fpath = source_root / value
        if fpath.exists() and fpath.stat().st_size > 0:
            rel = os.path.relpath(fpath, start=dest_root)
            lines.append(f"{key:<20} = {rel}")
    return lines


def run_sfincs_subprocess(
    sfincs_exe: str | Path,
    cwd: str | Path,
    timeout_s: float,
    log,
    label: str = "SFINCS run",
    n_threads: int | None = None,
) -> None:
    """
    Executes the SFINCS binary in ``cwd``, streaming stdout/stderr
    line-by-line to both ``log`` and the terminal, with a timeout.

    Args:
        sfincs_exe: Path to the sfincs executable.
        cwd:        Working directory SFINCS runs in; its own relative file
                    references (dep, msk, bnd, rstfile, ...) resolve
                    against this.
        timeout_s:  Wall-clock timeout (seconds) before the process is killed.
        log:        Logger to write SFINCS's stdout (info) / stderr (warning) to.
        label:      Used only in raised error messages.
        n_threads:  OpenMP thread count for SFINCS (sets OMP_NUM_THREADS for
                    the subprocess only). SFINCS defaults to 1 thread when
                    unset. None/0 leaves the environment unchanged.

    Raises:
        RuntimeError: on timeout or non-zero exit code.
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
