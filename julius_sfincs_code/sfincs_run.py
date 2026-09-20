"""
sfincs_run.py -- Shared SFINCS subprocess execution (Popen + threaded
stdout/stderr forwarding + timeout), used by every rule that executes the
SFINCS binary. Also holds the shared "hand-craft a sfincs.inp that borrows
geometry from a DIFFERENT SFINCS model directory via relative paths"
helpers used by both 13_build_sfincs.py (borrows from sfincs_skeleton/)
and 14_run_spinup.py (borrows from sfincs_skeleton/ too, as a sibling of
its own spinup/ directory) -- see either script's own module docstring
for why this is hand-written rather than done via HydroMT's own
sf.config.write(): HydroMT's get_set_file_variable silently absolutizes
any file reference outside the model's own root instead of preserving a
relative ../ path, so a genuinely portable cross-directory reference has
to be written by hand.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path


def parse_sfincs_inp(path: str | Path) -> dict[str, str]:
    """Parse a sfincs.inp file into a lowercased {key: value} dict.

    SFINCS's own format is plain ``key = value`` lines (comments start with
    ``!``) -- this is a generic reader, not aware of which keys mean what.
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
    """Build ``"key = <relative path>"`` lines forwarding every non-empty
    ``*file`` entry in ``cfg`` (as parsed by parse_sfincs_inp from
    ``source_root``'s own sfincs.inp) to a NEW sfincs.inp being written at
    ``dest_root``, via a relative path computed with os.path.relpath (not
    hand-derived ``../`` counting -- robust to whatever the actual nesting
    depth between the two directories turns out to be).

    Skips keys in ``exclude`` (typically {"rstfile"}: a restart file
    reference must never be forwarded from a source model that doesn't
    have one yet) and any ``*file`` entry whose target doesn't exist or is
    a 0-byte placeholder (e.g. sfincs_subgrid.nc/sfincs.weir when that
    feature is disabled for this basin).
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
    Execute the SFINCS binary in ``cwd``, streaming stdout/stderr line-by-line
    to both ``log`` and the terminal (via stderr) as it runs, with a timeout.

    Args:
        sfincs_exe: Path to the sfincs executable.
        cwd:        Working directory SFINCS runs in (its own relative file
                    references -- dep, msk, bnd, rstfile, ... -- resolve
                    against this).
        timeout_s:  Wall-clock timeout (seconds) before the process is killed.
        log:        Logger to write SFINCS's stdout (info) / stderr (warning) to.
        label:      Used only in the raised error messages (e.g.
                    "SFINCS spin-up", "SFINCS event run", "SFINCS calibration run").
        n_threads:  OpenMP thread count to give SFINCS (sets OMP_NUM_THREADS
                    for the subprocess only). SFINCS reads OMP_NUM_THREADS
                    directly (reporting it in its own startup banner) and
                    defaults to 1 thread when the variable is unset --
                    callers should pass their own ``snakemake.threads`` here
                    (paired with ``threads: workflow.cores`` on the rule, so
                    Snakemake reserves the whole machine for the run and
                    SFINCS actually uses it, instead of both blocking other
                    jobs AND running on a single core). None/0 leaves the
                    environment unchanged (whatever OMP_NUM_THREADS --
                    typically unset -- the parent process already has).

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

    # proc.wait() must run BEFORE joining the reader threads: t_out.join()/
    # t_err.join() block unconditionally until SFINCS's own stdout/stderr
    # pipes close, which only happens once it exits on its own -- so calling
    # them first makes the timeout unreachable until the process has already
    # finished, silently defeating it. Killing the process here closes its
    # pipes, which is what lets the reader threads finish and join() return.
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
