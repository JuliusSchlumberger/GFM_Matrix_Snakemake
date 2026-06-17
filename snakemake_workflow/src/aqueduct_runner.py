"""Functions for running the compiled Aqueduct flood model executable."""

import subprocess
from pathlib import Path

# Substring Aqueduct's Julia runtime prints (to stdout/stderr) on an
# unhandled out-of-memory crash, e.g. in `component_indices` during the
# flood-extent connected-component filter (see `core/src/core.jl`). This is
# distinct from the `LLVM ERROR: Unable to allocate section memory!` JIT
# crash that can occur when multiple Aqueduct instances run concurrently
# (see `resources: aqueduct_runs=1`) - that one is transient/concurrency
# related, not tile-size related, so it is intentionally not matched here.
OOM_SIGNATURE = "OutOfMemoryError"


def run_aqueduct(executable_path: str | Path, config_path: str | Path) -> None:
    """Run the Aqueduct flood model executable with a given TOML configuration.

    Args:
        executable_path: Path to the compiled Aqueduct executable.
        config_path: Path to the TOML configuration file for this tile/scenario run.

    Raises:
        subprocess.CalledProcessError: If the Aqueduct executable exits with
            a non-zero status code. The captured stdout/stderr are printed
            before the error is re-raised.
    """
    try:
        result = subprocess.run(
            [str(executable_path), str(config_path)],
            text=True,
            capture_output=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        raise


def is_oom_error(error: subprocess.CalledProcessError) -> bool:
    """Return True if a failed Aqueduct run's captured output indicates an OutOfMemoryError.

    The memory cost of `component_indices` (`core/src/core.jl`) is dominated
    by the tile's total pixel count, not by the SLR water level, so a tile
    that runs out of memory for one `waterlevel_name` will do so for
    (nearly) every `waterlevel_name`.
    """
    return OOM_SIGNATURE in (error.stdout or "") or OOM_SIGNATURE in (error.stderr or "")


def oom_marker_path(log_dir: str | Path, tile_id: str) -> Path:
    """Path to the OOM marker file for `tile_id` in `log_dir`."""
    return Path(log_dir) / f"{tile_id}.txt"


def tile_marked_oom(log_dir: str | Path, tile_id: str) -> bool:
    """Return True if `tile_id` was previously marked as too large for Aqueduct (see `mark_tile_oom`)."""
    return oom_marker_path(log_dir, tile_id).exists()


def mark_tile_oom(log_dir: str | Path, tile_id: str, reason: str) -> None:
    """Record that `tile_id` ran out of memory in Aqueduct.

    Once marked, other `waterlevel_name` scenarios for this `tile_id` skip
    Aqueduct entirely (see `tile_marked_oom`), since they would run out of
    memory for the same reason (tile size).

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        reason: Short human-readable reason the tile was marked.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    oom_marker_path(log_dir, tile_id).write_text(reason)


def log_skipped_tile(log_dir: str | Path, tile_id: str, waterlevel_name: str, reason: str) -> None:
    """Record that a tile/scenario was skipped instead of being run through Aqueduct.

    Writes one marker file per (tile_id, waterlevel_name) to `log_dir`, named
    `{tile_id}_{waterlevel_name}.txt`, so that skipped tiles can later be
    looked up by `tile_id` (e.g. to plot a map of skipped tiles against the
    tile grid). Writing one file per job avoids concurrent-write issues
    between parallel Snakemake jobs.

    Args:
        log_dir: Directory to write the marker file to. Created if missing.
        tile_id: The tile's `tile_id`.
        waterlevel_name: The SLR scenario name.
        reason: Short human-readable reason the tile/scenario was skipped.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{tile_id}_{waterlevel_name}.txt").write_text(reason)
