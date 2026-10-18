"""Shared sbatch-script building blocks for the HPC/SLURM job generators.

Each function returns lines (or a single line) of bash to splice into a generated sbatch
script. They were copy-pasted, identically, into generate_eikonal_jobs.py,
generate_hpc_exposure_jobs.py, generate_hpc_flood_totals_and_plots_job.py,
generate_hpc_postprocess_job.py and generate_hpc_preprocess_job.py (in
snakemake_workflow/scripts/); this is now the single definition, so the generated scripts
cannot drift apart. Pure string builders: no I/O.
"""

from __future__ import annotations


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


def _retry_wrapper_lines() -> list[str]:
    """Bash function def: retries a `snakemake --configfile ...` invocation
    on failure - confirmed live 2026-09: a job with NO SLURM dependency
    (build_shared_inputs.sbatch - submitted essentially instantly after
    resolved_config.yml is written on the login node, unlike every other
    generated job, which waits on something first) can hit a cross-node
    filesystem consistency gap on the shared P:\\ mount and read a
    corrupted/stale copy of that file (UnicodeDecodeError inside
    Snakemake's own --configfile loader) - the SAME exact byte at the SAME
    exact position every time it's happened, ruling out a random race and
    pointing at cache staleness specifically (very plausibly SMB/CIFS-style
    client caching on this mount, not NFS). Retrying (rather than a fixed
    sleep) matches this codebase's own retry_transient_io philosophy -
    assume transient, retry with backoff, fail loudly only once attempts
    are genuinely exhausted.

    2026-09-14: staging resolved_config.yml to node-local disk first
    (_stage_configfile_lines, below) did NOT stop this recurring, which
    means it isn't necessarily resolved_config.yml at all - the Snakefile
    itself has `configfile: "snakemake_workflow/config/config.yml"` +
    a conditional one for config_local.yml, BOTH loaded automatically on
    every snakemake invocation before --configfile is even merged in, and
    raw UnicodeDecodeErrors never report which file they came from. Each
    failed attempt below now dumps a checksum/mtime of every candidate
    config file, so a recurrence gives real forensic evidence instead of
    another guess.
    """
    return [
        "run_snakemake_with_retry() {",
        "  local attempt=1 max_attempts=10 delay=30",
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    if "$@"; then',
        "      return 0",
        "    fi",
        '    echo "  [retry $attempt/$max_attempts] snakemake invocation failed - retrying in ${delay}s..." >&2',
        '    echo "  [forensics] candidate config files at time of failure:" >&2',
        '    sha256sum snakemake_workflow/config/config.yml snakemake_workflow/config/config_local.yml "$LOCAL_CONFIGFILE" >&2 || true',
        '    stat snakemake_workflow/config/config.yml snakemake_workflow/config/config_local.yml "$LOCAL_CONFIGFILE" >&2 || true',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  snakemake invocation failed after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]


def _stage_configfile_lines() -> list[str]:
    """Bash function def: copies a --configfile path to node-local scratch
    and validates it parses as UTF-8 YAML before use, retrying the CHEAP
    copy+validate step (not a full snakemake run) on failure.

    Escalation from run_snakemake_with_retry, above - live evidence
    2026-09-14 showed 5 back-to-back FRESH `snakemake` retries (new
    process each time, ~75s apart) all hit the IDENTICAL byte/position
    UnicodeDecodeError reading resolved_config.yml over the shared P:\\
    mount, while the very same file read completely cleanly from a
    different client moments later. That rules out a short-lived cross-
    node race (which retrying the whole snakemake invocation assumed) and
    points at THIS COMPUTE NODE's own stale/corrupted client-side cache
    for that specific file - re-running snakemake on the same node
    against the same P:\\ path just keeps hitting the same bad cache
    entry. Staging a local copy first isolates the flaky network read to
    one small, cheap step that can be retried on its own; every
    subsequent read (Snakemake's own config parse, or
    generate_hpc_simulation_jobs.py's plain yaml.safe_load) then hits
    node-local disk instead of the network mount again.
    """
    return [
        "stage_configfile_locally() {",
        '  local src="$1" out_var="$2"',
        '  local stage_dir="${TMPDIR:-/tmp}/gfm_resolved_config"',
        '  mkdir -p "$stage_dir"',
        '  local dst="$stage_dir/resolved_config_${SLURM_JOB_ID:-$$}_$$.yml"',
        "  local attempt=1 max_attempts=8 delay=5",
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    cp -f -- "$src" "$dst" 2>/dev/null',
        "    if python -c \"import sys, yaml; yaml.safe_load(open(sys.argv[1], encoding='utf-8'))\" \"$dst\" 2>/dev/null; then",
        "      printf -v \"$out_var\" '%s' \"$dst\"",
        "      return 0",
        "    fi",
        '    echo "  [config-stage retry $attempt/$max_attempts] $src did not copy/parse cleanly - retrying in ${delay}s..." >&2',
        '    rm -f -- "$dst"',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  failed to stage a valid local copy of $src after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]


def _stage_tile_grid_lines() -> list[str]:
    """Bash function def: copies the tile_grid gpkg to node-local scratch,
    validated by comparing copied/source byte sizes (not UTF-8/YAML parsing
    like `_stage_configfile_lines` - a GeoPackage is a binary SQLite file),
    retrying on failure - see that function's own docstring for the matching
    retry rationale.

    Added 2026-10-13: `load_tile_grid()` (src/tiles.py) opens this gpkg via
    GDAL/OGR, which - unlike a plain `os.stat`/`open()` - has SQLite try to
    acquire an advisory file lock first. Confirmed live that this lock
    acquisition can hang indefinitely on Hydrax's 24vcpu compute nodes
    reading P:\\ directly (reproduced with a single, non-concurrent read -
    not a concurrency/contention issue: 24 simultaneous reads and 1 serial
    read both hung the full job time limit with zero output), while the
    identical read completes instantly from the login node. Staging the
    gpkg locally once per job (same pattern as resolved_config.yml) sidesteps
    GDAL/SQLite's own network-mount locking path entirely, regardless of its
    underlying (likely NFS lock-daemon-side) cause - a plain `cp` never
    triggers that lock acquisition.
    """
    return [
        "stage_tile_grid_locally() {",
        '  local src="$1" out_var="$2"',
        '  local stage_dir="${TMPDIR:-/tmp}/gfm_tile_grid"',
        '  mkdir -p "$stage_dir"',
        '  local dst="$stage_dir/tile_grid_${SLURM_JOB_ID:-$$}_$$.gpkg"',
        "  local attempt=1 max_attempts=8 delay=5",
        '  local src_size; src_size=$(stat -c%s -- "$src" 2>/dev/null)',
        '  while [ "$attempt" -le "$max_attempts" ]; do',
        '    cp -f -- "$src" "$dst" 2>/dev/null',
        '    local dst_size; dst_size=$(stat -c%s -- "$dst" 2>/dev/null)',
        '    if [ -n "$dst_size" ] && [ "$dst_size" = "$src_size" ]; then',
        "      printf -v \"$out_var\" '%s' \"$dst\"",
        "      return 0",
        "    fi",
        '    echo "  [tile-grid-stage retry $attempt/$max_attempts] $src did not copy cleanly - retrying in ${delay}s..." >&2',
        '    rm -f -- "$dst"',
        '    sleep "$delay"',
        "    attempt=$((attempt + 1))",
        "  done",
        '  echo "  failed to stage a valid local copy of $src after $max_attempts attempts - giving up." >&2',
        "  return 1",
        "}",
    ]
