"""Shared bash building blocks spliced into the generated SLURM sbatch scripts."""

from __future__ import annotations


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


def sbatch_header(
    name: str,
    sbatch_cfg: dict,
    log_dir: str,
    *,
    cpus_per_task: int | None = None,
    log_name: str | None = None,
    errexit: bool = True,
) -> list[str]:
    """Return a job script's shebang, #SBATCH block, shell options and env activation.

    One builder for every gfm-env generator, so their headers cannot drift.

    Parameters
    ----------
    name : str
        SLURM job name.
    sbatch_cfg : dict
        An `hpc.sbatch`-shaped dict (partition, account, time, mem,
        cpus_per_task, env_activate_cmd); an empty account omits --account.
    log_dir : str
        Linux directory of the `{log_name}_%j.out/.err` files.
    cpus_per_task : int, optional
        Overrides `sbatch_cfg["cpus_per_task"]` (1 for light dispatch jobs).
    log_name : str, optional
        Log file stem; defaults to `name`.
    errexit : bool
        False writes `set -uo pipefail` instead of `set -euo pipefail`, for
        scripts whose later commands must run after an earlier one failed;
        the env activation then gets an explicit `|| exit 1`.

    Returns
    -------
    list of str
        Lines ending with a blank line after the env activation.
    """
    cpus = sbatch_cfg["cpus_per_task"] if cpus_per_task is None else cpus_per_task
    log_name = name if log_name is None else log_name
    return [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={sbatch_cfg['partition']}",
        *_account_line(sbatch_cfg),
        f"#SBATCH --time={sbatch_cfg['time']}",
        f"#SBATCH --mem={sbatch_cfg['mem']}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={log_dir}/{log_name}_%j.out",
        f"#SBATCH --error={log_dir}/{log_name}_%j.err",
        "",
        "set -euo pipefail" if errexit else "set -uo pipefail",
        sbatch_cfg["env_activate_cmd"] if errexit else f"{sbatch_cfg['env_activate_cmd']} || exit 1",
        "",
    ]


def read_targets_lines(var: str, targets_file: str) -> list[str]:
    """Return bash lines loading `targets_file` (one target per line) into array `var`.

    Use as `"${VAR[@]}"`. Unlike an unchecked `$(cat file)`, a missing or empty
    file exits the script: an empty target list would make snakemake build its
    first rule instead of nothing.
    """
    return [
        f'mapfile -t {var} < "{targets_file}" || exit 1',
        f'if [ "${{#{var}[@]}}" -eq 0 ] || [ -z "${{{var}[0]}}" ]; then',
        f'    echo "ERROR: no targets in {targets_file}" >&2',
        "    exit 1",
        "fi",
    ]


def _retry_wrapper_lines() -> list[str]:
    """Bash fn retrying snakemake: P: share reads can hit stale caches (UnicodeDecodeError).

    Each failure dumps checksum/mtime of the candidate config files for forensics.
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
    """Bash fn staging a --configfile to node-local disk, validated as UTF-8 YAML.

    A compute node's stale P: share cache can corrupt the same file on every retry, so
    only the cheap copy+parse step is retried and later reads hit local disk.
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
    """Bash fn staging the tile_grid gpkg to node-local disk, validated by byte size.

    GDAL/SQLite's advisory lock on a gpkg read directly from the P: share can hang
    indefinitely on Hydrax compute nodes; a plain `cp` never takes that lock.
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
