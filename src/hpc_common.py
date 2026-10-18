"""Shared bash building blocks spliced into the generated SLURM sbatch scripts."""

from __future__ import annotations


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


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
