"""Generate HPC sbatch scripts that run preprocessing across hpc.n_nodes
PARALLEL nodes, then - once every node's batch has finished - one more job
that runs `snakemake generate_aqueduct_jobs` (wave sbatch generation) and
submits submit_waves.sh itself. So a full run (preprocessing AND
simulation) can be launched from a single driver script, with preprocessing
itself spread across the same number of nodes as the simulation waves use.

Preprocessing has no wave/hop_distance ordering constraint (every tile's
DEM/mask/friction/boundaries are independent of every other tile) - unlike
simulation, where hop>=1 tiles must wait for their neighbours. So instead of
one monolithic `snakemake generate_aqueduct_jobs --cores N` run on a single
node, this splits tiles by estimated size FIRST (same bbox-area pixel-count
proxy and hpc.large_tile_pixel_threshold hpc_dispatch.smk uses for
simulation batching - tiles at/above it use hpc.sbatch_large, the rest use
hpc.sbatch), then splits each size class's target files evenly across up to
hpc.n_nodes batches, writes one `snakemake --cores N <explicit target file
list>` sbatch script per batch, and submits all of them with NO dependency
between them (fully parallel) - then submits ONE more job with
`--dependency=afterany:<every batch job id>` that runs generate_aqueduct_jobs
(fast, since every preprocessing output already exists by then) and chains
into submit_waves.sh. This is the exact same afterany-join-multiple-jobs
pattern generate_aqueduct_jobs.py's own submit_waves.sh already uses between
simulation waves - just one more phase in front of wave 0.

Before any batch is submitted, submit_preprocess_and_dispatch.sh itself
builds every shared, tile-independent output this DAG has
(`compute_geoid_offset_raster`'s single file, PLUS
`cache_waterlevel_stations`'s one cached GeoPackage per (return_period,
waterlevel_name) scenario, 2026-08 - unlike every other rule, which is
per-tile) SYNCHRONOUSLY, in the same shell, before calling `sbatch` on
anything (2026-09-14 - previously this was its own zero-dependency sbatch
job, "wave 0"/`build_shared_inputs.sbatch`, submitted immediately after
resolved_config.yml was written; that specific job kept hitting a
cross-node stale-read of that file on the shared P:\\ mount, identical
byte/position every time, that not even retries survived - see
_stage_configfile_lines' docstring. Running it synchronously in the same
process/machine that just wrote resolved_config.yml removes the network
round-trip for this step entirely, not just mitigates it). Snakemake locks
its own working directory by default (one process at a time per
directory), so N concurrent `snakemake` invocations from the same
code_root would otherwise fail with LockException the instant a second
one starts - every batch's `snakemake` call therefore passes `--nolock`,
safe ONLY because the shared build above has already completed by the
time any batch is submitted, eliminating every real write-write race
`--nolock` would otherwise leave unprotected.

Target file paths are reconstructed directly (not via `rules.X.output.Y`
references, since this is a standalone script, not a Snakemake `script:`)
using the exact same path templates preprocessing.smk's rules declare -
mirrors the same approach generate_aqueduct_jobs.py already uses for its
own local-view boundaries check.

Uses the same local-view/Linux-view path resolution as
generate_aqueduct_jobs.py (config_hpc.yml, if present), so it can be run
either from the Windows preprocessing machine (writing scripts meant for
Hydrax) or natively ON Hydrax (config_hpc.yml unnecessary in that case).

Usage:
    python generate_hpc_preprocess_job.py [--config path/to/config.yml]
    bash <printed submit_preprocess_and_dispatch.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import (  # noqa: E402
    atomic_write, load_config, merged_slr_scenarios, retry_transient_io, split_batches_proportionally,
)


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


def _target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    paths = [f"{tile_dir}/inputs/dem.tif", f"{tile_dir}/inputs/mask.tif", f"{tile_dir}/inputs/friction.tif"]
    for rp in return_periods:
        for slr in waterlevel_names:
            paths.append(f"{tile_dir}/inputs/boundaries_{rp}_{slr}.gpkg")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--calibration", action="store_true",
        help="use generate_hpc_simulation_jobs.py directly for the post-preprocessing "
             "wave-dispatch phase instead of `snakemake generate_aqueduct_jobs` - skips "
             "Snakemake's own full-DAG preprocessing-output verification (safe here: this "
             "phase only runs after submit_preprocess_and_dispatch.sh's own afterany chain "
             "already guarantees every preprocessing batch above finished), and - unlike "
             "`rule generate_aqueduct_jobs`, whose own base_config_path is a hardcoded "
             "literal path to production config.yml regardless of --configfile (see "
             "hpc_dispatch.smk) - correctly honors a scenario --config end to end. "
             "Appropriate at calibration-subset scale, where the full-DAG build's "
             "multi-hour cost at production scale buys nothing anyway.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    linux_model_outputs = linux_config["simulation"]["model_outputs"]
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]
    large_pixel_threshold = hpc_cfg["large_tile_pixel_threshold"]

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Written once here and referenced (via GFM_CONFIG_PATH) by every snakemake
    # invocation this script generates - without this, a compute node
    # re-parsing the Snakefile from a bare `snakemake ...` call falls back to
    # its own default config.yml, silently ignoring whatever --config this
    # script itself was given (a scenario config, when --config points at
    # one - see generate_aqueduct_jobs.py's generate_wave_dispatch, which
    # already does the same thing for the simulation phase).
    #
    # atomic_write (temp file + os.replace), NOT a plain open()+write -
    # confirmed live 2026-09: a SLURM job on a different compute node reading
    # this over the shared P:\ network filesystem moments after it's written
    # can otherwise see a partial/garbled file (UnicodeDecodeError on a
    # mid-write or stale-cache read) - build_shared_inputs.sbatch hit exactly
    # this, while a later read of the same file (~5min on, after more of the
    # pipeline had run) succeeded. Same latent bug already existed in
    # generate_aqueduct_jobs.py's own resolved_config.yml write - fixed
    # there too, not just here.
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    # Local view (this machine's own reachable mount), not linux_config's -
    # a genuine pre-existing bug, found 2026-08-10 while testing the
    # shared-targets change below: linux_config's path is a Linux-style
    # string (e.g. /p/...) meant to be EMBEDDED into generated sbatch
    # scripts, not actually opened by whichever machine happens to run this
    # generator - reading it directly fails outright when generated from
    # Windows (fiona.errors.DriverError, no such path on Windows). Local and
    # Linux views point at the same underlying shared storage, so the DATA
    # read is identical regardless of which config's path string opens it -
    # only the STRING form embedded into sbatch scripts below needs to be
    # the Linux one. Same fix already applied in generate_exposure_jobs.py's
    # own chunk discovery for the identical reason.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    return_periods = [f"RP{rp}" for rp in linux_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(linux_config["boundary_conditions"], linux_config["adaptation"])

    # Every shared, non-tile-specific output this DAG has - the geoid-offset
    # raster (one file total) PLUS one cached water-level-station GeoPackage
    # per (return_period, waterlevel_name) scenario (cache_waterlevel_stations,
    # 2026-08 - see that rule's own docstring in preprocessing.smk). Both
    # classes need the SAME build-before-any-batch-starts treatment: a
    # --nolock batch racing a concurrent WRITE to either would hit the same
    # corruption risk compute_geoid_offset_raster's own phase-0 build already
    # exists to prevent - see module docstring.
    linux_shared_targets = [linux_config["vertical_datum_correction"]["offset_raster_path"]]
    linux_stations_cache_dir = f"{linux_config['paths']['processed_inputs_dir']}/WL_scenarios_cache"
    for rp in return_periods:
        for slr in waterlevel_names:
            linux_shared_targets.append(f"{linux_stations_cache_dir}/stations_{rp}_{slr}.gpkg")

    shared_targets_path = local_jobs_dir / "shared_targets.txt"
    with open(shared_targets_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(linux_shared_targets) + "\n")
    linux_shared_targets_file = f"{linux_jobs_dir}/shared_targets.txt"

    # Same bbox-area pixel-count proxy hpc_dispatch.smk uses for simulation
    # batching (area_deg2 * 3600**2 - DeltaDTM's ~1 arcsec native
    # resolution) - computable from tile geometry alone, no DEM read
    # needed, so it works before any preprocessing has run.
    bounds = tile_gdf.geometry.bounds
    tiles_by_class: dict[str, list[str]] = {"small": [], "large": []}
    for tile_id, minx, miny, maxx, maxy in zip(
        tile_gdf["tile_id"], bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"],
    ):
        approx_pixels = (maxx - minx) * (maxy - miny) * 3600.0 * 3600.0
        size_class = "large" if approx_pixels >= large_pixel_threshold else "small"
        tiles_by_class[size_class].append(str(int(tile_id)))
    for size_class in tiles_by_class:
        tiles_by_class[size_class].sort(key=int)

    # Node budget split PROPORTIONALLY to each class's share of tiles
    # (e.g. ~15% large / ~85% small on the real production grid), rather
    # than each class independently getting up to n_nodes - the latter
    # would let preprocessing use up to 2x n_nodes total (n_nodes for
    # small + n_nodes for large) even though "large" is a small minority
    # of the actual work. Shared with hpc_dispatch.smk's simulation-wave
    # batching (split_batches_proportionally, src/config_utils.py) - that
    # rule used to duplicate this exact logic with its own independent
    # min(n_nodes, len(class_tiles)) per class, which was the actual bug
    # (found live 2026-08-10: a wave with both size classes present could
    # claim up to 2x n_nodes at once) this shared helper now prevents from
    # recurring in either place.
    present_classes = [c for c in ("small", "large") if tiles_by_class[c]]
    class_n_nodes = split_batches_proportionally(
        {c: len(tiles_by_class[c]) for c in present_classes}, n_nodes,
    )

    batches = []  # [(size_class, batch_id, [tile_id, ...]), ...]
    for size_class, class_tiles in tiles_by_class.items():
        if not class_tiles:
            continue
        n_batches = class_n_nodes[size_class]
        k, m = divmod(len(class_tiles), n_batches)
        for i in range(n_batches):
            batch_tiles = class_tiles[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
            batches.append((size_class, f"{i:03d}", batch_tiles))

    batch_script_paths = []
    for size_class, batch_id, batch_tiles in batches:
        sbatch_cfg = hpc_cfg["sbatch_large"] if size_class == "large" else hpc_cfg["sbatch"]
        name = f"preprocess_{size_class}_batch_{batch_id}"
        targets = [
            p for tile_id in batch_tiles
            for p in _target_paths(f"{linux_model_outputs}/{tile_id}", return_periods, waterlevel_names)
        ]

        # Target list written to its own file (thousands of paths per
        # batch) and expanded at runtime via $(cat ...), rather than
        # inlining every path as a literal CLI argument in the sbatch
        # script body - keeps the script itself short and diffable.
        targets_path = local_jobs_dir / f"{name}_targets.txt"
        with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(targets) + "\n")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={sbatch_cfg['partition']}",
            *_account_line(sbatch_cfg),
            f"#SBATCH --time={sbatch_cfg['time']}",
            f"#SBATCH --mem={sbatch_cfg['mem']}",
            f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -euo pipefail",
            sbatch_cfg["env_activate_cmd"],
            "",
            *_retry_wrapper_lines(),
            *_stage_configfile_lines(),
            "",
            f'cd "{linux_code_root}"',
            f'echo "=== Preprocessing batch {size_class}/{batch_id}: {len(batch_tiles)} tiles ==="',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            "",
            # Hard pre-flight check, not just a submission-order convention:
            # --nolock (above) is only safe because submit_preprocess_and_
            # dispatch.sh itself builds EVERY shared, non-tile-specific
            # output this DAG has (geoid-offset raster + every cached
            # water-level-station GeoPackage - see linux_shared_targets
            # above) SYNCHRONOUSLY, before submitting any batch - see module
            # docstring. That safety currently depends entirely on every
            # batch actually being submitted via submit_preprocess_and_
            # dispatch.sh; a batch launched by hand (or an old/stale sbatch
            # script re-submitted directly, without that shared build having
            # run first) skips that ordering silently and could race a
            # shared build with --nolock disabling Snakemake's own
            # protection. This turns that into a loud, immediate,
            # unambiguous failure instead of an intermittent LockException or
            # (worse) silent corruption of a shared file - confirmed this is
            # a real failure mode, not hypothetical: exactly this happened on
            # 2026-08-08 (job 243423/243440 - LockException from a batch that
            # started before the shared build had run).
            f'while IFS= read -r shared_target; do',
            f'    if [ ! -f "$shared_target" ]; then',
            f'        echo "ERROR: shared preprocessing input not found: $shared_target" >&2',
            '        echo "This batch must not start before the shared-inputs build (in'
            ' submit_preprocess_and_dispatch.sh) completes." >&2',
            '        echo "Submit via submit_preprocess_and_dispatch.sh (which orders this'
            ' correctly) rather than running this .sbatch file directly/out of order." >&2',
            "        exit 1",
            "    fi",
            f'done < "{linux_shared_targets_file}"',
            "",
            (
                f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                '--rerun-triggers=mtime '
                f'$(cat "{linux_jobs_dir}/{name}_targets.txt")'
            ),
            "",
        ]
        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        batch_script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({size_class}, {len(batch_tiles)} tiles, {len(targets)} target files)")

    # Phase 2: once every preprocessing batch above has finished, generate
    # the wave sbatch scripts (fast - every input already exists) and
    # submit them. Lightweight (just calls snakemake + a shell script), so
    # it uses hpc.sbatch (the smaller of the two) rather than needing its
    # own dedicated config.
    #
    # --calibration routes this through generate_hpc_simulation_jobs.py
    # instead of `snakemake generate_aqueduct_jobs` - the latter's own
    # base_config_path (hpc_dispatch.smk) is a hardcoded literal path to
    # production config.yml, NOT derived from GFM_CONFIG_PATH, so
    # resolved_config.yml (and therefore every solver parameter +
    # tile_grid.path baked into it) would silently revert to production
    # defaults for a scenario run - see this script's own module
    # docstring / --calibration's help text. The default (non-calibration)
    # path still sets GFM_CONFIG_PATH below for whatever partial effect it
    # has (live in-memory Snakemake params - tile_ids, return_periods,
    # batches - DO correctly reflect it; resolved_config.yml does not, a
    # known, documented gap for that path only).
    if args.calibration:
        generate_call = (
            f'python snakemake_workflow/scripts/generate_hpc_simulation_jobs.py '
            f'--config "$LOCAL_CONFIGFILE"'
        )
    else:
        generate_call = (
            f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
            f'snakemake generate_aqueduct_jobs --cores 1 --nolock '
            f'--rerun-triggers=mtime'
        )

    dispatch_cfg = hpc_cfg["sbatch"]
    dispatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gfm_generate_jobs_and_dispatch",
        f"#SBATCH --partition={dispatch_cfg['partition']}",
        *_account_line(dispatch_cfg),
        f"#SBATCH --time={dispatch_cfg['time']}",
        f"#SBATCH --mem={dispatch_cfg['mem']}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --output={linux_jobs_dir}/logs/generate_jobs_and_dispatch_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/generate_jobs_and_dispatch_%j.err",
        "",
        "set -euo pipefail",
        dispatch_cfg["env_activate_cmd"],
        "",
        *_retry_wrapper_lines(),  # only used by the non-calibration snakemake branch below; harmless if unused
        *_stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== Generating wave sbatch scripts ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        generate_call,
        "",
        'echo "=== Submitting simulation waves ==="',
        f'bash "{linux_jobs_dir}/submit_waves.sh"',
        "",
    ]
    dispatch_script_path = local_jobs_dir / "generate_jobs_and_dispatch.sbatch"
    with open(dispatch_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(dispatch_lines))
    print(f"  wrote {dispatch_script_path}")

    # Master driver: build every shared, tile-independent input SYNCHRONOUSLY
    # first, right here in this script (see module docstring, 2026-09-14 -
    # this used to be its own zero-dependency sbatch job; running it inline
    # instead means the machine that just wrote resolved_config.yml is the
    # SAME machine that reads it back, removing the cross-node network-mount
    # read that job kept failing on). Only once that has genuinely finished
    # does this submit every preprocessing batch (still fully parallel
    # amongst themselves - no dependency between them, none needed since
    # shared inputs are now guaranteed to already exist), then the dispatch
    # job depending on ALL batches via afterany.
    shared_cfg = hpc_cfg["sbatch"]
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        shared_cfg["env_activate_cmd"],
        "",
        *_retry_wrapper_lines(),
        *_stage_configfile_lines(),
        "",
        f'cd "{linux_code_root}"',
        'echo "=== Building shared preprocessing inputs (synchronous, on this login node) ==="',
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        (
            f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
            'snakemake --cores 1 --nolock --rerun-triggers=mtime '
            f'$(cat "{linux_shared_targets_file}") 2>&1 | tee "{linux_jobs_dir}/logs/build_shared_inputs.log"'
        ),
        "",
        'IDS=""',
    ]
    for script in batch_script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
            'IDS="${IDS:+$IDS:}$JID"',
        ]
    submit_lines += [
        "",
        f'JID=$(sbatch --parsable --dependency=afterany:$IDS "{linux_jobs_dir}/generate_jobs_and_dispatch.sbatch")',
        (
            f'echo "submitted {linux_jobs_dir}/generate_jobs_and_dispatch.sbatch -> job $JID '
            f'(depends on all {len(batch_script_paths)} preprocessing batches)"'
        ),
    ]
    submit_script_path = local_jobs_dir / "submit_preprocess_and_dispatch.sh"
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    node_summary = ", ".join(f"{size_class}={class_n_nodes[size_class]}" for size_class in present_classes)
    print(
        f"\nDone. 1 synchronous shared-inputs build + {len(batch_script_paths)} preprocessing batch(es) "
        f"({node_summary} nodes, {n_nodes} total budget) + 1 dispatch job written to {local_jobs_dir}"
    )
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/submit_preprocess_and_dispatch.sh")


if __name__ == "__main__":
    main()
