"""Generate HPC sbatch scripts that run postprocessing (merge_chunk ->
prepare_exposure_grid_chunk -> compute_flood_fraction_chunk) across
hpc.n_nodes PARALLEL nodes - the postprocessing-side counterpart to
generate_hpc_preprocess_job.py, filling the gap between the simulation
dispatch and the exposure-analysis dispatch (generate_exposure_jobs.py).

Also plans and appends the exposure-analysis dispatch onto the SAME
submission (submit_postprocess_and_exposure.sh) via
generate_exposure_jobs.generate_exposure_dispatch, so the whole remaining
pipeline - postprocessing then exposure analysis - is submitted in one
call, with exposure analysis's own first phase gated on afterany of
postprocessing's last phase. See generate_exposure_dispatch's own
docstring for why this stays a single synchronous, upfront submission
rather than a job calling `sbatch` again later from a compute node.

Unlike simulation, postprocessing has no wave/hop_distance ordering
constraint at all - every (chunk, return_period, waterlevel_name) job is
independent of every other one. There IS a real ordering constraint WITHIN
one chunk, though: prepare_exposure_grid_chunk needs ONE merge_chunk output
(the (return_periods[0], protection.baseline_waterlevel_name) reference, for
grid metadata only - see postprocessing.smk's own rule docstring) to already
exist, and compute_flood_fraction_chunk needs prepare_exposure_grid_chunk's
population output. Rather than track that fine-grained per-chunk dependency,
this uses the same simple phase-barrier pattern already proven for
preprocessing (build_shared_inputs -> batches) and the exposure-analysis
dispatch (pass1 -> reduce_shares -> pass2 -> reduce_write): three phases,
each one fully parallel across nodes internally, each gated on the ENTIRE
previous phase via --dependency=afterany:

  Phase 1 (merge_chunk):               every (chunk, rp, slr) - no dependency
  Phase 2 (prepare_exposure_grid_chunk): every chunk - afterany: all phase 1
  Phase 3 (compute_flood_fraction_chunk): every (chunk, rp, slr) - afterany: all phase 2

Per-chunk mode (2026-10-09, used whenever there are at least as many chunks
as hpc.n_nodes): instead of the three barriers above, each batch gets WHOLE
chunks and requests their flood-fraction targets, so one Snakemake call runs
each chunk's merge -> exposure grid -> flood fraction chain in order - the
fine-grained per-chunk dependency resolved by Snakemake itself rather than by
SLURM phase barriers. On the ssp245 deltas run the barriers cost ~2.5 h of
queueing out of a 4.3 h wall clock (~1 h of actual compute). Same rules on the
same inputs, so outputs are identical. Chunks are spread over batches by tile
count (longest-processing-time-first); a chunk never spans two batches, so no
two jobs build the same chunk's exposure grid. With fewer chunks than nodes
the three-phase mode is kept - whole-chunk batches would leave nodes idle.

No phase-0 shared-inputs step is needed here (unlike preprocessing's
compute_geoid_offset_raster) - none of these three rules has a single
output shared across every chunk, so there is no write-write race for
--nolock to leave unprotected.

Each batch is a plain `snakemake --cores N --nolock --rerun-triggers=mtime
<target files>` call, the exact same pattern generate_hpc_preprocess_job.py
already uses successfully for preprocessing - not a new standalone CLI
script (unlike run_aqueduct_cli.py for simulation), since these rules'
Snakemake DAG-build cost is cheap (each target's own dependency chain is
just a handful of per-tile waterdepth files or one other chunk file, not
the entire domain like generate_aqueduct_jobs' _PREPROCESS_OUTPUTS gate).

Chunk grid construction (_build_chunk_grid) is copied from the root
Snakefile's own module-level code - it can't be imported directly (the
Snakefile is not a plain importable module), so this mirrors it exactly;
if that logic ever changes there, update it here too.

Uses the same local-view/Linux-view path resolution as
generate_hpc_preprocess_job.py (config_hpc.yml, if present): the tile grid
is read via the LOCAL config (this machine's own reachable mount), while
every path string embedded into generated sbatch scripts uses the Linux
view.

Usage:
    python generate_hpc_postprocess_job.py [--config path/to/config.yml]
    bash <printed submit_postprocess.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import yaml
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402
from generate_exposure_jobs import generate_exposure_dispatch, generate_exposure_resume_dispatch  # noqa: E402


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


def _retry_wrapper_lines() -> list[str]:
    """Bash function def: retries a `snakemake --configfile ...` invocation
    on failure - see generate_hpc_preprocess_job.py's matching helper for
    the full rationale (confirmed live 2026-09: a job with no/minimal
    dependency delay can read a stale/corrupted resolved_config.yml over
    the shared P:\\ mount - same exact failure every time, pointing at
    cross-node filesystem cache staleness, not a random race).

    2026-09-14: staging resolved_config.yml locally did NOT stop this
    recurring, so it may not be resolved_config.yml at all - the Snakefile
    auto-loads config.yml + config_local.yml (configfile: directives)
    before --configfile is even merged, and a raw UnicodeDecodeError never
    reports which file it came from. Dumping checksums/mtimes of every
    candidate on failure turns a recurrence into real evidence.
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
    copy+validate step (not a full snakemake run) on failure - see
    generate_hpc_preprocess_job.py's matching helper for the full
    rationale (live evidence 2026-09-14: 5 fresh `snakemake` retries in a
    row hit the identical byte/position UnicodeDecodeError on the same
    node, while the file read cleanly from a different client moments
    later - a node-local stale-cache issue that re-running snakemake
    against the same P:\\ path doesn't fix).
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


def _build_chunk_grid(tile_gdf: gpd.GeoDataFrame, chunk_size_deg: float) -> gpd.GeoDataFrame:
    """Mirrors the root Snakefile's own _build_chunk_grid exactly."""
    minx, miny, maxx, maxy = tile_gdf.total_bounds
    sz = chunk_size_deg
    xs = np.arange(np.floor(minx / sz) * sz, np.ceil(maxx / sz) * sz, sz)
    ys = np.arange(np.floor(miny / sz) * sz, np.ceil(maxy / sz) * sz, sz)
    rows = []
    for x in xs:
        for y in ys:
            cell = shapely_box(x, y, x + sz, y + sz)
            if not tile_gdf.geometry.intersects(cell).any():
                continue
            xi, yi = int(round(x)), int(round(y))
            lat = f"N{yi:02d}" if yi >= 0 else f"S{-yi:02d}"
            lon = f"E{xi:03d}" if xi >= 0 else f"W{-xi:03d}"
            rows.append({"chunk_id": f"{lat}{lon}", "geometry": cell})
    return gpd.GeoDataFrame(rows, crs=tile_gdf.crs)


def _split_evenly(targets: list[str], n_nodes: int) -> list[list[str]]:
    """Split `targets` into up to n_nodes contiguous, near-equal batches."""
    n_batches = min(n_nodes, len(targets))
    k, m = divmod(len(targets), n_batches)
    return [targets[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_batches)]


def _split_by_chunk(
    targets_by_chunk: dict[str, list[str]], chunk_weight: dict[str, float], n_nodes: int,
) -> list[list[str]]:
    """Assign WHOLE chunks to up to n_nodes batches, heaviest chunk first onto
    the currently lightest batch (longest-processing-time-first). Keeping a
    chunk's targets in one batch is what lets one Snakemake call run that
    chunk's merge_chunk -> prepare_exposure_grid_chunk ->
    compute_flood_fraction_chunk chain itself, with no cross-batch barrier,
    and guarantees no two batches ever build the same chunk's shared
    exposure grid concurrently."""
    chunks = [c for c in targets_by_chunk if targets_by_chunk[c]]
    n_batches = min(n_nodes, len(chunks))
    loads = [0.0] * n_batches
    batches: list[list[str]] = [[] for _ in range(n_batches)]
    for cid in sorted(chunks, key=lambda c: -chunk_weight.get(c, 1.0)):
        j = loads.index(min(loads))
        batches[j] += targets_by_chunk[cid]
        loads[j] += chunk_weight.get(cid, 1.0)
    return batches


def _write_batches(
    local_jobs_dir: Path, linux_jobs_dir: str, linux_code_root: str, sbatch_cfg: dict,
    phase_name: str, batches: list[list[str]], configfile_path: str,
) -> list[str]:
    """Write one sbatch script per batch of targets (plain
    `GFM_CONFIG_PATH=<configfile_path> snakemake --cores N <targets>` call,
    matching generate_hpc_preprocess_job.py's own pattern - configfile_path
    (resolved_config.yml) is what makes a compute node re-parsing the
    Snakefile see the same config this dispatch was generated from, rather
    than silently falling back to its own default config.yml;
    GFM_CONFIG_PATH rather than --configfile since 2026-09-14 - see the
    Snakefile's own comment on that env var), return their Linux paths.
    """
    script_paths = []
    for i, batch_targets in enumerate(batches):
        name = f"postprocess_{phase_name}_batch_{i:03d}"

        targets_path = local_jobs_dir / f"{name}_targets.txt"
        with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch_targets) + "\n")

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
            f'echo "=== Postprocessing {phase_name} batch {i:03d}: {len(batch_targets)} target(s) ==="',
            f'stage_configfile_locally "{configfile_path}" LOCAL_CONFIGFILE || exit 1',
            "",
            (
                f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                # --rerun-incomplete: an output a cancelled/killed earlier job
                # was still writing stays flagged incomplete in Snakemake's
                # metadata - redo just those instead of aborting the batch.
                f'--rerun-triggers=mtime --rerun-incomplete '
                f'$(cat "{linux_jobs_dir}/{name}_targets.txt")'
            ),
            "",
        ]
        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_targets)} target(s))")
    return script_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--resume", action="store_true",
        help="rebalance only the still-missing targets across the full node budget per phase "
             "(skipping any phase that's already 100% done), instead of a fresh dispatch. "
             "Cancel the original dispatch's still-running jobs FIRST - see "
             "generate_hpc_simulation_jobs.py's own --resume docstring for why (same reasoning: "
             "a stale batch racing a new one over the same still-incomplete targets would redo "
             "work concurrently for as long as both stay alive).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]
    sbatch_cfg = hpc_cfg["sbatch"]

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # Normally already written by the preceding preprocess/simulation
    # phases for this same hpc.jobs_dir (generate_hpc_preprocess_job.py) -
    # only write it here if this script is invoked standalone, so this
    # script is safe to run on its own too. Referenced (via GFM_CONFIG_PATH)
    # by every snakemake call this script generates - see _write_batches.
    # atomic_write, not a plain open()+write - see generate_hpc_preprocess_job.py's
    # matching comment: a SLURM job on a different node reading this over the
    # shared P:\ filesystem moments after a non-atomic write can see a
    # partial/garbled file (confirmed live 2026-09).
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    if not resolved_config_path.exists():
        atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    # Local view (this machine's own reachable mount) for the tile-grid
    # read - see generate_hpc_preprocess_job.py's own note on this exact
    # point. Linux view for the merged_outputs path embedded into targets.
    tile_gdf = retry_transient_io(gpd.read_file, local_config["tile_grid"]["path"])
    return_periods = [f"RP{rp}" for rp in local_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(local_config["boundary_conditions"], local_config["adaptation"])
    baseline_slr = local_config["protection"]["baseline_waterlevel_name"]

    chunk_size_deg = local_config["postprocessing"]["chunk_size_deg"]
    chunk_grid = _build_chunk_grid(tile_gdf, chunk_size_deg)
    chunk_ids = chunk_grid["chunk_id"].tolist()
    print(f"{len(chunk_ids)} populated chunk(s) (chunk_size_deg={chunk_size_deg}), "
          f"{len(return_periods)} RPs, {len(waterlevel_names)} SLRs, n_nodes={n_nodes}\n")

    linux_merged = linux_config["postprocessing"]["merged_outputs"]
    local_merged = local_config["postprocessing"]["merged_outputs"]

    # (chunk_id, linux_target, local_target) triples - resume mode filters on
    # local existence, fresh mode uses every linux target unconditionally.
    # Avoids needing a separate linux->local path translation helper.
    phase1_pairs = [
        (cid, f"{linux_merged}/chunks/waterdepth_{cid}_{rp}_{slr}.tif",
         f"{local_merged}/chunks/waterdepth_{cid}_{rp}_{slr}.tif")
        for cid in chunk_ids for rp in return_periods for slr in waterlevel_names
    ]
    phase2_pairs = [
        (cid, f"{linux_merged}/chunks/exposure_population_grid_{cid}.tif",
         f"{local_merged}/chunks/exposure_population_grid_{cid}.tif")
        for cid in chunk_ids
    ]
    phase3_pairs = [
        (cid, f"{linux_merged}/chunks/flood_fraction/flood_fraction_{cid}_{rp}_{slr}.tif",
         f"{local_merged}/chunks/flood_fraction/flood_fraction_{cid}_{rp}_{slr}.tif")
        for cid in chunk_ids for rp in return_periods for slr in waterlevel_names
    ]

    # Dispatch mode (2026-10-09). Per chunk (one phase): each batch gets whole
    # chunks and Snakemake runs every chunk's merge -> exposure grid -> flood
    # fraction chain itself - no all-of-phase-N barrier, which on the ssp245
    # deltas run cost ~2.5 h of SLURM queueing out of a 4.3 h wall clock for
    # ~1 h of compute. Same rules, same inputs: outputs are identical to the
    # three-phase dispatch. Parallelism is capped at the number of chunks,
    # though, so a study with fewer chunks than n_nodes keeps the three-phase
    # dispatch (it can spread one chunk's merges over several nodes).
    per_chunk = len(chunk_ids) >= n_nodes
    # merge cost scales with the number of tiles per chunk - LPT weight
    chunk_weight = {
        row.chunk_id: 1.0 + float(tile_gdf.geometry.intersects(row.geometry).sum())
        for row in chunk_grid.itertuples()
    }

    if args.resume:
        # One glob per flat output directory, not one exists() call per
        # target (77,000+ individually would each be a separate network
        # round-trip over P:\ - confirmed live 2026-08-16, >60s and still
        # not done) - postprocessing's chunk outputs all live in a handful
        # of FLAT directories (see check_postprocess_progress.py's own
        # docstring on this), so a single directory listing gives every
        # existing filename at once, then membership is a cheap in-memory
        # set lookup.
        chunks_dir_local = Path(local_merged) / "chunks"
        flood_frac_dir_local = chunks_dir_local / "flood_fraction"
        existing_merge = {p.name for p in chunks_dir_local.glob("waterdepth_*.tif")} if chunks_dir_local.is_dir() else set()
        existing_grid = {p.name for p in chunks_dir_local.glob("exposure_population_grid_*.tif")} if chunks_dir_local.is_dir() else set()
        existing_ff = {p.name for p in flood_frac_dir_local.glob("flood_fraction_*.tif")} if flood_frac_dir_local.is_dir() else set()

        phase1_sel = [p for p in phase1_pairs if Path(p[2]).name not in existing_merge]
        phase2_sel = [p for p in phase2_pairs if Path(p[2]).name not in existing_grid]
        phase3_sel = [p for p in phase3_pairs if Path(p[2]).name not in existing_ff]
        print(f"Resume: phase 1 {len(phase1_sel)}/{len(phase1_pairs)} remaining, "
              f"phase 2 {len(phase2_sel)}/{len(phase2_pairs)} remaining, "
              f"phase 3 {len(phase3_sel)}/{len(phase3_pairs)} remaining\n")
        name_prefix = "resume_postprocess_"
        submit_name = "submit_resume_postprocess_and_exposure.sh"
    else:
        phase1_sel, phase2_sel, phase3_sel = phase1_pairs, phase2_pairs, phase3_pairs
        name_prefix = "postprocess_"
        submit_name = "submit_postprocess_and_exposure.sh"

    if per_chunk:
        # One phase: per chunk, the still-needed flood-fraction targets (which
        # pull in that chunk's merges and exposure grid as Snakemake inputs),
        # plus - in resume mode - any merge/grid output missing on its own
        # (e.g. a merged waterdepth deleted after its flood fraction exists).
        # A fresh dispatch only needs the flood-fraction targets: every
        # (chunk, rp, slr) merge output is an input of exactly one of them.
        targets_by_chunk: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for sel in ([phase3_sel] if not args.resume else [phase1_sel, phase2_sel, phase3_sel]):
            for cid, lx, _ in sel:
                targets_by_chunk[cid].append(lx)
        n_targets = sum(len(v) for v in targets_by_chunk.values())
        print(f"Per-chunk dispatch ({len(chunk_ids)} chunk(s) >= n_nodes={n_nodes}): "
              f"{n_targets} target(s), merge -> exposure grid -> flood fraction per chunk, no phase barriers")
        chunk_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}chunks", _split_by_chunk(targets_by_chunk, chunk_weight, n_nodes), linux_resolved_config,
        ) if n_targets else []
        phases = [("postprocessing (per chunk: merge -> exposure grid -> flood fraction)", chunk_scripts)]
    else:
        print(f"Three-phase dispatch ({len(chunk_ids)} chunk(s) < n_nodes={n_nodes} - spreads merges over more nodes)")
        phase1_targets = [lx for _, lx, _ in phase1_sel]
        phase2_targets = [lx for _, lx, _ in phase2_sel]
        phase3_targets = [lx for _, lx, _ in phase3_sel]

        # Phase 1: merge_chunk - every (chunk, rp, slr). Requesting the
        # waterdepth output also produces this rule's other declared output
        # (provenance) in the same job.
        print(f"Phase 1 (merge_chunk): {len(phase1_targets)} target(s)")
        phase1_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}merge", _split_evenly(phase1_targets, n_nodes), linux_resolved_config,
        ) if phase1_targets else []

        # Phase 2: prepare_exposure_grid_chunk - every chunk (not rp/slr).
        # Depends on phase 1 (specifically each chunk's own (return_periods[0],
        # baseline_slr) merge output, for grid metadata only - gated on ALL of
        # phase 1 via afterany rather than tracked per-chunk, same
        # simplification generate_hpc_preprocess_job.py's own phase-barriers use).
        print(f"\nPhase 2 (prepare_exposure_grid_chunk): {len(phase2_targets)} target(s) "
              f"(reference scenario: {return_periods[0]}_{baseline_slr})")
        phase2_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}exposure_grid", _split_evenly(phase2_targets, n_nodes), linux_resolved_config,
        ) if phase2_targets else []

        # Phase 3: compute_flood_fraction_chunk - every (chunk, rp, slr).
        # Depends on phase 2 (population grid) via afterany.
        print(f"\nPhase 3 (compute_flood_fraction_chunk): {len(phase3_targets)} target(s)")
        phase3_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}flood_fraction", _split_evenly(phase3_targets, n_nodes), linux_resolved_config,
        ) if phase3_targets else []
        phases = [
            ("phase 1 (merge_chunk)", phase1_scripts),
            ("phase 2 (prepare_exposure_grid_chunk)", phase2_scripts),
            ("phase 3 (compute_flood_fraction_chunk)", phase3_scripts),
        ]

    # Master driver: each phase's batches (parallel), gated on ALL of the
    # previous phase via afterany (three-phase mode); per-chunk mode is a
    # single phase. Same afterany-join-multiple-jobs pattern already used for
    # preprocessing's build_shared_inputs -> batches and the exposure
    # dispatch's pass1 -> reduce_shares -> pass2 -> reduce_write. A phase
    # with ZERO batches (resume mode, already 100% done) is skipped
    # entirely - PREV_IDS then correctly carries forward from the last
    # phase that DID have batches, same "skip empty phase" pattern
    # generate_aqueduct_jobs.generate_resume_dispatch already uses.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", "", 'PREV_IDS=""']
    for phase_label, scripts in phases:
        if not scripts:
            continue
        submit_lines.append(f'\n# {phase_label} ({len(scripts)} batch(es))')
        submit_lines.append('IDS=""')
        for script in scripts:
            submit_lines += [
                'if [ -z "$PREV_IDS" ]; then',
                f'  JID=$(sbatch --parsable "{script}")',
                "else",
                f'  JID=$(sbatch --parsable --dependency=afterany:$PREV_IDS "{script}")',
                "fi",
                f'echo "{phase_label}: submitted {script} -> job $JID"',
                'IDS="${IDS:+$IDS:}$JID"',
            ]
        submit_lines.append('PREV_IDS="$IDS"')

    # Append the exposure-analysis dispatch onto the SAME submission,
    # continuing the afterany chain from postprocessing's own job IDs
    # ($PREV_IDS, empty string if every postprocessing phase was already
    # done) rather than writing a separate script - see
    # generate_exposure_jobs.py's own module docstring for why this stays
    # one synchronous, upfront submission. In resume mode, use the
    # matching exposure resume/fresh path depending on whether exposure
    # batches were ever generated before (batch_*_chunks.txt existing under
    # hpc_jobs/exposure/) - if postprocessing itself is what needed
    # resuming, exposure analysis may never have started at all yet, which
    # needs a FRESH exposure dispatch, not a resume of nothing.
    print()
    if args.resume and any((local_jobs_dir / "exposure").glob("batch_*_chunks.txt")):
        submit_lines = generate_exposure_resume_dispatch(
            local_config, linux_config, submit_lines, prev_ids_expr="$PREV_IDS",
        )
    else:
        submit_lines = generate_exposure_dispatch(
            local_config, linux_config, chunk_ids, submit_lines, prev_ids_expr="$PREV_IDS",
        )

    submit_script_path = local_jobs_dir / submit_name
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_total_scripts = sum(len(s) for _, s in phases)
    print(f"\nDone. {len(phases)} postprocessing phase(s) + exposure analysis, {n_total_scripts}+ sbatch script(s) "
          f"written to {local_jobs_dir}")
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/{submit_name}")
    print("(This one call submits the ENTIRE remaining pipeline - postprocessing then exposure "
          "analysis - all at once; SLURM's own --dependency=afterany chain handles the timing.)")


if __name__ == "__main__":
    main()
