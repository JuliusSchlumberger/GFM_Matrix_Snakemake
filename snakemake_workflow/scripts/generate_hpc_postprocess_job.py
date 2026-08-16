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
from shapely.geometry import box as shapely_box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402
from generate_exposure_jobs import generate_exposure_dispatch, generate_exposure_resume_dispatch  # noqa: E402


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


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


def _write_batches(
    local_jobs_dir: Path, linux_jobs_dir: str, linux_code_root: str, sbatch_cfg: dict,
    phase_name: str, targets: list[str], n_nodes: int,
) -> list[str]:
    """Split `targets` evenly across up to n_nodes batches, write one sbatch
    script per batch (plain `snakemake --cores N <targets>` call, matching
    generate_hpc_preprocess_job.py's own pattern), return their Linux paths.
    """
    n_batches = min(n_nodes, len(targets))
    k, m = divmod(len(targets), n_batches)
    script_paths = []
    for i in range(n_batches):
        batch_targets = targets[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
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
            f'cd "{linux_code_root}"',
            f'echo "=== Postprocessing {phase_name} batch {i:03d}: {len(batch_targets)} target(s) ==="',
            "",
            (
                f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                f'--rerun-triggers=mtime $(cat "{linux_jobs_dir}/{name}_targets.txt")'
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

    # (linux_target, local_target) pairs, index-aligned by construction -
    # resume mode filters on local existence, fresh mode uses every linux
    # target unconditionally. Avoids needing a separate linux->local path
    # translation helper.
    phase1_pairs = [
        (f"{linux_merged}/chunks/waterdepth_{cid}_{rp}_{slr}.tif",
         f"{local_merged}/chunks/waterdepth_{cid}_{rp}_{slr}.tif")
        for cid in chunk_ids for rp in return_periods for slr in waterlevel_names
    ]
    phase2_pairs = [
        (f"{linux_merged}/chunks/exposure_population_grid_{cid}.tif",
         f"{local_merged}/chunks/exposure_population_grid_{cid}.tif")
        for cid in chunk_ids
    ]
    phase3_pairs = [
        (f"{linux_merged}/chunks/flood_fraction/flood_fraction_{cid}_{rp}_{slr}.tif",
         f"{local_merged}/chunks/flood_fraction/flood_fraction_{cid}_{rp}_{slr}.tif")
        for cid in chunk_ids for rp in return_periods for slr in waterlevel_names
    ]

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

        phase1_targets = [lx for lx, lo in phase1_pairs if Path(lo).name not in existing_merge]
        phase2_targets = [lx for lx, lo in phase2_pairs if Path(lo).name not in existing_grid]
        phase3_targets = [lx for lx, lo in phase3_pairs if Path(lo).name not in existing_ff]
        print(f"Resume: phase 1 {len(phase1_targets)}/{len(phase1_pairs)} remaining, "
              f"phase 2 {len(phase2_targets)}/{len(phase2_pairs)} remaining, "
              f"phase 3 {len(phase3_targets)}/{len(phase3_pairs)} remaining\n")
        name_prefix = "resume_postprocess_"
        submit_name = "submit_resume_postprocess_and_exposure.sh"
    else:
        phase1_targets = [lx for lx, _ in phase1_pairs]
        phase2_targets = [lx for lx, _ in phase2_pairs]
        phase3_targets = [lx for lx, _ in phase3_pairs]
        name_prefix = "postprocess_"
        submit_name = "submit_postprocess_and_exposure.sh"

    # Phase 1: merge_chunk - every (chunk, rp, slr). Requesting the
    # waterdepth output also produces this rule's other declared outputs
    # (provenance, overlap_minmax) in the same job.
    print(f"Phase 1 (merge_chunk): {len(phase1_targets)} target(s)")
    phase1_scripts = _write_batches(
        local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
        f"{name_prefix}merge", phase1_targets, n_nodes,
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
        f"{name_prefix}exposure_grid", phase2_targets, n_nodes,
    ) if phase2_targets else []

    # Phase 3: compute_flood_fraction_chunk - every (chunk, rp, slr).
    # Depends on phase 2 (population grid) via afterany.
    print(f"\nPhase 3 (compute_flood_fraction_chunk): {len(phase3_targets)} target(s)")
    phase3_scripts = _write_batches(
        local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
        f"{name_prefix}flood_fraction", phase3_targets, n_nodes,
    ) if phase3_targets else []

    # Master driver: phase 1 batches (parallel, no dependency) -> phase 2
    # batches (afterany: all phase 1) -> phase 3 batches (afterany: all
    # phase 2). Same afterany-join-multiple-jobs pattern already used for
    # preprocessing's build_shared_inputs -> batches and the exposure
    # dispatch's pass1 -> reduce_shares -> pass2 -> reduce_write. A phase
    # with ZERO batches (resume mode, already 100% done) is skipped
    # entirely - PREV_IDS then correctly carries forward from the last
    # phase that DID have batches, same "skip empty phase" pattern
    # generate_aqueduct_jobs.generate_resume_dispatch already uses.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", "", 'PREV_IDS=""']
    for phase_label, scripts in [
        ("phase 1 (merge_chunk)", phase1_scripts),
        ("phase 2 (prepare_exposure_grid_chunk)", phase2_scripts),
        ("phase 3 (compute_flood_fraction_chunk)", phase3_scripts),
    ]:
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

    n_total_scripts = len(phase1_scripts) + len(phase2_scripts) + len(phase3_scripts)
    print(f"\nDone. 3 postprocessing phases + exposure analysis, {n_total_scripts}+ sbatch script(s) "
          f"written to {local_jobs_dir}")
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/{submit_name}")
    print("(This one call submits the ENTIRE remaining pipeline - postprocessing then exposure "
          "analysis - all at once; SLURM's own --dependency=afterany chain handles the timing.)")


if __name__ == "__main__":
    main()
