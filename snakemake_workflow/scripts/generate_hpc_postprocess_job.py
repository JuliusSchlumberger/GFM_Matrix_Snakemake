"""Generate HPC sbatch scripts for postprocessing across hpc.n_nodes parallel nodes.

Rules: merge_chunk -> prepare_exposure_grid_chunk -> compute_flood_fraction_chunk,
then the exposure-analysis dispatch and flood-totals/plots job appended to the
same submit script (one synchronous submission from the login node).

With at least as many chunks as nodes, whole chunks are assigned per batch (LPT by
tile count) and Snakemake chains each chunk itself; otherwise three phases, each
gated on the whole previous one via --dependency=afterany. Outputs are identical.
Chunk IDs come from tiles.build_chunk_grid, as in the Snakefile. The tile grid is
read via the local config; embedded paths use the Linux view (config_hpc.yml).

Usage:
    python generate_hpc_postprocess_job.py [--config path/to/config.yml]
    bash <printed submit_postprocess.sh path>
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402
from hpc_common import _retry_wrapper_lines, _stage_configfile_lines, _stage_tile_grid_lines, sbatch_header  # noqa: E402
from generate_hpc_exposure_jobs import generate_exposure_dispatch, generate_exposure_resume_dispatch  # noqa: E402
from generate_hpc_flood_totals_and_plots_job import write_flood_totals_and_plots_job  # noqa: E402
from tiles import build_chunk_grid  # noqa: E402


def _stage_file_locally(src_path: str) -> str:
    # Node-local copy: reading off P: from a Hydrax compute node can hang on the
    # GDAL/SQLite lock. Only this read is redirected; linux_config's
    # tile_grid.path must stay the real path since downstream scripts embed it.
    stage_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gfm_staged_inputs")
    os.makedirs(stage_dir, exist_ok=True)
    dst = os.path.join(stage_dir, f"{os.getpid()}_{os.path.basename(src_path)}")
    retry_transient_io(shutil.copy2, src_path, dst)
    return dst



def _split_evenly(targets: list[str], n_nodes: int) -> list[list[str]]:
    """Split `targets` into up to n_nodes contiguous, near-equal batches."""
    n_batches = min(n_nodes, len(targets))
    k, m = divmod(len(targets), n_batches)
    return [targets[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n_batches)]


def _split_by_chunk(
    targets_by_chunk: dict[str, list[str]], chunk_weight: dict[str, float], n_nodes: int,
) -> list[list[str]]:
    # Whole chunks per batch, LPT; no two batches build the same exposure grid.
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
    phase_name: str, batches: list[list[str]], configfile_path: str, tile_grid_path: str,
) -> list[str]:
    """Write one `GFM_CONFIG_PATH=<configfile_path> snakemake` sbatch per batch.

    Returns their Linux paths. configfile_path (resolved_config.yml) makes the
    compute node's Snakefile parse see this dispatch's config.
    """
    script_paths = []
    for i, batch_targets in enumerate(batches):
        name = f"postprocess_{phase_name}_batch_{i:03d}"

        targets_path = local_jobs_dir / f"{name}_targets.txt"
        with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch_targets) + "\n")

        lines = [
            *sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs"),
            *_retry_wrapper_lines(),
            *_stage_configfile_lines(),
            *_stage_tile_grid_lines(),
            "",
            f'cd "{linux_code_root}"',
            f'echo "=== Postprocessing {phase_name} batch {i:03d}: {len(batch_targets)} target(s) ==="',
            f'stage_configfile_locally "{configfile_path}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{tile_grid_path}" LOCAL_TILE_GRID || exit 1',
            "python -c 'import sys, yaml; cfg_path, new_path = sys.argv[1], sys.argv[2]; "
            "cfg = yaml.safe_load(open(cfg_path, encoding=\"utf-8\")); cfg[\"tile_grid\"][\"path\"] = new_path; "
            "yaml.safe_dump(cfg, open(cfg_path, \"w\", encoding=\"utf-8\"))' "
            '"$LOCAL_CONFIGFILE" "$LOCAL_TILE_GRID"',
            "",
            (
                f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry '
                f'snakemake --cores {sbatch_cfg["cpus_per_task"]} --nolock '
                # --rerun-incomplete: redo outputs a killed earlier job left flagged
                # incomplete instead of aborting the batch.
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
    """Write the postprocessing + exposure (+ flood totals/plots) dispatch scripts."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--resume", action="store_true",
        help="rebalance only the still-missing targets across the full node budget per phase "
             "(skipping any phase that's already 100%% done), instead of a fresh dispatch. "
             "Cancel the original dispatch's still-running jobs FIRST - see "
             "generate_hpc_simulation_jobs.py's own --resume docstring for why (same reasoning: "
             "a stale batch racing a new one over the same still-incomplete targets would redo "
             "work concurrently for as long as both stay alive).",
    )
    parser.add_argument(
        "--no-flood-totals-and-plots", action="store_true",
        help="don't append the flood_totals_and_plots phase (compute_flood_totals.py + "
             "run_analysis.py --only-plots) onto the end of this chain - by default it IS "
             "appended, gated --dependency=afterany on exposure analysis's own "
             "final reduce_write job, so the full postprocessing -> exposure -> flood-totals -> "
             "plots pipeline is one single submission from the login node.",
    )
    parser.add_argument("--flood-totals-partition", default="24vcpu")
    parser.add_argument("--flood-totals-cpus", type=int, default=24)
    parser.add_argument("--flood-totals-mem", default="32G")
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

    # Only written if missing (standalone use); atomic_write because another
    # node reading over P: right after a non-atomic write can see a partial file.
    resolved_config_path = local_jobs_dir / "resolved_config.yml"
    if not resolved_config_path.exists():
        atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    # Local view for the tile-grid read; Linux view for embedded target paths.
    _local_tile_grid_path = _stage_file_locally(local_config["tile_grid"]["path"])
    try:
        tile_gdf = retry_transient_io(gpd.read_file, _local_tile_grid_path)
    finally:
        try:
            os.remove(_local_tile_grid_path)
        except OSError:
            pass
    return_periods = [f"RP{rp}" for rp in local_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(local_config["boundary_conditions"], local_config["adaptation"])
    baseline_slr = local_config["protection"]["baseline_waterlevel_name"]

    chunk_size_deg = local_config["postprocessing"]["chunk_size_deg"]
    chunk_grid = build_chunk_grid(tile_gdf, chunk_size_deg)
    chunk_ids = chunk_grid["chunk_id"].tolist()
    print(f"{len(chunk_ids)} populated chunk(s) (chunk_size_deg={chunk_size_deg}), "
          f"{len(return_periods)} RPs, {len(waterlevel_names)} SLRs, n_nodes={n_nodes}\n")

    linux_merged = linux_config["postprocessing"]["merged_outputs"]
    local_merged = local_config["postprocessing"]["merged_outputs"]

    # (chunk_id, linux_target, local_target): resume filters on local existence.
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

    # Per-chunk mode avoids all-of-phase barriers (~2.5 h queueing of 4.3 h wall
    # clock on ssp245 deltas) but parallelism is capped at the chunk count.
    per_chunk = len(chunk_ids) >= n_nodes
    # merge cost scales with the number of tiles per chunk - LPT weight
    chunk_weight = {
        row.chunk_id: 1.0 + float(tile_gdf.geometry.intersects(row.geometry).sum())
        for row in chunk_grid.itertuples()
    }

    if args.resume:
        # One glob per flat output directory: per-target exists() over P: is a
        # network round-trip each (77,000+ targets).
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
        # Per chunk: the needed flood-fraction targets (pulling in merges and the
        # exposure grid), plus in resume mode any merge/grid output missing alone.
        targets_by_chunk: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for sel in ([phase3_sel] if not args.resume else [phase1_sel, phase2_sel, phase3_sel]):
            for cid, lx, _ in sel:
                targets_by_chunk[cid].append(lx)
        n_targets = sum(len(v) for v in targets_by_chunk.values())
        print(f"Per-chunk dispatch ({len(chunk_ids)} chunk(s) >= n_nodes={n_nodes}): "
              f"{n_targets} target(s), merge -> exposure grid -> flood fraction per chunk, no phase barriers")
        chunk_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}chunks", _split_by_chunk(targets_by_chunk, chunk_weight, n_nodes), linux_resolved_config, linux_config["tile_grid"]["path"],
        ) if n_targets else []
        phases = [("postprocessing (per chunk: merge -> exposure grid -> flood fraction)", chunk_scripts)]
    else:
        print(f"Three-phase dispatch ({len(chunk_ids)} chunk(s) < n_nodes={n_nodes} - spreads merges over more nodes)")
        phase1_targets = [lx for _, lx, _ in phase1_sel]
        phase2_targets = [lx for _, lx, _ in phase2_sel]
        phase3_targets = [lx for _, lx, _ in phase3_sel]

        # Requesting waterdepth also produces provenance in the same job.
        print(f"Phase 1 (merge_chunk): {len(phase1_targets)} target(s)")
        phase1_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}merge", _split_evenly(phase1_targets, n_nodes), linux_resolved_config, linux_config["tile_grid"]["path"],
        ) if phase1_targets else []

        # Needs each chunk's (return_periods[0], baseline_slr) merge for grid metadata.
        print(f"\nPhase 2 (prepare_exposure_grid_chunk): {len(phase2_targets)} target(s) "
              f"(reference scenario: {return_periods[0]}_{baseline_slr})")
        phase2_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}exposure_grid", _split_evenly(phase2_targets, n_nodes), linux_resolved_config, linux_config["tile_grid"]["path"],
        ) if phase2_targets else []

        print(f"\nPhase 3 (compute_flood_fraction_chunk): {len(phase3_targets)} target(s)")
        phase3_scripts = _write_batches(
            local_jobs_dir, linux_jobs_dir, linux_code_root, sbatch_cfg,
            f"{name_prefix}flood_fraction", _split_evenly(phase3_targets, n_nodes), linux_resolved_config, linux_config["tile_grid"]["path"],
        ) if phase3_targets else []
        phases = [
            ("phase 1 (merge_chunk)", phase1_scripts),
            ("phase 2 (prepare_exposure_grid_chunk)", phase2_scripts),
            ("phase 3 (compute_flood_fraction_chunk)", phase3_scripts),
        ]

    # Phases with zero batches (resume, done) are skipped; PREV_IDS carries over.
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

    # Resume mode needs a FRESH exposure dispatch if no batch_*_chunks.txt exists.
    print()
    if args.resume and any((local_jobs_dir / "exposure").glob("batch_*_chunks.txt")):
        submit_lines = generate_exposure_resume_dispatch(
            local_config, linux_config, submit_lines, prev_ids_expr="$PREV_IDS",
        )
    else:
        submit_lines = generate_exposure_dispatch(
            local_config, linux_config, chunk_ids, submit_lines, prev_ids_expr="$PREV_IDS",
        )

    # Final phase, gated on exposure's last job ($JID = reduce_write).
    if not args.no_flood_totals_and_plots:
        _, linux_ftp_script = write_flood_totals_and_plots_job(
            local_config, linux_config,
            partition=args.flood_totals_partition, cpus=args.flood_totals_cpus, mem=args.flood_totals_mem,
        )
        submit_lines.append("\n# flood totals + plots (final phase)")
        submit_lines += [
            f'JID=$(sbatch --parsable --dependency=afterany:$JID "{linux_ftp_script}")',
            f'echo "submitted {linux_ftp_script} -> job $JID (depends on exposure analysis\'s reduce_write)"',
        ]

    submit_script_path = local_jobs_dir / submit_name
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_total_scripts = sum(len(s) for _, s in phases)
    tail = "" if args.no_flood_totals_and_plots else " -> flood totals -> plots"
    print(f"\nDone. {len(phases)} postprocessing phase(s) + exposure analysis{tail}, "
          f"{n_total_scripts}+ sbatch script(s) written to {local_jobs_dir}")
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/{submit_name}")
    print(f"(This one call submits the ENTIRE remaining pipeline - postprocessing -> exposure "
          f"analysis{tail} - all at once; SLURM's own --dependency=afterany chain handles the timing.)")


if __name__ == "__main__":
    main()
