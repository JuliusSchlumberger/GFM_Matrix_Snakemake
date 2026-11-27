"""Write the HPC postprocessing chain across hpc.n_nodes parallel nodes; entry point `generate()`.

Rules: merge_chunk -> prepare_exposure_grid_chunk -> compute_flood_fraction_chunk, then the
exposure analysis and the flood-totals/plots job, all in one submit script.

With at least as many chunks as nodes, whole chunks are packed per batch (LPT by tile count) and
Snakemake chains each chunk itself; otherwise three phases, each gated on the whole previous one
(afterany). Outputs are identical. Chunk IDs come from tiles.build_chunk_grid, as in the
Snakefile. The tile grid is read via the local config; embedded paths use config_hpc.yml.

`postprocessing.merge_only` (validation scenarios) writes the merges only, no exposure or
flood totals: merge_chunk, or, with a simulation.dem_source_manifest (tiles in the local
DEM's projected CRS, which merge_chunk rejects), per (RP, SLR) postprocessing.native_merge_jobs
merge_england_local_tiles.py jobs over native-CRS chunks and one VRT job, under
merged_outputs/local_native/.

Usage:
    python generate_hpc_postprocess_job.py [--config path] [--resume] [--no-flood-totals-and-plots] [--profile-set P.F=V]
    bash <printed submit script>
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, load_hpc_config, merged_waterlevel_names, retry_transient_io  # noqa: E402
from hpc_common import (  # noqa: E402
    SNAKEMAKE,
    JobChain,
    add_profile_set_argument,
    apply_profile_sets,
    even_split,
    lpt_pack,
    read_targets_lines,
    tile_grid_override_lines,
)
from generate_hpc_exposure_jobs import add_exposure_jobs, add_exposure_resume_jobs  # noqa: E402
from generate_hpc_flood_totals_and_plots_job import add_flood_totals_job  # noqa: E402
from tiles import build_chunk_grid  # noqa: E402


def _stage_file_locally(src_path: str) -> str:
    # Node-local copy: reading off the shared data root from a cluster compute node can hang on the
    # GDAL/SQLite lock. Only this read is redirected; linux_config's
    # tile_grid.path must stay the real path since downstream scripts embed it.
    stage_dir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "gfm_staged_inputs")
    os.makedirs(stage_dir, exist_ok=True)
    dst = os.path.join(stage_dir, f"{os.getpid()}_{os.path.basename(src_path)}")
    retry_transient_io(shutil.copy2, src_path, dst)
    return dst


def _split_by_chunk(
    targets_by_chunk: dict[str, list[str]], chunk_weight: dict[str, float], n_nodes: int,
) -> list[list[str]]:
    # Whole chunks per batch; no two batches build the same exposure grid.
    chunks = [c for c in targets_by_chunk if targets_by_chunk[c]]
    buckets = lpt_pack({c: chunk_weight.get(c, 1.0) for c in chunks}, min(n_nodes, len(chunks)))
    return [[t for c in bucket for t in targets_by_chunk[c]] for bucket in buckets]


def _raise_open_files_limit_lines() -> list[str]:
    # A merge holds every overlapping tile raster open at once; a fine-tile chunk or a
    # local-native merge (~1,800 England tiles) exceeds the common soft limit of 1024.
    return ['ulimit -n "$(ulimit -Hn)" 2>/dev/null || true']


def _add_native_merge_jobs(chain: JobChain, linux_code_root: str, name_prefix: str, return_periods: list[str],
                           waterlevel_names: list[str], configfile_path: str, n_batches: int) -> list[str]:
    """Add, per (RP, SLR), `n_batches` native-chunk merge jobs and a VRT job after them; return the VRT jobs."""
    def job(name: str, args: str, after: list[str] = ()) -> str:
        chain.add(name, "postprocess", [
            f'cd "{linux_code_root}"',
            *_raise_open_files_limit_lines(),
            f'stage_configfile_locally "{configfile_path}" LOCAL_CONFIGFILE || exit 1',
            f'"$GFM_PY" preparation/merge_england_local_tiles.py --config "$LOCAL_CONFIGFILE" {args}',
        ], after=after, resources={"cpus_per_task": 1}, helpers=("stage_config",))
        return name

    vrt_jobs = []
    for rp in return_periods:
        for slr in waterlevel_names:
            event = f"--return-periods {rp} --waterlevel-name {slr}"
            merges = [job(f"{name_prefix}local_native_merge_{rp}_{slr}_batch_{k:03d}",
                          f"{event} --batch {k} --n-batches {n_batches}") for k in range(n_batches)]
            vrt_jobs.append(job(f"{name_prefix}local_native_vrt_{rp}_{slr}", f"{event} --vrt", after=merges))
    return vrt_jobs


def postprocessing_rules() -> list[str]:
    """Return the rule names defined in rules/postprocessing.smk."""
    smk = Path(__file__).resolve().parents[1] / "rules" / "postprocessing.smk"
    return re.findall(r"^rule (\w+):", smk.read_text(encoding="utf-8"), flags=re.MULTILINE)


def _add_batches(chain: JobChain, linux_code_root: str, phase_name: str, batches: list[list[str]],
                 configfile_path: str, tile_grid_path: str, after: list[str]) -> list[str]:
    """Add one `GFM_CONFIG_PATH=<staged config> snakemake` job per batch; return their names.

    The staged resolved_config.yml makes the compute node's Snakefile parse see this dispatch's config.
    Only postprocessing rules may run: a missing or outdated tile result fails the batch instead of
    being preprocessed and solved inside it.
    """
    allowed_rules = " ".join(postprocessing_rules())
    names = []
    for i, batch_targets in enumerate(batches):
        name = f"postprocess_{phase_name}_batch_{i:03d}"
        with open(chain.local_dir / f"{name}_targets.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch_targets) + "\n")
        chain.add(name, "postprocess", [
            f'cd "{linux_code_root}"',
            f'echo "=== Postprocessing {phase_name} batch {i:03d}: {len(batch_targets)} target(s) ==="',
            *_raise_open_files_limit_lines(),
            f'stage_configfile_locally "{configfile_path}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{tile_grid_path}" LOCAL_TILE_GRID || exit 1',
            *tile_grid_override_lines(),
            "",
            *read_targets_lines("TARGETS", f"{chain.linux_dir}/{name}_targets.txt"),
            # --rerun-incomplete: redo outputs a killed earlier job left flagged incomplete
            # instead of aborting the batch.
            f'GFM_CONFIG_PATH="$LOCAL_CONFIGFILE" run_snakemake_with_retry {SNAKEMAKE} --cores "$SLURM_CPUS_PER_TASK" --nolock '
            # --allowed-rules takes several values: keep another option between it and the targets.
            f'--allowed-rules {allowed_rules} --rerun-triggers=mtime --rerun-incomplete "${{TARGETS[@]}}"',
        ], after=after, helpers=("retry", "stage_config", "stage_tile_grid"))
        names.append(name)
        print(f"  {name}: {len(batch_targets)} target(s)")
    return names


def generate(config_path: str | Path, *, resume: bool = False, flood_totals: bool = True,
             profile_sets: list[str] | None = None) -> str:
    """Write the postprocessing + exposure (+ flood totals/plots) chain; return its submit script's Linux path."""
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    hpc_cfg = apply_profile_sets(linux_config["hpc"], profile_sets)

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = hpc_cfg["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    n_nodes = hpc_cfg["n_nodes"]

    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)
    # Rewritten every time: a different config may have left its own. atomic_write because another
    # node reading over the shared mount right after a non-atomic write can see a partial file.
    atomic_write(local_jobs_dir / "resolved_config.yml", lambda f: yaml.safe_dump(linux_config, f),
                 encoding="utf-8", newline="")
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"
    linux_tile_grid = linux_config["tile_grid"]["path"]

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
    waterlevel_names = merged_waterlevel_names(local_config["boundary_conditions"], local_config["adaptation"])
    baseline_slr = local_config["protection"]["baseline_waterlevel_name"]
    merge_only = local_config["postprocessing"]["merge_only"]
    native_merge = bool(local_config["simulation"].get("dem_source_manifest"))
    if native_merge and not merge_only:
        raise ValueError(
            "simulation.dem_source_manifest is set but postprocessing.merge_only is not: the exposure "
            "chain needs the EPSG:4326 chunks, which merge_chunk cannot build from projected local tiles"
        )

    chunk_size_deg = local_config["postprocessing"]["chunk_size_deg"]
    chunk_grid = build_chunk_grid(tile_gdf, chunk_size_deg)
    chunk_ids = chunk_grid["chunk_id"].tolist()
    print(f"{len(chunk_ids)} populated chunk(s) (chunk_size_deg={chunk_size_deg}), "
          f"{len(return_periods)} RPs, {len(waterlevel_names)} SLRs, n_nodes={n_nodes}")

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

    if resume:
        # One glob per flat output directory: per-target exists() over the shared mount is a
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
              f"phase 3 {len(phase3_sel)}/{len(phase3_pairs)} remaining")
        name_prefix = "resume_postprocess_"
        submit_name = "submit_resume_postprocess_and_exposure.sh"
    else:
        phase1_sel, phase2_sel, phase3_sel = phase1_pairs, phase2_pairs, phase3_pairs
        name_prefix = "postprocess_"
        submit_name = "submit_postprocess_and_exposure.sh"

    chain = JobChain(hpc_cfg, local_jobs_dir, linux_jobs_dir)
    last: list[str] = []  # the most recent non-empty phase: the next one waits for all of it
    if native_merge:
        n_batches = int(local_config["postprocessing"]["native_merge_jobs"])
        n_events = len(return_periods) * len(waterlevel_names)
        print(f"Local-native merge: {n_events} event(s) x {n_batches} chunk batch(es) + one VRT job each "
              f"(native_chunk_m={local_config['postprocessing']['native_chunk_m']})")
        last = _add_native_merge_jobs(chain, linux_code_root, name_prefix, return_periods, waterlevel_names,
                                      linux_resolved_config, n_batches)
    elif merge_only:
        phase1_targets = [lx for _, lx, _ in phase1_sel]
        print(f"Merge only (merge_chunk): {len(phase1_targets)} target(s)")
        last = _add_batches(chain, linux_code_root, f"{name_prefix}merge", even_split(phase1_targets, n_nodes),
                            linux_resolved_config, linux_tile_grid, after=[])
    elif per_chunk:
        # Per chunk: the needed flood-fraction targets (pulling in merges and the
        # exposure grid), plus in resume mode any merge/grid output missing alone.
        targets_by_chunk: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for sel in ([phase3_sel] if not resume else [phase1_sel, phase2_sel, phase3_sel]):
            for cid, lx, _ in sel:
                targets_by_chunk[cid].append(lx)
        n_targets = sum(len(v) for v in targets_by_chunk.values())
        print(f"Per-chunk dispatch ({len(chunk_ids)} chunk(s) >= n_nodes={n_nodes}): {n_targets} target(s), "
              "merge -> exposure grid -> flood fraction per chunk, no phase barriers")
        if n_targets:
            last = _add_batches(chain, linux_code_root, f"{name_prefix}chunks",
                                _split_by_chunk(targets_by_chunk, chunk_weight, n_nodes),
                                linux_resolved_config, linux_tile_grid, after=[])
    else:
        print(f"Three-phase dispatch ({len(chunk_ids)} chunk(s) < n_nodes={n_nodes} - spreads merges over more nodes)")
        # Requesting waterdepth also produces provenance in the same job. Phase 2 needs each
        # chunk's (return_periods[0], baseline_slr) merge for its grid metadata.
        print(f"Phase 2 reference scenario: {return_periods[0]}_{baseline_slr}")
        for phase, sel in (("merge", phase1_sel), ("exposure_grid", phase2_sel), ("flood_fraction", phase3_sel)):
            targets = [lx for _, lx, _ in sel]
            if targets:
                last = _add_batches(chain, linux_code_root, f"{name_prefix}{phase}", even_split(targets, n_nodes),
                                    linux_resolved_config, linux_tile_grid, after=last)

    if merge_only:
        print("merge_only: no exposure analysis, flood totals or plots")
    else:
        # Resume needs a fresh exposure dispatch if no batch_*_chunks.txt exists.
        if resume and any((local_jobs_dir / "exposure").glob("batch_*_chunks.txt")):
            exposure_last = add_exposure_resume_jobs(chain, local_config, linux_config, after=last)
        else:
            exposure_last = add_exposure_jobs(chain, local_config, linux_config, chunk_ids, after=last)
        if flood_totals:
            add_flood_totals_job(chain, linux_config, after=[exposure_last])

    submit = chain.write_submit(submit_name)
    print(f"Done: {len(chain.jobs)} job(s) -> {local_jobs_dir}")
    return submit


def main() -> None:
    # allow_abbrev=False: a removed flag that is a prefix of its replacement must be an error.
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False,
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"))
    parser.add_argument(
        "--resume", action="store_true",
        help="only the still-missing targets per phase, rebalanced over the node budget; cancel the "
             "original dispatch's still-running jobs first",
    )
    parser.add_argument("--no-flood-totals-and-plots", action="store_true",
                        help="end the chain after the exposure analysis")
    add_profile_set_argument(parser)
    args = parser.parse_args()
    submit = generate(args.config, resume=args.resume, flood_totals=not args.no_flood_totals_and_plots,
                      profile_sets=args.profile_set)
    print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
