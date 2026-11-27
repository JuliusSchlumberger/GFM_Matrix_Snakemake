"""Write the SFINCS calibration's HPC chain (calibration_roughness_SFINCS); entry point `run()`.

stage1: candidate shards + per-shard screen jobs (line mode: + boundary line and pre-flight) and a
bridge job (afterany) that selects tiles and runs `stage2 --submit`.
--boundary-mode (coast | line) is the SFINCS boundary of run_final_tile.sh; it also sets which
tiles are eligible (coast: every screened tile; line: a suitable line and a passed pre-flight).
stage2: spreads unfinished tiles over jobs, longest processing time first, plus the post jobs
(afterany): --post-shards jobs for the tile drivers and the dedup, then the raw and the dedup
analysis (variant_selection.json, figures) as two jobs.
post: the post jobs of a phase alone (--no-dedup: reuse the existing dedup results; --rescore:
rewrite the raw calibration results from the rasters first).
rerun: resets a finished calibration's tiles in place (`--part models`: one phase of the flood-model
runs; `--part sfincs`: forcing, SFINCS model build and run, then the sweep) and writes stage 2.

Usage (normally through hpc/gfm_hpc.py sfincs ...):
    python generate_calibration_jobs.py stage1 [--submit]
    python generate_calibration_jobs.py stage2 --submit   # what the bridge job runs
    python generate_calibration_jobs.py rerun --part models --phase duration [--reset-runs] [--dry-run] [--submit]

Job resources: the sfincs_* profiles of config_hpc.yml (--profile-set to override).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))
from gfm_config import read_root  # noqa: E402

# Eikonal solves per tile, in units of the 10-solve sweep of eikonal_min_per_mcell, by phase
# (run_calibration_models.phase_runs; a duration run solves twice, bathtubs ~free).
RUN_LOAD = {"sweep": 1.8, "duration": 2.0}
DONE_MARKER = "calibration_run.done"
TILE_MARKERS = ("calibration_run.done", "calibration_run.failed", "calibration_run.step")
FALLBACK_MCELLS = 2.0  # tile without a pre-flight record
POST_SHARDS = 4  # post jobs sharing the tile drivers and the dedup (I/O-bound: one node each)
DEFAULT_CONFIG = _HERE.parent / "snakemake_workflow" / "config" / "config.yml"


def _hpc_view(config: str | Path, profile_sets: list[str] | None) -> dict:
    from config_utils import load_hpc_config
    from hpc_common import apply_profile_sets

    linux_config = load_hpc_config(config)
    apply_profile_sets(linux_config["hpc"], profile_sets)
    return linux_config


def _slurm_hours(hours: int) -> str:
    return f"{hours // 24}-{hours % 24:02d}:00:00"


def _chain(linux_config: dict, local_dir: Path, linux_dir: str):
    from hpc_common import JobChain

    (local_dir / "logs").mkdir(parents=True, exist_ok=True)
    return JobChain(linux_config["hpc"], local_dir, linux_dir)


def _submit(submit: str) -> None:
    from hpc_common import submit_script

    submit_script(submit)


def cmd_stage1(args) -> str:
    """Write the stage-1 jobs, the bridge job and submit_chain.sh; return the submit script's Linux path.

    Side effects: writes {base}/resolved_config.yml (Linux path view) and runs
    ``select_calibration_tiles.py pool``.
    """
    import yaml

    config_path = Path(args.config)
    linux_config = _hpc_view(config_path, args.profile_set)
    base = read_root(config_path) / args.base_dir_name
    base.mkdir(parents=True, exist_ok=True)
    (base / "resolved_config.yml").write_text(yaml.safe_dump(linux_config), encoding="utf-8", newline="\n")

    subprocess.run([sys.executable, str(_HERE / "select_calibration_tiles.py"), "pool",
                    "--config", str(config_path), "--base-dir-name", args.base_dir_name,
                    "--n-shards", str(args.n_jobs)], check=True)

    code = f"{linux_config['paths']['code_root']}/sfincs_tiles"
    lb = f"{linux_config['paths']['root']}/{args.base_dir_name}"
    common = f"--config {lb}/resolved_config.yml --base-dir-name {args.base_dir_name}"
    chain = _chain(linux_config, base / "hpc_jobs_stage1", f"{lb}/hpc_jobs_stage1")
    # No -e: a job's later steps still run after an earlier one partly failed (the pre-flight
    # after tiles errored in the line build). Steps the rest cannot do without end in `|| exit 1`.
    line = args.boundary_mode == "line"
    mode = f"--boundary-mode {args.boundary_mode}"
    for k in range(args.n_jobs):
        shard, screened = f"{lb}/candidates/shard_{k:03d}.txt", f"{lb}/candidates/screened_{k:03d}.txt"
        n = len((base / "candidates" / f"shard_{k:03d}.txt").read_text().split())
        lines = [f'"$GFM_PY" build_station_boundary_lines.py {common} --tiles-dir-name model_outputs '
                 f"--tile-ids-file {screened} --skip-existing --plot all",
                 f'"$SFINCS_PY" preflight_sfincs_mask.py {common} --tiles-dir-name model_outputs '
                 f"--tile-ids-file {screened} --skip-existing"] if line else []
        chain.add(f"cal_stage1_{k:03d}", "sfincs_stage1", [
            f'cd "{code}" || exit 1',
            f"# {n} candidate tile(s): screen" + (" -> boundary lines -> SFINCS mask pre-flight" if line else ""),
            f'"$GFM_PY" select_calibration_tiles.py screen {common} --tile-ids-file {shard} --out-ids {screened} || exit 1',
            *lines,
        ], filename=f"stage1_{k:03d}.sbatch", resources={"time": _slurm_hours(args.stage1_hours)}, errexit=False)
    chain.add("cal_bridge", "sfincs_bridge", [
        f'cd "{code}" || exit 1',
        "# after every stage-1 job: " + ("line summary -> " if line else "")
        + "tile selection + maps -> stage-2 jobs (submitted)",
        *([f'"$GFM_PY" build_station_boundary_lines.py {common} --summarize'] if line else []),
        f'"$GFM_PY" select_calibration_tiles.py select {common} {mode} --n-tiles {args.n_tiles} || exit 1',
        f'"$GFM_PY" generate_calibration_jobs.py stage2 {common} {mode} --n-jobs {args.stage2_jobs} --submit',
    ], after=chain.names(), filename="bridge.sbatch", resources={"time": _slurm_hours(2)}, errexit=False)
    submit = chain.write_submit("submit_chain.sh")
    print(f"stage 1: {args.n_jobs} job(s) + bridge -> {base / 'hpc_jobs_stage1'}")
    if args.submit:
        _submit(submit)
    return submit


def _add_post_job(chain, linux_config: dict, base_dir_name: str, dedup_dir_name: str, phase: str,
                  after: list[str], dedup: bool = True, n_shards: int = POST_SHARDS,
                  rescore: bool = False) -> list[str]:
    """Add a phase's post jobs; return their names.

    n_shards jobs each compute their share of the tile drivers and (with `dedup`) of the dedup
    results. Then, both after all shards, one job writes the progress report and the raw
    analysis (variant_selection.json, figures; it merges the driver parts) and one the dedup
    analysis, which stops when a dedup shard did not finish. `dedup` False reuses the existing
    dedup results. `rescore` first rewrites every tile's raw calibration result from its
    rasters (each shard its share).
    """
    root = linux_config["paths"]["root"]
    cfg = f"{root}/{base_dir_name}/resolved_config.yml"
    cd = f'cd "{linux_config["paths"]["code_root"]}/sfincs_tiles" || exit 1'
    shards = []
    for k in range(n_shards):
        spec, name = f"{k}/{n_shards}", f"cal_post_{phase}_shard{k}"
        chain.add(name, "sfincs_post", [
            cd,
            *([f'"$SFINCS_PY" compute_calibration_result.py --config {cfg} --base-dir-name {base_dir_name} '
               f"--shard {spec} || exit 1"] if rescore else []),
            f'"$GFM_PY" compute_tile_drivers.py --config {cfg} --base-dir-name {base_dir_name} --shard {spec}',
            *([f'"$SFINCS_PY" compute_calibration_result_dedup.py --config {cfg} --base-dir-name {base_dir_name} '
               f"--out-dir-name {dedup_dir_name} --shard {spec} || exit 1"] if dedup else []),
        ], after=after, filename=f"post_{phase}_shard{k}.sbatch", errexit=False)
        shards.append(name)
    raw = f"cal_post_{phase}"
    chain.add(raw, "sfincs_post", [
        cd,
        f'"$GFM_PY" check_calibration_progress.py --config {cfg} --base-dir-name {base_dir_name}',
        f'"$GFM_PY" run_calibration_sweep_analysis.py --config {cfg} --base-dir-name {base_dir_name}',
    ], after=shards, filename=f"post_models_{phase}.sbatch", errexit=False)
    deduped = f"cal_post_{phase}_dedup"
    guard = [
        f'n_ok=$(ls {root}/{dedup_dir_name}/dedup_shard_*of{n_shards}.ok 2>/dev/null | wc -l)',
        f'[ "$n_ok" -eq {n_shards} ] || {{ echo "dedup incomplete: $n_ok of {n_shards} shard(s) finished"; exit 1; }}',
    ] if dedup else []
    chain.add(deduped, "sfincs_post", [
        cd, *guard,
        f'"$GFM_PY" run_calibration_sweep_analysis.py --config {cfg} --base-dir-name {dedup_dir_name} '
        f"--raster-base-dir-name {base_dir_name}",
    ], after=shards, filename=f"post_models_{phase}_dedup.sbatch", errexit=False)
    return [*shards, raw, deduped]


def cmd_stage2(args) -> str | None:
    """Write the stage-2 jobs for unfinished tiles (+ post job); return the submit script's Linux path.

    Per-tile minutes = overhead_min + active Mcells (pre-flight) x
    (sfincs_min_per_mcell + eikonal_min_per_mcell x RUN_LOAD[phase]). None when every tile is done.
    """
    from hpc_common import lpt_pack

    linux_config = _hpc_view(args.config, args.profile_set)
    base = read_root(Path(args.config)) / args.base_dir_name
    ids = [t.strip() for t in (base / "tile_ids.txt").read_text().split() if t.strip()]
    est, n_done = {}, 0
    for tid in ids:
        if (base / tid / "outputs" / DONE_MARKER).exists():
            n_done += 1
            continue
        pf = base / "boundary_lines" / "preflight" / f"{tid}.json"
        mcells = json.loads(pf.read_text()).get("n_active", 0) / 1e6 if pf.exists() else FALLBACK_MCELLS
        est[tid] = args.overhead_min + mcells * (args.sfincs_min_per_mcell
                                                 + args.eikonal_min_per_mcell * RUN_LOAD[args.phase])
    if not est:
        print(f"Nothing to do: {n_done} tile(s) already done.")
        return None

    lb = f"{linux_config['paths']['root']}/{args.base_dir_name}"
    lout = f"{lb}/hpc_jobs_stage2"
    chain = _chain(linux_config, base / "hpc_jobs_stage2", lout)
    hpc = linux_config["hpc"]
    if not (hpc["container"]["cmd"] and hpc["container"]["sfincs_image"]):
        raise ValueError("hpc.container.cmd and hpc.container.sfincs_image must be set (config_hpc.yml)")
    exports = {"GFM_DATA_ROOT": linux_config["paths"]["root"], "GFM_CODE_ROOT": linux_config["paths"]["code_root"],
               "CONTAINER_CMD": hpc["container"]["cmd"], "SFINCS_IMAGE": hpc["container"]["sfincs_image"],
               "SFINCS_TIMEOUT_S": hpc["sfincs_timeout_s"]}
    loads = []
    for j, tids in enumerate(lpt_pack(est, min(args.n_jobs, len(est)))):
        load = sum(est[t] for t in tids)
        loads.append(load)
        (chain.local_dir / f"tiles_{j:03d}.txt").write_text("\n".join(tids) + "\n", newline="\n")
        hours = max(1, math.ceil(load * args.time_margin / 60.0))
        # No -e: a failing tile must not end the loop (run_final_tile.sh records it and exits 0).
        chain.add(f"cal_final_{j:03d}", "sfincs_tile", [
            f'cd "{linux_config["paths"]["code_root"]}/sfincs_tiles" || exit 1',
            f"# {len(tids)} tile(s), estimated {load / 60:.1f} h (generate_calibration_jobs.py stage2)",
            "while read -r TILE; do",
            '  [ -n "$TILE" ] || continue',
            f"  BASE_DIR_NAME={args.base_dir_name} CAL_PHASE={args.phase} SFINCS_BOUNDARY_MODE={args.boundary_mode} "
            f'bash {linux_config["paths"]["code_root"]}/sfincs_tiles/run_final_tile.sh "$TILE" </dev/null',
            f"done < {lout}/tiles_{j:03d}.txt",
        ], filename=f"final_{j:03d}.sbatch", resources={"time": _slurm_hours(hours)}, errexit=False,
            env="sfincs", exports=exports)
    if args.post:
        _add_post_job(chain, linux_config, args.base_dir_name, args.dedup_dir_name, args.phase, chain.names(),
                      n_shards=args.post_shards)
    submit = chain.write_submit("submit_all.sh")
    loads.sort()
    print(f"{len(est)} tile(s) to run ({n_done} already done) over {len(loads)} job(s), phase {args.phase}")
    print(f"estimated hours per job: {loads[0] / 60:.1f}-{loads[-1] / 60:.1f} (x{args.time_margin:g} margin for "
          f"--time; total {sum(loads) / 60:.0f} h)")
    if args.submit:
        _submit(submit)
    return submit


def cmd_post(args) -> str:
    """Write the post job of `--phase` alone and its submit script."""
    linux_config = _hpc_view(args.config, args.profile_set)
    base = read_root(Path(args.config)) / args.base_dir_name
    lout = f"{linux_config['paths']['root']}/{args.base_dir_name}/hpc_jobs_stage2"
    chain = _chain(linux_config, base / "hpc_jobs_stage2", lout)
    _add_post_job(chain, linux_config, args.base_dir_name, args.dedup_dir_name, args.phase, [], dedup=args.dedup,
                  n_shards=args.post_shards, rescore=args.rescore)
    submit = chain.write_submit(f"submit_post_{args.phase}.sh")
    if args.submit:
        _submit(submit)
    return submit


def reset_tiles(base: Path, part: str, reset_runs: bool, dry_run: bool) -> dict[str, int]:
    """Delete the per-tile markers that make run_final_tile.sh skip a tile; return counts.

    `part` models: the tile markers. `part` sfincs: the tile markers and sfincs_model/sfincs.inp
    (forcing and model are rebuilt). With `reset_runs` also every flood-model raster and result.
    Listed tiles without a directory are counted, not created.
    """
    counts = {"tiles": 0, "missing": 0, "rasters": 0}
    for t in (base / "tile_ids.txt").read_text().split():
        if not (base / t).is_dir():
            counts["missing"] += 1
            continue
        out = base / t / "outputs"
        counts["tiles"] += 1
        doomed = []
        doomed = [out / m for m in TILE_MARKERS]
        if part == "sfincs":
            doomed.append(base / t / "sfincs_model" / "sfincs.inp")
        if reset_runs and out.is_dir():
            rasters = sorted(out.glob("*_waterdepth_RP100_SLR_0*.tif"))
            counts["rasters"] += len(rasters)
            doomed += rasters + [out / f for f in ("model_runs.json", "eikonal_solver_diagnostics.json",
                                                   "calibration_result.json")]
        if not dry_run:
            for p in doomed:
                p.unlink(missing_ok=True)
    return counts


def cmd_rerun(args) -> str | None:
    """Rerun part of a finished calibration in place: reset tiles, then write (and submit) stage 2.

    Guards: tile_ids.txt exists; the checkout has the code the part needs; a non-sweep phase needs
    the sweep's variant_selection.json. `--dry-run` only reports what would be reset.
    """
    root = read_root(Path(args.config))
    base = root / args.base_dir_name
    if not (base / "tile_ids.txt").exists():
        raise SystemExit(f"no {base / 'tile_ids.txt'}")
    if args.part == "models":
        if "def phase_runs" not in (_HERE / "run_calibration_models.py").read_text(encoding="utf-8"):
            raise SystemExit(f"{_HERE / 'run_calibration_models.py'} has no phase_runs - sync the code first")
        selection = root / args.dedup_dir_name / "variant_selection.json"
        if args.phase != "sweep" and not selection.exists():
            raise SystemExit(f"--phase {args.phase} needs {selection} - run the sweep phase and its post job first")
    else:
        if not (_HERE / "connected_forcing.py").exists():
            raise SystemExit(f"{_HERE / 'connected_forcing.py'} missing - sync the code first")
        if "--coast-offset-cells" not in (_HERE / "run_final_tile.sh").read_text(encoding="utf-8"):
            raise SystemExit(f"{_HERE / 'run_final_tile.sh'} has no coast boundary - sync the code first")
        if args.phase != "sweep":
            raise SystemExit("--part sfincs reruns the sweep phase; run --part models for other phases")

    counts = reset_tiles(base, args.part, args.reset_runs, args.dry_run)
    what = f"{counts['tiles']} tile(s) for phase {args.phase}"
    if args.reset_runs:
        what += f", {counts['rasters']} model raster(s)"
    if args.dry_run:
        print(f"dry run: would reset {what} ({counts['missing']} listed without a tile directory)")
        return None
    print(f"reset {what} ({counts['missing']} listed without a tile directory)")

    # Job files of an earlier, larger run would otherwise stay next to the new ones.
    jobs = base / "hpc_jobs_stage2"
    for pattern in ("final_*.sbatch", "tiles_*.txt"):
        for p in jobs.glob(pattern):
            p.unlink()
    stage2 = argparse.Namespace(**vars(args))
    stage2.config = str(base / "resolved_config.yml")
    if args.part == "models":
        # SFINCS outputs exist, so run_final_tile.sh skips SFINCS: estimate only the model runs.
        stage2.sfincs_min_per_mcell, stage2.overhead_min = 0.0, 5.0
    elif not args.reset_runs:
        # Eikonal rasters exist, so the sweep is skipped: its time estimate is ~0.
        stage2.eikonal_min_per_mcell = 1.0
    submit = cmd_stage2(stage2)
    print(f"progress: python sfincs_tiles/check_calibration_progress.py --config {stage2.config} "
          f"--base-dir-name {args.base_dir_name}")
    return submit


COMMANDS = {"stage1": cmd_stage1, "stage2": cmd_stage2, "post": cmd_post, "rerun": cmd_rerun}


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    """Add the options of `command` (stage1, stage2, post, rerun) to `parser`."""
    from hpc_common import add_profile_set_argument

    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--submit", action="store_true", help="run the written submit script")
    parser.add_argument("--boundary-mode", choices=["coast", "line"], default="coast",
                        help="SFINCS boundary: coast = 3 cells seaward of the eikonal coastline, no station "
                             "line needed; line = along the station boundary line")
    add_profile_set_argument(parser)
    if command == "stage1":
        parser.add_argument("--n-jobs", type=int, default=10)
        parser.add_argument("--stage2-jobs", type=int, default=20, help="forwarded to the bridge's stage2 call")
        parser.add_argument("--n-tiles", type=int, default=1500)
        parser.add_argument("--stage1-hours", type=int, default=24)
        return
    parser.add_argument("--phase", choices=["sweep", "duration"], default="sweep",
                        help="run_calibration_models.py phase (CAL_PHASE of run_final_tile.sh)")
    parser.add_argument("--dedup-dir-name", default=None, help="post job: default <base-dir-name>_dedup")
    parser.add_argument("--post-shards", type=int, default=POST_SHARDS,
                        help="post jobs splitting the tile drivers and the dedup (each a sfincs_post job)")
    if command == "post":
        parser.add_argument("--no-dedup", dest="dedup", action="store_false",
                            help="rerun both analyses on the existing dedup results, without recomputing them")
        parser.add_argument("--rescore", action="store_true",
                            help="first rewrite every tile's raw calibration result from its rasters (new metrics)")
        return
    parser.add_argument("--n-jobs", type=int, default=20)
    parser.add_argument("--overhead-min", type=float, default=35.0)  # inputs, model build + subgrid, postprocessing
    parser.add_argument("--sfincs-min-per-mcell", type=float, default=12.5)  # p90 over 550 SFINCS runs, 4 threads
    parser.add_argument("--eikonal-min-per-mcell", type=float, default=10.0,
                        help="10 eikonal solves, per million active SFINCS cells (rough); scaled by RUN_LOAD")
    parser.add_argument("--time-margin", type=float, default=1.5)
    parser.add_argument("--no-post", dest="post", action="store_false",
                        help="no post job (variant selection, dedup, analyses) after the jobs")
    if command == "rerun":
        parser.add_argument("--part", choices=["models", "sfincs"], required=True)
        parser.add_argument("--reset-runs", action="store_true",
                            help="also delete every flood-model raster and result")
        parser.add_argument("--dry-run", action="store_true", help="only report what would be reset")


def run(command: str, args: argparse.Namespace) -> str | None:
    """Run `command` with parsed `args`; return the written submit script's Linux path, if any."""
    args.dedup_dir_name = getattr(args, "dedup_dir_name", None) or f"{args.base_dir_name}_dedup"
    return COMMANDS[command](args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        p = sub.add_parser(command)
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        add_arguments(p, command)
    args = parser.parse_args(argv)
    submit = run(args.command, args)
    if submit and not args.submit:
        print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
