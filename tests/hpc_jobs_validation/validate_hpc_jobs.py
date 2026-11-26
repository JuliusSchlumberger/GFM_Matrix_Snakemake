"""Generated HPC job chains on a synthetic site: job count, tiles per job, dependencies, job shape."""

import csv
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpc_fixture as fx  # noqa: E402

for _sub in ("src", "snakemake_workflow/scripts", "sfincs_tiles", "calibration_studies"):
    sys.path.insert(0, str(fx.REPO / _sub))
import generate_calibration_batch_jobs as studies  # noqa: E402
import generate_calibration_jobs as sfincs  # noqa: E402
import generate_hpc_exposure_jobs as exposure  # noqa: E402
import generate_hpc_postprocess_job as post  # noqa: E402
import generate_hpc_preprocess_job as pre  # noqa: E402
import generate_hpc_simulation_jobs as sim  # noqa: E402
from hpc_common import THREAD_ENV  # noqa: E402

JOBS_DIR = "model_outputs/hpc_jobs"


def _chain(tmp_path: Path, submit: str) -> dict[str, dict]:
    jobs = fx.jobs(fx.local_path(submit, tmp_path))
    for j in jobs.values():
        env = "sfincs" if j["profile"] == "sfincs_tile" else "gfm"
        fx.assert_common_job_shape(fx.local_path(j["script"], tmp_path), env)
    return jobs


def _targets_tiles(path: Path) -> set[str]:
    return {line.split("/")[-3] for line in path.read_text().split()}


def test_preprocess_batches_then_checked_dispatch(tmp_path) -> None:
    submit = pre.generate(fx.make_fixture(tmp_path))
    jobs = _chain(tmp_path, submit)
    assert list(jobs) == ["preprocess_batch_000", "preprocess_batch_001", "gfm_generate_jobs_and_dispatch"]
    d = tmp_path / "root" / JOBS_DIR
    assert [_targets_tiles(d / f"preprocess_batch_00{i}_targets.txt") for i in (0, 1)] == [{"1", "2", "3"}, {"4", "5"}]
    assert jobs["gfm_generate_jobs_and_dispatch"]["after"] == ["preprocess_batch_000", "preprocess_batch_001"]
    assert jobs["gfm_generate_jobs_and_dispatch"]["cpus_per_task"] == 1
    dispatch = (d / "generate_jobs_and_dispatch.sbatch").read_text()
    assert "hpc/gfm_hpc.py check-targets" in dispatch and "preprocess_batch_001_targets.txt" in dispatch
    assert dispatch.index("check-targets") < dispatch.index("run simulate")
    assert submit.endswith("/submit_preprocess_and_dispatch.sh")
    assert '"$GFM_PY" -m snakemake' in (d / "submit_preprocess_and_dispatch.sh").read_text()
    batch = (d / "preprocess_batch_000.sbatch").read_text()
    assert '--cores "$SLURM_CPUS_PER_TASK"' in batch and THREAD_ENV in batch


def test_preprocess_only_and_fuse(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path / "a")
    assert list(_chain(tmp_path / "a", pre.generate(cfg, preprocess_only=True))) == [
        "preprocess_batch_000", "preprocess_batch_001"]
    with pytest.raises(ValueError, match="hop_distance"):
        pre.generate(cfg, fuse=True)
    cfg = fx.make_fixture(tmp_path / "b", all_hop0=True)
    submit = pre.generate(cfg, fuse=True)
    assert submit.endswith("/submit_fused_batches.sh")
    assert "SIMULATE_TARGETS" in (tmp_path / "b" / "root" / JOBS_DIR / "preprocess_batch_000.sbatch").read_text()


def _triples(path: Path) -> list[tuple[str, str, str]]:
    return [tuple(r) for r in csv.reader(path.open())]


def test_simulation_waves_keep_whole_tiles_in_order(tmp_path) -> None:
    jobs = _chain(tmp_path, sim.generate(fx.make_fixture(tmp_path)))
    d = tmp_path / "root" / JOBS_DIR
    tiles = {n: [t for t, _, _ in _triples(d / f"{n}_triples.csv")] for n in jobs if n.startswith("wave")}
    assert tiles == {"wave0_batch_000": ["1"] * 2 + ["2"] * 2, "wave0_batch_001": ["3"] * 2,
                     "wave1_batch_000": ["4"] * 2, "wave1_batch_001": ["5"] * 2}
    assert jobs["wave0_batch_000"]["after"] == []
    assert jobs["wave1_batch_000"]["after"] == ["wave0_batch_000", "wave0_batch_001"]
    assert jobs["gfm_generate_postprocess_and_dispatch"]["after"] == [n for n in jobs if n.startswith("wave")]
    wave = (d / "wave0_batch_000.sbatch").read_text()
    assert "--workers" not in wave and THREAD_ENV in wave  # run_simulation_batch.py reads the job's cores


def test_simulation_resume_and_tiers(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path, tiers=True)
    fx.add_sim_results(tmp_path)
    jobs = _chain(tmp_path, sim.generate(cfg, resume=True))
    d = tmp_path / "root" / JOBS_DIR
    items = {n: _triples(d / f"{n}_triples.csv") for n in jobs if n.startswith("resume_wave")}
    assert sorted(i for v in items.values() for i in v) == [
        ("2", "RP100", "SLR_500"), ("3", "RP100", "SLR_0"), ("3", "RP100", "SLR_500"),
        ("5", "RP100", "SLR_0"), ("5", "RP100", "SLR_500")]
    assert jobs["resume_wave0_batch_000"]["partition"] == "24vcpu"
    assert items["resume_wave1_batch_000"] == [("5", "RP100", "SLR_0"), ("5", "RP100", "SLR_500")]  # one job per tile
    assert jobs["resume_wave1_batch_000"]["after"] == ["resume_wave0_batch_000", "resume_wave0_batch_001"]


def test_wave_packing_sizes_jobs_from_solve_count() -> None:
    packing = {"solve_minutes": 2.0, "target_hours": 2.0}
    assert sim.packed_batch_count(1200, 600, packing, 4) == 5  # 40 core-hours / 8 per job
    assert sim.packed_batch_count(3, 3, packing, 4) == 1
    assert sim.packed_batch_count(10_000, 4, packing, 4) == 4  # never more jobs than tiles


def test_simulation_wave_packing_overrides_tiers(tmp_path) -> None:
    packing = {"solve_minutes": 120.0, "target_hours": 1.0, "profile": "simulate_packed"}
    cfg = fx.make_fixture(tmp_path, tiers=True, overrides={"hpc": {"wave_packing": packing}})
    jobs = _chain(tmp_path, sim.generate(cfg))
    waves = {n: j for n, j in jobs.items() if n.startswith("wave")}
    # wave 0: 3 tiles x 2 solves x 2 h / (4 cores x 1 h) -> 3 jobs; wave 1: 2 tiles -> 2 jobs
    assert sorted(waves) == ["wave0_batch_000", "wave0_batch_001", "wave0_batch_002",
                             "wave1_batch_000", "wave1_batch_001"]
    assert {(j["partition"], j["cpus_per_task"], j["mem"], j["time"]) for j in waves.values()} == {
        ("4vcpu", 4, "30G", "0-08:00:00")}


def test_postprocess_native_merge_chunk_batches_then_vrt(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path, merge_only=True, overrides={
        "simulation": {"dem_source_manifest": "{root}/manifest.csv"},
        "postprocessing": {"native_chunk_m": 20000, "native_merge_jobs": 3}})
    jobs = _chain(tmp_path, post.generate(cfg))
    for slr in ("SLR_0", "SLR_500"):
        merges = [f"postprocess_local_native_merge_RP100_{slr}_batch_{k:03d}" for k in range(3)]
        assert all(jobs[m]["after"] == [] and jobs[m]["cpus_per_task"] == 1 for m in merges)
        assert jobs[f"postprocess_local_native_vrt_RP100_{slr}"]["after"] == merges
    d = tmp_path / "root" / JOBS_DIR
    assert "--batch 2 --n-batches 3" in (d / "postprocess_local_native_merge_RP100_SLR_0_batch_002.sbatch").read_text()
    assert "--vrt" in (d / "postprocess_local_native_vrt_RP100_SLR_0.sbatch").read_text()
    assert len(jobs) == 8


def test_postprocess_per_chunk_then_exposure_then_flood_totals(tmp_path) -> None:
    jobs = _chain(tmp_path, post.generate(fx.make_fixture(tmp_path)))
    chunk_jobs = [n for n in jobs if n.startswith("postprocess_postprocess_chunks")]
    assert len(chunk_jobs) == 2
    d = tmp_path / "root" / JOBS_DIR
    chunks = [{line.split("flood_fraction_")[1][:7] for line in (d / f"{n}_targets.txt").read_text().split()}
              for n in chunk_jobs]
    assert chunks[0].isdisjoint(chunks[1]) and chunks[0] | chunks[1] == {"N00E000", "N00E005", "N00E010", "N05E000"}
    assert jobs["exposure_pass1_batch_000"]["after"] == chunk_jobs
    assert jobs["exposure_reduce_write"]["after"] == ["exposure_pass2_batch_000", "exposure_pass2_batch_001"]
    assert jobs["flood_totals_and_plots"]["after"] == ["exposure_reduce_write"]
    assert jobs["flood_totals_and_plots"]["cpus_per_task"] == 24
    pass1 = (d / "exposure" / "exposure_pass1_batch_000.sbatch").read_text()
    assert 'stage_configfile_locally' in pass1 and '--config "$LOCAL_CONFIGFILE"' in pass1


def test_postprocess_three_phases_chain(tmp_path) -> None:
    jobs = _chain(tmp_path, post.generate(fx.make_fixture(tmp_path, n_nodes=5), flood_totals=False))
    phase = {p: [n for n in jobs if n.startswith(f"postprocess_postprocess_{p}_")]
             for p in ("merge", "exposure_grid", "flood_fraction")}
    assert all(phase.values())
    assert all(jobs[n]["after"] == phase["merge"] for n in phase["exposure_grid"])
    assert all(jobs[n]["after"] == phase["exposure_grid"] for n in phase["flood_fraction"])
    assert "flood_totals_and_plots" not in jobs


def test_exposure_standalone(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path)
    ff = tmp_path / "root" / "merged_results" / "chunks" / "flood_fraction"
    ff.mkdir(parents=True)
    for cid in ("N00E000", "N05E000"):
        (ff / f"flood_fraction_{cid}_RP100_SLR_0.tif").write_bytes(b"x")
        (ff.parent / f"exposure_population_grid_{cid}.tif").write_bytes(b"x")
    jobs = _chain(tmp_path, exposure.generate(cfg))
    assert len(jobs) == 6 and jobs["exposure_pass1_batch_000"]["after"] == []


def test_sfincs_stage2_lpt_and_post(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path)
    base = fx.add_sfincs_stage2(tmp_path)
    args = Namespace(config=str(cfg), base_dir_name=base.name, dedup_dir_name=f"{base.name}_dedup", n_jobs=2, boundary_mode="coast",
                     phase="sweep", overhead_min=35.0, sfincs_min_per_mcell=12.5, eikonal_min_per_mcell=10.0,
                     time_margin=1.5, post=True, submit=False, profile_set=["sfincs_post.mem=120G"], post_shards=4)
    jobs = _chain(tmp_path, sfincs.cmd_stage2(args))
    shards = [f"cal_post_sweep_shard{k}" for k in range(4)]
    assert list(jobs) == ["cal_final_000", "cal_final_001", *shards, "cal_post_sweep", "cal_post_sweep_dedup"]
    out = base / "hpc_jobs_stage2"
    assert [(out / f"tiles_00{j}.txt").read_text().split() for j in (0, 1)] == [["3"], ["1", "2"]]
    assert all(jobs[s]["after"] == ["cal_final_000", "cal_final_001"] for s in shards)
    assert jobs["cal_post_sweep"]["after"] == shards and jobs["cal_post_sweep_dedup"]["after"] == shards
    assert jobs["cal_post_sweep"]["mem"] == "120G"
    shard2 = (out / "post_sweep_shard2.sbatch").read_text()
    assert "compute_tile_drivers.py" in shard2 and "--shard 2/4" in shard2 and "compute_calibration_result_dedup.py" in shard2
    dedup = (out / "post_models_sweep_dedup.sbatch").read_text()
    assert "dedup_shard_*of4.ok" in dedup and dedup.index("dedup_shard_") < dedup.index("run_calibration_sweep_analysis")
    final = (out / "final_000.sbatch").read_text()
    assert "SFINCS_THREADS" not in final and THREAD_ENV in final  # run_final_tile.sh: the job's cores
    for var in ("GFM_DATA_ROOT", "GFM_CODE_ROOT", "CONTAINER_CMD", "SFINCS_IMAGE", "SFINCS_TIMEOUT_S"):
        assert f"export {var}=" in final


def test_sfincs_post_without_dedup_reruns_both_analyses_only(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path)
    base = fx.add_sfincs_stage2(tmp_path)
    args = Namespace(config=str(cfg), base_dir_name=base.name, dedup_dir_name=f"{base.name}_dedup", phase="sweep",
                     submit=False, profile_set=[], dedup=False, post_shards=2)
    jobs = _chain(tmp_path, sfincs.cmd_post(args))
    out = base / "hpc_jobs_stage2"
    scripts = "".join(f.read_text() for f in out.glob("post_*.sbatch"))
    assert list(jobs) == ["cal_post_sweep_shard0", "cal_post_sweep_shard1", "cal_post_sweep", "cal_post_sweep_dedup"]
    assert "compute_calibration_result_dedup.py" not in scripts and "dedup_shard_" not in scripts
    assert f"--raster-base-dir-name {base.name}" in (out / "post_models_sweep_dedup.sbatch").read_text()


def test_solver_study_lpt_by_pixels(tmp_path) -> None:
    cfg = fx.make_fixture(tmp_path)
    fx.add_study_inputs(tmp_path)
    submit = studies.generate("sweep_budget", cfg, friction_source="uniform", n_nodes=2)
    jobs = _chain(tmp_path, submit)
    hpc_jobs = tmp_path / "root" / "calibration_500_tiles" / "hpc_jobs"
    tiles = [(hpc_jobs / f"{n}.sbatch").read_text().split("--tile-ids ")[1].split() for n in jobs]
    assert tiles == [["2", "5"], ["3", "1", "4"]]  # 1, 1, 1, 0.64, 0.25 deg2: largest first, ties to job 0
    assert "--friction-source uniform" in (hpc_jobs / f"{next(iter(jobs))}.sbatch").read_text()
