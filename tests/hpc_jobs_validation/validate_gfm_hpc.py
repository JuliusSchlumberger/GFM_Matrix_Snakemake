"""hpc/gfm_hpc.py and hpc_common: profiles, packers, scenario names, the dispatch gate, rerun, validate, plan."""

import os
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpc_fixture as fx  # noqa: E402

sys.path.insert(0, str(fx.REPO / "hpc"))
import gfm_hpc  # noqa: E402
import harness  # noqa: E402
import generate_calibration_jobs as sfincs  # noqa: E402
import hpc_common  # noqa: E402
from hpc_common import apply_profile_sets, even_split, lpt_pack, missing_paths, resolve_profile  # noqa: E402

HPC = {"account": "acct", "profiles": {"default": {"partition": "1vcpu", "time": "1-00:00:00", "mem": "7G",
                                                    "cpus_per_task": 1},
                                        "post": {"partition": "4vcpu", "cpus_per_task": 4, "mem": ""}}}


def test_resolve_profile_layers() -> None:
    assert resolve_profile(HPC, "post") == {"account": "acct", "partition": "4vcpu", "time": "1-00:00:00",
                                            "mem": "7G", "cpus_per_task": 4}
    assert resolve_profile(HPC, "post", {"time": "0-01:00:00"})["time"] == "0-01:00:00"
    with pytest.raises(KeyError, match="hpc.profiles.nope"):
        resolve_profile(HPC, "nope")
    with pytest.raises(ValueError, match="mem"):
        resolve_profile({"profiles": {"default": {"time": "1", "cpus_per_task": 1}, "x": {}}}, "x")


def test_profile_set() -> None:
    hpc = {"profiles": {"default": {}, "post": None}}
    apply_profile_sets(hpc, ["post.partition=16vcpu", "post.cpus_per_task=16", "default.time=2-00:00:00"])
    assert hpc["profiles"] == {"default": {"time": "2-00:00:00"}, "post": {"partition": "16vcpu", "cpus_per_task": 16}}
    for bad, msg in (("post.partition", "PROFILE.FIELD=VALUE"), ("partition=4vcpu", "PROFILE.FIELD=VALUE"),
                     ("nope.mem=1G", "unknown profile"), ("post.nodes=2", "field must be")):
        with pytest.raises(ValueError, match=msg):
            apply_profile_sets(hpc, [bad])


def test_packers() -> None:
    assert even_split(list("abcde"), 2) == [list("abc"), list("de")]
    assert even_split(list("ab"), 5) == [["a"], ["b"]]
    assert even_split([], 3) == []
    # Descending cost, ties in insertion order, least-loaded bucket with the lowest index.
    assert lpt_pack({"a": 1, "b": 5, "c": 3, "d": 3, "e": 1}, 2) == [["b", "a", "e"], ["c", "d"]]
    assert lpt_pack({"a": 1}, 3) == [["a"], [], []]


def test_scenario_names_and_stale_warning(tmp_path, capsys) -> None:
    for name in ("config.yml", "config_local.yml", "s.yml", "s_materialized.yml", "s__v_materialized.yml"):
        (tmp_path / name).write_text("{}")
    old = time.time() - 100
    for name in ("config.yml", "config_local.yml", "s.yml"):
        os.utime(tmp_path / name, (old, old))
    assert gfm_hpc.scenario_config("production", config_dir=tmp_path) == tmp_path / "config.yml"
    assert gfm_hpc.scenario_config("s", config_dir=tmp_path) == tmp_path / "s_materialized.yml"
    assert gfm_hpc.scenario_config("s", "v", config_dir=tmp_path) == tmp_path / "s__v_materialized.yml"
    assert "WARNING" not in capsys.readouterr().err
    os.utime(tmp_path / "s.yml", (old + 200, old + 200))
    gfm_hpc.scenario_config("s", config_dir=tmp_path)
    assert "older than s.yml" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="materialize_configs.py t"):
        gfm_hpc.scenario_config("t", config_dir=tmp_path)


def test_dispatch_gate_counts_missing_preprocessing_outputs(tmp_path, capsys, monkeypatch) -> None:
    import config_utils

    monkeypatch.setattr(config_utils, "path_ready", lambda p: False)  # no retry delays
    present = tmp_path / "dem.tif"
    present.write_bytes(b"x")
    targets = tmp_path / "preprocess_batch_000_targets.txt"
    targets.write_text("\n".join([str(present)] + [str(tmp_path / f"m{i}.tif") for i in range(7)]) + "\n")
    n, first = missing_paths(targets.read_text().split())
    assert (n, len(first)) == (7, 5)
    with pytest.raises(SystemExit) as exc:
        gfm_hpc.main(["check-targets", str(targets)])
    assert exc.value.code == 1
    assert "7 of 8 preprocessing output(s) missing" in capsys.readouterr().err
    targets.write_text(f"{present}\n")
    gfm_hpc.main(["check-targets", str(targets)])


# ── SFINCS rerun ──────────────────────────────────────────────────────────────


def _rerun_site(tmp_path: Path) -> tuple[Path, Path]:
    """A finished calibration: tiles 1-3 with markers, rasters and SFINCS model; tile 9 listed, absent."""
    cfg = fx.make_fixture(tmp_path)
    base = fx.add_sfincs_stage2(tmp_path)
    (base / "tile_ids.txt").write_text("1\n2\n3\n9\n")
    for t in ("1", "2", "3"):
        out = base / t / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        for f in ("calibration_run.done", "calibration_run.failed", "calibration_run.step", "model_runs.json",
                  "calibration_result.json", "eikonal_solver_diagnostics.json",
                  "bathtub_waterdepth_RP100_SLR_0.tif", "eikonal_waterdepth_RP100_SLR_0_f0.25.tif"):
            (out / f).write_text("x")
        (base / t / "sfincs_model").mkdir(exist_ok=True)
        (base / t / "sfincs_model" / "sfincs.inp").write_text("x")
    hpc_view = yaml.safe_load(yaml.safe_dump(hpc_common_load(cfg)))
    hpc_view["paths"]["root"] = (tmp_path / "root").as_posix()
    (base / "resolved_config.yml").write_text(yaml.safe_dump(hpc_view))
    jobs = base / "hpc_jobs_stage2"
    jobs.mkdir()
    for stale in ("final_007.sbatch", "tiles_007.txt"):
        (jobs / stale).write_text("old")
    return cfg, base


def hpc_common_load(cfg: Path) -> dict:
    from config_utils import load_hpc_config

    return load_hpc_config(cfg)


def _rerun_args(cfg: Path, base: Path, **kw) -> Namespace:
    parser = gfm_hpc.build_parser()
    argv = ["sfincs", "rerun", "--config", str(cfg), "--base-dir-name", base.name, "--n-jobs", "2"]
    for k, v in kw.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return parser.parse_args(argv)


@pytest.fixture
def submitted(monkeypatch):
    calls = []
    monkeypatch.setattr(hpc_common, "submit_script", calls.append)
    return calls


def test_rerun_models_keeps_runs_and_clears_stale_job_files(tmp_path, submitted) -> None:
    cfg, base = _rerun_site(tmp_path)
    submit = sfincs.run("rerun", _rerun_args(cfg, base, part="models", submit=True))
    out = base / "1" / "outputs"
    assert not any((out / m).exists() for m in sfincs.TILE_MARKERS)
    assert (out / "model_runs.json").exists() and (out / "bathtub_waterdepth_RP100_SLR_0.tif").exists()
    assert (base / "1" / "sfincs_model" / "sfincs.inp").exists()
    jobs = base / "hpc_jobs_stage2"
    assert not (jobs / "final_007.sbatch").exists() and not (jobs / "tiles_007.txt").exists()
    assert submitted == [submit]
    names = list(fx.jobs(jobs / "submit_all.sh"))
    assert names == ["cal_final_000", "cal_final_001", "cal_post_sweep"]


def test_rerun_models_reset_runs(tmp_path, submitted) -> None:
    cfg, base = _rerun_site(tmp_path)
    sfincs.run("rerun", _rerun_args(cfg, base, part="models", reset_runs=True, no_post=True))
    out = base / "2" / "outputs"
    assert sorted(p.name for p in out.iterdir()) == []
    assert submitted == []  # no --submit
    assert list(fx.jobs(base / "hpc_jobs_stage2" / "submit_all.sh")) == ["cal_final_000", "cal_final_001"]


def test_rerun_sfincs_part(tmp_path, submitted) -> None:
    cfg, base = _rerun_site(tmp_path)
    sfincs.run("rerun", _rerun_args(cfg, base, part="sfincs"))
    assert not (base / "3" / "sfincs_model" / "sfincs.inp").exists()
    assert not (base / "3" / "outputs" / "calibration_run.done").exists()
    assert (base / "3" / "outputs" / "model_runs.json").exists()
    assert not (base / "hpc_jobs_stage2" / "final_007.sbatch").exists()


def test_rerun_dry_run_and_guards(tmp_path, submitted, capsys) -> None:
    cfg, base = _rerun_site(tmp_path)
    assert sfincs.run("rerun", _rerun_args(cfg, base, part="models", reset_runs=True, dry_run=True)) is None
    assert "would reset 3 tile(s) for phase sweep, 6 model raster(s) (1 listed without" in capsys.readouterr().out
    assert (base / "1" / "outputs" / "calibration_run.done").exists()
    assert (base / "hpc_jobs_stage2" / "final_007.sbatch").exists()
    with pytest.raises(SystemExit, match="variant_selection.json"):
        sfincs.run("rerun", _rerun_args(cfg, base, part="models", phase="duration"))
    with pytest.raises(SystemExit, match="sweep phase"):
        sfincs.run("rerun", _rerun_args(cfg, base, part="sfincs", phase="duration"))
    (base / "tile_ids.txt").unlink()
    with pytest.raises(SystemExit, match="tile_ids.txt"):
        sfincs.run("rerun", _rerun_args(cfg, base, part="models"))
    assert submitted == []


def test_rerun_needs_synced_code(tmp_path, monkeypatch) -> None:
    cfg, base = _rerun_site(tmp_path)
    code = tmp_path / "code"
    code.mkdir()
    (code / "run_calibration_models.py").write_text("def main(): pass\n")
    monkeypatch.setattr(sfincs, "_HERE", code)
    with pytest.raises(SystemExit, match="sync the code"):
        sfincs.run("rerun", _rerun_args(cfg, base, part="models"))
    with pytest.raises(SystemExit, match="connected_forcing.py missing"):
        sfincs.run("rerun", _rerun_args(cfg, base, part="sfincs"))


# ── validate and the plan ─────────────────────────────────────────────────────


def test_validate_runs_groups_with_config_placeholders(tmp_path, monkeypatch) -> None:
    cfg = fx.make_fixture(tmp_path)
    with open(cfg.parent / "config_local.yml", "a", encoding="utf-8") as f:
        yaml.safe_dump({"validation": {"runs": {
            "a": [["validation/x.py", "--config", "{config}", "--out", "{root}/v"]],
            "b": [["validation/y.py", "--global-config", "{config:other}", "--rp", "RP100"]]}}}, f)
    monkeypatch.setattr(gfm_hpc, "scenario_config", lambda name, variant=None: tmp_path / f"{name}.yml")
    monkeypatch.setattr(gfm_hpc, "hpc_view", lambda path: f"/view/{Path(path).name}")
    ran = []
    monkeypatch.setattr(gfm_hpc, "_python", lambda *a: ran.append([str(x) for x in a]))
    gfm_hpc.main(["validate", "--config", str(cfg)])
    assert ran == [["validation/x.py", "--config", "/view/config.yml", "--out", f"{fx.LINUX_ROOT}/v"],
                   ["validation/y.py", "--global-config", "/view/other.yml", "--rp", "RP100"]]
    ran.clear()
    gfm_hpc.main(["validate", "--config", str(cfg), "--only", "b"])
    assert [r[0] for r in ran] == ["validation/y.py"]
    with pytest.raises(SystemExit, match="no group"):
        gfm_hpc.main(["validate", "--config", str(cfg), "--only", "c"])


def test_validate_submit_writes_a_validation_job(tmp_path, submitted) -> None:
    cfg = fx.make_fixture(tmp_path)
    gfm_hpc.main(["validate", "--config", str(cfg), "--only", "scores", "--submit"])
    jobs = fx.jobs(fx.local_path(submitted[0], tmp_path))
    job = jobs["validate_config_scores"]
    assert (job["partition"], job["time"]) == ("4vcpu", "2-00:00:00")
    script = fx.local_path(job["script"], tmp_path)
    fx.assert_common_job_shape(script)
    view = tmp_path / "state" / "configs" / "config.yml"  # the node reads the HPC view, complete in itself
    assert view.exists()
    assert f"hpc/gfm_hpc.py validate --only scores --config {view}" in script.read_text()


def test_hydrax_plan_keeps_its_step_ids() -> None:
    # A running controller's steps.json is keyed by these ids.
    plan = harness.load_plan(fx.REPO / "hpc" / "plans" / "hydrax_rerun.yml")
    assert [s["id"] for s in plan["steps"]] == [
        "product", "materialize", "production_preprocess", "reset_studies", "studies_preprocess", "sweep_landcover",
        "sweep_uniform", "wet_tiles", "obstacle_coupling", "study_figures", "reset_sfincs", "sfincs_sweep",
        "sfincs_duration", "choose_parameters", "rematerialize", "reset_local", "local_station_caches", "local_runs",
        "validate_england_local", "validate_england_global", "validate_england_map", "validate_wales",
        "validate_sf_bay", "exposure_table", "local_variants", "approve_production", "reset_production",
        "production_run"]
    harness.load_plan(fx.REPO / "hpc" / "plans" / "smoke.yml")


def test_plan_validation_groups_exist() -> None:
    plan = (fx.REPO / "hpc" / "plans" / "hydrax_rerun.yml").read_text()
    for name, groups in (("england_local_input", {"scores", "agreement_map", "exposure_table"}),
                         ("england_global_input", {"scores"}), ("wales_local_input", {"scores"}),
                         ("sf_bay_local_input", {"scores"})):
        assert f"--scenario {name}" in plan
        cfg = yaml.safe_load((fx.REPO / "snakemake_workflow" / "config" / f"{name}.yml").read_text())
        assert set(cfg["validation"]["runs"]) == groups
