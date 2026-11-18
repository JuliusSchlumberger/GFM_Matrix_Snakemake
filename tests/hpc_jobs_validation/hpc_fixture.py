"""Synthetic HPC site: five tiles over two hop waves and four chunks, a config dir with local + site layers."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import geopandas as gpd
import yaml
from shapely.geometry import box

REPO = Path(__file__).resolve().parents[2]
LINUX_ROOT = "/p/fx/root"
LINUX_CODE = "/u/fx/code"
GFM_PY = "/envs/gfm/bin/python"
SFINCS_PY = "/envs/sfincs/bin/python"
JOB_ENV = ["unset PROJ_LIB PROJ_DATA GDAL_DATA", "export HDF5_USE_FILE_LOCKING=FALSE"]

TILES = [  # tile_id, hop_distance, bbox (EPSG:4326)
    (1, 0, (0.1, 0.1, 0.9, 0.9)),
    (2, 0, (1.0, 1.0, 2.0, 2.0)),
    (3, 0, (6.0, 1.0, 7.0, 2.0)),
    (4, 1, (11.0, 1.0, 11.5, 1.5)),
    (5, 1, (1.0, 6.0, 2.0, 7.0)),
]
TIERS = [
    {"hops": [0], "n_nodes": 2, "partition": "24vcpu", "cpus_per_task": 24, "mem": "168G", "time": "0-16:00:00"},
    {"hops": None, "n_nodes": None, "partition": "1vcpu", "cpus_per_task": 1, "mem": "7G", "time": "0-01:00:00"},
]


def site(state_root: str, tiers: bool = False) -> dict:
    four = {"partition": "4vcpu", "cpus_per_task": 4, "mem": "30G"}
    hpc = {
        "account": "acct1",
        "python": {"gfm": GFM_PY, "sfincs": SFINCS_PY},
        "container": {"cmd": "apptainer", "sfincs_image": "docker://deltares/sfincs-cpu:test"},
        "state_root": state_root,
        "job_env": JOB_ENV,
        "profiles": {
            "default": {"partition": "1vcpu", "mem": "7G"},
            "flood_totals": {"partition": "24vcpu", "cpus_per_task": 24, "mem": "32G"},
            "sfincs_stage1": dict(four), "sfincs_bridge": dict(four), "sfincs_tile": dict(four),
            "sfincs_post": {**four, "time": "0-12:00:00"},
            "validation": {**four, "time": "2-00:00:00"},
            "study_figures": {**four, "time": "0-04:00:00"},
        },
    }
    if tiers:
        hpc["wave_tiers"] = TIERS
    return {"paths": {"root": LINUX_ROOT, "code_root": LINUX_CODE}, "hpc": hpc}


def make_fixture(base: Path, *, n_nodes: int = 2, all_hop0: bool = False, tiers: bool = False,
                 merge_only: bool = False) -> Path:
    """Build the fixture under `base`; return its config.yml (siblings: config_local.yml, config_hpc.yml)."""
    if base.exists():
        shutil.rmtree(base)
    root, cfg_dir = base / "root", base / "config"
    cfg_dir.mkdir(parents=True)
    shutil.copy(REPO / "snakemake_workflow" / "config" / "config.yml", cfg_dir / "config.yml")
    local = {
        "paths": {"root": root.as_posix(), "code_root": REPO.as_posix()},
        "boundary_conditions": {"return_periods": [100], "waterlevel_names": ["SLR_0", "SLR_500"]},
        "adaptation": {"slr_intensities": []},
        "hpc": {"n_nodes": n_nodes},
        "postprocessing": {"merge_only": merge_only},
    }
    (cfg_dir / "config_local.yml").write_text(yaml.safe_dump(local), encoding="utf-8")
    (cfg_dir / "config_hpc.yml").write_text(yaml.safe_dump(site((base / "state").as_posix(), tiers)),
                                            encoding="utf-8")
    (root / "processed_inputs" / "mask").mkdir(parents=True)
    gpd.GeoDataFrame(
        {"tile_id": [t for t, _, _ in TILES], "hop_distance": [0 if all_hop0 else h for _, h, _ in TILES]},
        geometry=[box(*b) for _, _, b in TILES], crs="EPSG:4326",
    ).to_file(root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg", driver="GPKG")
    return cfg_dir / "config.yml"


def add_sim_results(base: Path) -> None:
    """Tiles 1 and 4 done, tile 2 half done."""
    for tid, names in ((1, ["SLR_0", "SLR_500"]), (4, ["SLR_0", "SLR_500"]), (2, ["SLR_0"])):
        d = base / "root" / "model_outputs" / str(tid) / "results"
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / f"waterdepth_RP100_{n}.tif").write_bytes(b"x")


def add_sfincs_stage2(base: Path, base_dir_name: str = "calibration_roughness_SFINCS") -> Path:
    """Tiles 1, 2, 3, 5 selected; 1-3 with pre-flight cell counts, 5 done. Return the base dir."""
    b = base / "root" / base_dir_name
    (b / "boundary_lines" / "preflight").mkdir(parents=True, exist_ok=True)
    (b / "tile_ids.txt").write_text("1\n2\n3\n5\n")
    for tid, n in ((1, 2_000_000), (2, 500_000), (3, 4_000_000)):
        (b / "boundary_lines" / "preflight" / f"{tid}.json").write_text(json.dumps({"n_active": n}))
    (b / "5" / "outputs").mkdir(parents=True, exist_ok=True)
    (b / "5" / "outputs" / "calibration_run.done").write_text("x")
    return b


def add_study_inputs(base: Path, study: str = "calibration_500_tiles") -> None:
    s = base / "root" / study
    s.mkdir(parents=True, exist_ok=True)
    (s / "candidate_tiles.txt").write_text("1\n2\n3\n4\n5\n")
    for src in ("landcover", "uniform"):
        (s / f"sweep_budget_{src}").mkdir(parents=True, exist_ok=True)
        (s / f"sweep_budget_{src}" / "wet_tiles_selected.txt").write_text("2\n3\n5\n")


def jobs(submit_local: Path) -> dict[str, dict]:
    """Return the jobs.json next to a submit script, by job name."""
    manifest = json.loads((submit_local.parent / f"{submit_local.stem}.jobs.json").read_text())
    return {j["name"]: j for j in manifest["jobs"]}


def local_path(linux: str, base: Path) -> Path:
    """Map a fixture Linux path (/p/fx/root/...) back to the local fixture tree."""
    return base / "root" / Path(linux).relative_to(LINUX_ROOT)


def assert_common_job_shape(script: Path, env: str = "gfm") -> None:
    """Every job: account, interpreter exports, its env's bin first on PATH, the common env lines."""
    text = script.read_text()
    assert "#SBATCH --account=acct1" in text
    assert f'export GFM_PY="{GFM_PY}"' in text and f'export SFINCS_PY="{SFINCS_PY}"' in text
    assert f'export PATH="/envs/{env}/bin:$PATH"' in text
    for line in JOB_ENV:
        assert line in text
    name = re.search(r"#SBATCH --job-name=(\S+)", text).group(1)
    assert f"/{name}_%j.out" in text and f"/{name}_%j.err" in text
    for banned in ("env_activate", "module load", "conda activate", "site_env"):
        assert banned not in text
    assert not re.search(r"(^|[\s(])python\s", text, re.M), "bare python on PATH"
    assert not re.search(r"(^\s*|run_snakemake_with_retry\s+)snakemake\s", text, re.M), "snakemake not via $GFM_PY -m"
