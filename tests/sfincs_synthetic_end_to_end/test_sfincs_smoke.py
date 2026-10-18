"""Smoke-test SFINCS build -> run -> postprocess on the smallest finished calibration tile.

Needs the data root (`paths.root`), a local sfincs.exe (`paths.sfincs_exe` in config_local.yml) and a finished
FIXTURE_BASE_DIR_NAME tile; skips otherwise. The fixture is only read: its build inputs
are copied into a local temp dir (pytest's tmp_path) and the model is built and run
there, never under the data root.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SFINCS_TILES_DIR = REPO_ROOT / "sfincs_tiles"
sys.path.insert(0, str(SFINCS_TILES_DIR))

from build_sfincs_tile import build_sfincs_tile  # noqa: E402
from gfm_config import read_root  # noqa: E402
from run_sfincs_tile import compute_max_inundation  # noqa: E402
from sfincs_run import run_sfincs_subprocess  # noqa: E402

FIXTURE_BASE_DIR_NAME = "calibration_roughness_SFINCS"  # read-only
SMOKE_BASE_DIR_NAME = "sfincs_smoke_test"
PREP_FILES = ["elevation_combined.tif", "manning_n.tif", "matched_boundary_points.gpkg", "corrected_hydrographs.csv"]
CONFIG_DIR = REPO_ROOT / "snakemake_workflow" / "config"


class _SilentLog:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass


def _fixture_tile(root: Path) -> str | None:
    base = root / FIXTURE_BASE_DIR_NAME
    sizes = []
    for pf in (base / "boundary_lines" / "preflight").glob("*.json"):
        rec = json.loads(pf.read_text())
        if rec.get("ok") and (base / pf.stem / "outputs" / "calibration_run.done").exists():
            sizes.append((rec["n_active"], pf.stem))
    for _, t in sorted(sizes):
        tile = base / t
        if all((tile / "sfincs_model" / f).exists() for f in PREP_FILES) and (tile / f"boundary_line_{t}.gpkg").exists():
            return t
    return None


def _sfincs_exe() -> Path | None:
    """`paths.sfincs_exe` from config_local.yml (machine-specific), or None if unset."""
    local_path = CONFIG_DIR / "config_local.yml"
    if not local_path.exists():
        return None
    with open(local_path, encoding="utf-8") as f:
        exe = ((yaml.safe_load(f) or {}).get("paths") or {}).get("sfincs_exe")
    return Path(exe) if exe else None


def _prereqs_available() -> tuple[Path, Path, str] | None:
    config_path = CONFIG_DIR / "config.yml"
    sfincs_exe = _sfincs_exe()
    if not config_path.exists() or sfincs_exe is None or not sfincs_exe.exists():
        return None
    try:
        root = read_root(config_path)
    except Exception:
        return None
    tile_id = _fixture_tile(root)
    return None if tile_id is None else (root, sfincs_exe, tile_id)


def _stage_fixture(root: Path, tile_id: str, stage_root: Path) -> Path:
    """Copy the fixture tile's build inputs from `root` to `stage_root`/SMOKE_BASE_DIR_NAME."""
    src, dst = root / FIXTURE_BASE_DIR_NAME / tile_id, stage_root / SMOKE_BASE_DIR_NAME / tile_id
    shutil.copytree(src / "inputs", dst / "inputs")
    (dst / "sfincs_model").mkdir(parents=True)
    for f in PREP_FILES:
        shutil.copy2(src / "sfincs_model" / f, dst / "sfincs_model" / f)
    line = dst / f"boundary_line_{tile_id}.gpkg"
    shutil.copy2(src / line.name, line)
    return line


def test_build_run_postprocess_produces_plausible_output(tmp_dir: Path) -> None:
    print("=== SFINCS smoke test: full build -> run -> postprocess on a real fixture tile ===")
    prereqs = _prereqs_available()
    if prereqs is None:
        pytest.skip("the data root (paths.root), a finished calibration_roughness_SFINCS tile, and/or the local "
                    "sfincs.exe (paths.sfincs_exe in config_local.yml) aren't available in this environment")
    root, sfincs_exe, tile_id = prereqs
    # build_sfincs_tile only touches <root>/<base_dir_name>/<tile_id>/{inputs,sfincs_model},
    # so a local stage root keeps every write off the data root.
    stage_root = tmp_dir
    line = _stage_fixture(root, tile_id, stage_root)
    print(f"  fixture: tile {tile_id} from {FIXTURE_BASE_DIR_NAME}, copied to {stage_root / SMOKE_BASE_DIR_NAME}")

    sfincs_dir = build_sfincs_tile(tile_id, stage_root, base_dir_name=SMOKE_BASE_DIR_NAME, boundary_lines_gpkg=line)
    assert sfincs_dir.exists()
    inp_path = sfincs_dir / "sfincs.inp"
    assert inp_path.exists(), "build_sfincs_tile did not produce sfincs.inp"
    print(f"  build: OK, wrote {sfincs_dir}")

    run_sfincs_subprocess(sfincs_exe, sfincs_dir, timeout_s=300.0, log=_SilentLog(), label=f"smoke test tile {tile_id}")
    map_path = sfincs_dir / "sfincs_map.nc"
    assert map_path.exists(), "SFINCS ran but produced no sfincs_map.nc"
    print("  run: OK, produced sfincs_map.nc")

    land_mask_path = stage_root / SMOKE_BASE_DIR_NAME / tile_id / "inputs" / "mask.tif"
    hmax, grid_info = compute_max_inundation(sfincs_dir, land_mask_path)

    assert hmax.ndim == 2 and hmax.size > 0, hmax.shape
    finite = hmax[np.isfinite(hmax)]
    # Plausibility only: a units/sign bug gives depths in the thousands of metres.
    if finite.size:
        assert finite.max() < 100.0, f"implausible max depth {finite.max()} m - likely a units/decode bug"
        assert finite.min() >= 0.0, f"negative flood depth {finite.min()} m - should never happen"
    print(f"  postprocess: OK, {finite.size} finite cell(s) of {hmax.size} total"
          + (f", max depth {finite.max():.3f} m" if finite.size else " (tile fully dry this run - not itself a failure)"))
    print("PASS: build -> run -> postprocess completed end to end with a bounded, plausible result")
    print()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sfincs_smoke_") as td:
        try:
            test_build_run_postprocess_produces_plausible_output(Path(td))
        except pytest.skip.Exception as exc:
            print(f"SKIP: {exc}\n")
    print("SFINCS end-to-end smoke test complete.")


if __name__ == "__main__":
    main()
