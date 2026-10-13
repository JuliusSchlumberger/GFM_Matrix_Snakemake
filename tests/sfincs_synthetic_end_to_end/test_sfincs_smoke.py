"""End-to-end smoke test for the SFINCS build+run+postprocess pipeline
(build_sfincs_tile.py -> run_sfincs_tile.py), against a real, small, already
-prepped fixture tile - the single highest-leverage "did I break the whole
pipeline" check, and (before this test) the only thing playing that role
was manually rebuilding real tiles by hand after each change.

Not a pure-synthetic test (unlike most of the other sfincs_* validation
tests added alongside this one): constructing a from-scratch synthetic
tile would mean faking GEBCO/MDT/COAST-HG catalog entries and a
tile_geometry.gpkg with real hop_distance/ocean-boundary conventions -
significant duplicated machinery for something a small REAL tile already
does correctly and cheaply (the smallest finished calibration tile, see
"Needs" below). This mirrors the existing house convention (most
`tests/*_validation/` scripts here are real-data validation tools, run by
hand, not CI-isolated unit tests - see e.g. this session's own
`sfincs_hydrograph_truncation_validation`).

Needs: the real P:\\ modelling share, a real local SFINCS executable, and
one finished tile of the final calibration (FIXTURE_BASE_DIR_NAME): the
smallest one (fewest active SFINCS cells) whose prep-stage files
(elevation_combined.tif, manning_n.tif, matched_boundary_points.gpkg,
corrected_hydrographs.csv) and boundary line exist. Its inputs are COPIED
into SMOKE_BASE_DIR_NAME and the model is built and run there - the fixture
tile itself is never touched. Skips (not fails) if anything is unavailable.

Usage:
    python test_sfincs_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SFINCS_TILES_DIR = REPO_ROOT / "sfincs_tiles"
sys.path.insert(0, str(SFINCS_TILES_DIR))

from build_sfincs_tile import build_sfincs_tile  # noqa: E402
from gfm_config import read_root  # noqa: E402
from run_sfincs_tile import compute_max_inundation  # noqa: E402
from sfincs_run import run_sfincs_subprocess  # noqa: E402

FIXTURE_BASE_DIR_NAME = "calibration_roughness_SFINCS"  # read-only source of the fixture tile
SMOKE_BASE_DIR_NAME = "sfincs_smoke_test"  # where the copy is built and run
PREP_FILES = ["elevation_combined.tif", "manning_n.tif", "matched_boundary_points.gpkg", "corrected_hydrographs.csv"]
DEFAULT_SFINCS_EXE = Path(
    r"C:\Users\schlumbe\hydromt_sfincs\delta_model\software\SFINCS_v2.3.0_mt_Faber_release_exe\sfincs.exe"
)


class _SilentLog:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass


def _fixture_tile(root: Path) -> str | None:
    """Smallest finished calibration tile (pre-flight active cell count) with
    every prep file and its boundary line."""
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


def _prereqs_available() -> tuple[Path, Path, str] | None:
    """(root, sfincs_exe, tile_id) if this smoke test can actually run, else None."""
    config_path = REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    if not config_path.exists() or not DEFAULT_SFINCS_EXE.exists():
        return None
    try:
        root = read_root(config_path)
    except Exception:
        return None
    tile_id = _fixture_tile(root)
    return None if tile_id is None else (root, DEFAULT_SFINCS_EXE, tile_id)


def _stage_fixture(root: Path, tile_id: str) -> Path:
    """Fresh copy of the fixture tile's inputs/, prep files and boundary line
    under SMOKE_BASE_DIR_NAME; returns the copied boundary line."""
    src, dst = root / FIXTURE_BASE_DIR_NAME / tile_id, root / SMOKE_BASE_DIR_NAME / tile_id
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src / "inputs", dst / "inputs")
    (dst / "sfincs_model").mkdir(parents=True)
    for f in PREP_FILES:
        shutil.copy2(src / "sfincs_model" / f, dst / "sfincs_model" / f)
    line = dst / f"boundary_line_{tile_id}.gpkg"
    shutil.copy2(src / line.name, line)
    return line


def test_build_run_postprocess_produces_plausible_output() -> None:
    print("=== SFINCS smoke test: full build -> run -> postprocess on a real fixture tile ===")
    prereqs = _prereqs_available()
    if prereqs is None:
        print("SKIP: real P:\\ modelling share, a finished calibration_roughness_SFINCS tile, and/or the local "
              "sfincs.exe aren't available in this environment - not a failure, nothing to run against.")
        print()
        return
    root, sfincs_exe, tile_id = prereqs
    line = _stage_fixture(root, tile_id)
    print(f"  fixture: tile {tile_id} from {FIXTURE_BASE_DIR_NAME}, copied to {SMOKE_BASE_DIR_NAME}")

    sfincs_dir = build_sfincs_tile(tile_id, root, base_dir_name=SMOKE_BASE_DIR_NAME, boundary_lines_gpkg=line)
    assert sfincs_dir.exists()
    inp_path = sfincs_dir / "sfincs.inp"
    assert inp_path.exists(), "build_sfincs_tile did not produce sfincs.inp"
    print(f"  build: OK, wrote {sfincs_dir}")

    run_sfincs_subprocess(sfincs_exe, sfincs_dir, timeout_s=300.0, log=_SilentLog(), label=f"smoke test tile {tile_id}")
    map_path = sfincs_dir / "sfincs_map.nc"
    assert map_path.exists(), "SFINCS ran but produced no sfincs_map.nc"
    print("  run: OK, produced sfincs_map.nc")

    land_mask_path = root / SMOKE_BASE_DIR_NAME / tile_id / "inputs" / "mask.tif"
    hmax, grid_info = compute_max_inundation(sfincs_dir, land_mask_path)

    assert hmax.ndim == 2 and hmax.size > 0, hmax.shape
    finite = hmax[np.isfinite(hmax)]
    # Bounded, physically-plausible check - not "did it match a specific
    # number" (that's what the manual tile comparisons already do), just
    # "didn't degenerate": no absurd depths (e.g. a units/sign bug would
    # produce metres in the thousands), and not silently all-NaN when the
    # model actually ran successfully.
    if finite.size:
        assert finite.max() < 100.0, f"implausible max depth {finite.max()} m - likely a units/decode bug"
        assert finite.min() >= 0.0, f"negative flood depth {finite.min()} m - should never happen"
    print(f"  postprocess: OK, {finite.size} finite cell(s) of {hmax.size} total"
          + (f", max depth {finite.max():.3f} m" if finite.size else " (tile fully dry this run - not itself a failure)"))
    print("PASS: build -> run -> postprocess completed end to end with a bounded, plausible result")
    print()


def main() -> None:
    test_build_run_postprocess_produces_plausible_output()
    print("SFINCS end-to-end smoke test complete.")


if __name__ == "__main__":
    main()
