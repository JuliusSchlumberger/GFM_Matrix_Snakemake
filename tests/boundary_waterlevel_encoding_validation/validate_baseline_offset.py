"""Test that the 2020 baseline offset enters both the total and the static boundary level.

The static level is MDT + offset + (total(SLR) - total(SLR_0)); the offset cancels in the
difference, so it must come back from the MDT station file or SLR_0 would sit at the MDT epoch.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "preparation"))

import boundaries as bnd  # noqa: E402
import prepare_boundary_conditions as pbc  # noqa: E402

FILE_T = "wl_{return_period}_{waterlevel_name}.nc"
VAR_T = "wl_{return_period}_{waterlevel_name}"


def _write_case(tmp_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    coords = {"station_x_coordinate": ("stations", [0.0, 1.0, 2.0]),
              "station_y_coordinate": ("stations", [50.0, 50.0, 50.0])}
    storm_tide = np.array([2.0, 3.0, 1.5])
    mdt = np.array([0.30, np.nan, -0.10])
    offset = np.array([0.05, 0.04, 0.06])
    coast_rp = xr.Dataset({"storm_tide_rp_0100": ("stations", storm_tide)}, coords=coords)
    slr = xr.Dataset({"SLR_0mm": ("stations", np.zeros(3)), "SLR_500mm": ("stations", [0.55, 0.45, 0.50])}, coords=coords)
    pbc.combine_scenarios(coast_rp, slr, tmp_path, [100], [0.0, 0.5], FILE_T, VAR_T, mdt_arr=mdt, baseline_offset_arr=offset)
    mdt_path = tmp_path / "mdt.nc"
    xr.Dataset({"MDT": ("stations", mdt), "SLR_baseline_offset": ("stations", offset)}, coords=coords).to_netcdf(mdt_path)
    return storm_tide, np.nan_to_num(mdt) + offset, slr["SLR_500mm"].values, mdt_path


def test_total_and_static_levels_include_offset(tmp_path: Path) -> None:
    storm_tide, msl_2020, fingerprint, mdt_path = _write_case(tmp_path)
    with xr.open_dataset(tmp_path / FILE_T.format(return_period="RP100", waterlevel_name="SLR_0")) as ds:
        np.testing.assert_allclose(ds["wl_RP100_SLR_0"].values, storm_tide + msl_2020)
    _, static = bnd.static_station_levels(tmp_path, FILE_T, VAR_T, mdt_path, "RP100", "SLR_500")
    np.testing.assert_allclose(static, msl_2020 + fingerprint)


def test_static_level_refuses_mdt_file_without_offset(tmp_path: Path) -> None:
    _, _, _, mdt_path = _write_case(tmp_path)
    with xr.open_dataset(mdt_path) as ds:
        old = ds.drop_vars("SLR_baseline_offset").load()
    old_path = tmp_path / "mdt_old.nc"
    old.to_netcdf(old_path)
    with pytest.raises(KeyError, match="SLR_baseline_offset"):
        bnd.static_station_levels(tmp_path, FILE_T, VAR_T, old_path, "RP100", "SLR_500")
