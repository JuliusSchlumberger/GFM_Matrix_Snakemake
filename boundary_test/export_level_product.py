"""Write the level product (§8.4) as production inputs: a COAST-RP-shaped levels file and hydrographs.

Levels: same stations, order, coordinates and `storm_tide_rp_XXXX` variables as COAST-RP.nc, so
`prepare_boundary_conditions.py` reads it unchanged through the `coast_rp` catalog entry (m above
local MSL; MDT, the 2020 offset and SLR are added there). Stations without a product level keep
COAST-RP's (`level_source`). RP1 is not produced; the pipeline does not use it.

Hydrographs: `storm_tide_hydrograph` (stations, return_period, time), 10-min steps around the
peak, NaN outside each station's window, for `duration_correction.coast_hg_path`
(`event_volume.load_coast_hg` slices it at the scenario's RP).

Usage:
    python boundary_test/export_level_product.py --out-dir .cleanup_baseline/level_product
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "boundary_test"))
from build_level_product import COAST_RP, RPS  # noqa: E402

LEVELS_FILE = "GFM_storm_tide_levels.nc"
HYDROGRAPHS_FILE = "GFM_storm_tide_hydrographs.nc"
STEP = np.timedelta64(600, "s")


def export_levels(out: Path) -> None:
    lev = pd.read_parquet(out / "levels.parquet")
    coast = xr.open_dataset(COAST_RP / "COAST-RP.nc")
    assert len(lev) == coast.sizes["stations"] and (lev.station_id.to_numpy() == coast.station_id.values.astype(str)).all()
    source = lev.level_source.fillna("coast_rp").to_numpy(dtype=object)
    data = {}
    for rp in RPS:
        z = lev[f"rp{rp}_m"].to_numpy(dtype=np.float64)
        fallback = coast[f"storm_tide_rp_{rp:04d}"].values.astype(np.float64)
        missing = ~np.isfinite(z)
        source[missing] = "coast_rp"
        data[f"storm_tide_rp_{rp:04d}"] = ("stations", np.where(missing, fallback, z),
                                           {"units": "m", "long_name": f"storm tide, return period {rp} y",
                                            "vertical_reference": "local mean sea level"})
    data["level_source"] = ("stations", source.astype(str))
    for col in ("r", "atlas", "gpd_shape", "tc_masked_share"):
        data[col] = ("stations", lev[col].to_numpy())
    ds = xr.Dataset(data, coords={"station_id": ("stations", coast.station_id.values),
                                  "station_x_coordinate": ("stations", coast.station_x_coordinate.values),
                                  "station_y_coordinate": ("stations", coast.station_y_coordinate.values)},
                    attrs={"title": "GFM storm-tide return levels (level product ii)",
                           "source": "boundary_test/build_level_product.py; INPUT_ANALYSIS.md §8.4",
                           "return_periods_y": list(RPS)})
    ds.to_netcdf(out / LEVELS_FILE)
    print(f"{LEVELS_FILE}: {len(lev)} stations, {int((source == 'coast_rp').sum())} with COAST-RP levels")


def export_hydrographs(out: Path) -> None:
    lev = pd.read_parquet(out / "levels.parquet", columns=["lon", "lat", "station_id"])
    hyd = np.load(out / "hydrographs.npy", mmap_mode="r")
    half = hyd.shape[2] // 2
    ds = xr.Dataset({"storm_tide_hydrograph": (("stations", "return_period", "time"), hyd,
                                               {"units": "m", "vertical_reference": "local mean sea level"})},
                    coords={"station_id": ("stations", lev.station_id.to_numpy()),
                            "station_x_coordinate": ("stations", lev.lon.to_numpy()),
                            "station_y_coordinate": ("stations", lev.lat.to_numpy()),
                            "return_period": list(RPS), "time": np.arange(-half, half + 1) * STEP},
                    attrs={"title": "GFM per-RP storm-tide hydrographs (accepted generator, §8.3)",
                           "source": "boundary_test/build_level_product.py"})
    enc = {"storm_tide_hydrograph": {"zlib": True, "complevel": 1, "chunksizes": (64, len(RPS), hyd.shape[2])}}
    ds.to_netcdf(out / HYDROGRAPHS_FILE, encoding=enc)
    print(f"{HYDROGRAPHS_FILE}: {hyd.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    out = Path(parser.parse_args().out_dir)
    export_levels(out)
    export_hydrographs(out)


if __name__ == "__main__":
    main()
