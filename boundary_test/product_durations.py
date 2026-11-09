"""T(0.25) and T(0.5) of the level product's hydrographs against COAST-HG at every station (§8.4).

Production with COAST-HG uses its RP100 average-tide hydrograph for every RP; the level product
has one hydrograph per RP. Both are measured as in `generator_validation.profile_measures`
(hourly +-72 h profile around the maximum, normalised by it; T(f) = hours within f x peak of the
peak). Writes <out-dir>/durations.parquet (one row per station: COAST-HG average and spring, and
the product at each RP).

Usage:
    python boundary_test/product_durations.py --out-dir <storm_tide_product_dir>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from storm_tide_product import RPS  # noqa: E402
from generator_validation import hourly_profile, profile_measures  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sfincs_tiles"))
from gfm_config import read_root  # noqa: E402

DATA_ROOT = read_root(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")

COAST_HG = DATA_ROOT / "inputs" / "COAST_HG" / "COAST-HG_RP100.nc"
MEASURES = ("t025_hr", "t05_hr")
CHUNK = 2000


def _measures(series: np.ndarray) -> dict:
    if not np.isfinite(series).any():
        return dict.fromkeys(MEASURES, np.nan)
    m = profile_measures(hourly_profile(np.where(np.isfinite(series), series, -np.inf)))
    return {k: m[k] for k in MEASURES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    out = Path(parser.parse_args().out_dir)
    lev = pd.read_parquet(out / "levels.parquet", columns=["station_id", "lon", "lat", "mhws_m"])
    hyd = np.load(out / "hydrographs.npy", mmap_mode="r")
    rows = [dict() for _ in range(len(lev))]
    with xr.open_dataset(COAST_HG) as hg:
        assert (hg.station_id.values.astype(str) == lev.station_id.to_numpy().astype(str)).all()
        for start in range(0, len(lev), CHUNK):
            sl = slice(start, start + CHUNK)
            avg = hg.hydrograph_average_tide_signal.isel(stations=sl).values.astype(np.float64)
            spr = hg.hydrograph_spring_tide_signal.isel(stations=sl).values.astype(np.float64)
            block = np.asarray(hyd[sl], dtype=np.float64)
            for i in range(block.shape[0]):
                row = rows[start + i]
                for name, s in (("coast_hg", avg[i]), ("coast_hg_spring", spr[i])):
                    row.update({f"{name}_{k}": v for k, v in _measures(s).items()})
                for j, rp in enumerate(RPS):
                    row.update({f"rp{rp}_{k}": v for k, v in _measures(block[i, j]).items()})
            print(f"  {min(start + CHUNK, len(lev))}/{len(lev)}", flush=True)
    res = lev.join(pd.DataFrame(rows, index=lev.index))
    res.to_parquet(out / "durations.parquet")
    for col in ("coast_hg_t025_hr", "rp2_t025_hr", "rp100_t025_hr", "coast_hg_t05_hr", "rp100_t05_hr"):
        print(f"{col}: median {res[col].median():.1f} h")


if __name__ == "__main__":
    main()
