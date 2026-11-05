"""Step 1c: COAST-RP return levels vs GEV fits to GESLA and GTSM annual maxima at the GESLA sites.

Total water level, relative to local MSL:
  gesla  observed record with each year's mean removed (the gauge datum and the MSL trend go;
         interannual MSL variability goes too, so the RP levels are those of a fixed MSL)
  gtsm   GTSM surge residual (1979-2024) + the GTSM tide re-predicted hourly from its harmonic
         fit (`levels_tides.py`), since the GTSM tide series only covers 1996-2014
COAST-RP is relative to GTSM's model MSL; its ETC-only and TC-only files are read alongside.
Annual maxima need >= YEAR_COVERAGE of a year's hours; GEV by maximum likelihood (scipy), with a
BOOTSTRAP-member nonparametric bootstrap for the 90% interval of each RP level.

`--msl linear` removes a linear trend from GESLA instead of the annual means, so interannual MSL
variability (ENSO, NAO, steric) stays in the annual maxima; GTSM, which has none, is not refitted.

Writes <out-dir>/extremes.parquet (one row per site and source, RP levels in m), or
<out-dir>/extremes_linear.parquet (GESLA only) with `--msl linear`.

Usage:
    python boundary_test/levels_extremes.py --out-dir .cleanup_baseline/level_sources [--workers 3] [--msl linear]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
from scipy.stats import genextreme
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "boundary_test"))
from extract_gtsm_series import load_series, station_series  # noqa: E402
from levels_tides import HOURLY_START, on_hourly_grid, remove_annual_means  # noqa: E402

INPUTS = Path("P:/11212688-004-global-floodmaps/modelling/inputs/COAST-RP_dataset")
RPS = (2, 10, 50, 100)
COAST_RP_RPS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)
YEAR_COVERAGE = 0.8
MIN_ANNUAL_MAXIMA = 15
BOOTSTRAP = 100
SITES_PER_TASK = 10
EARTH_RADIUS_KM = 6371.0
SAME_STATION_KM = 0.1


def annual_maxima(idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    year = (HOURLY_START + idx.astype("timedelta64[h]")).astype("datetime64[Y]").astype(int) + 1970
    s = pd.Series(z).groupby(year)
    counts = s.size()
    return s.max()[counts >= YEAR_COVERAGE * 8766].to_numpy()


def remove_linear_trend(idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    t = idx.astype(float)
    return z - np.polyval(np.polyfit(t, z, 1), t)


def gev_levels(am: np.ndarray, rng: np.random.Generator) -> dict:
    if len(am) < MIN_ANNUAL_MAXIMA:
        return {"n_annual_maxima": len(am)}
    p = 1.0 - 1.0 / np.asarray(RPS, dtype=float)
    c, loc, scale = genextreme.fit(am)
    est = genextreme.ppf(p, c, loc, scale)
    boot = []
    for _ in range(BOOTSTRAP):
        s = rng.choice(am, len(am), replace=True)
        try:
            boot.append(genextreme.ppf(p, *genextreme.fit(s, c, loc=loc, scale=scale)))
        except Exception:  # degenerate resample; the interval uses the others
            continue
    lo, hi = np.nanquantile(np.array(boot), [0.05, 0.95], axis=0)
    row = {"n_annual_maxima": len(am), "gev_shape": -c, "am_max_m": float(am.max())}
    for k, rp in enumerate(RPS):
        row.update({f"rp{rp}_m": est[k], f"rp{rp}_p05_m": lo[k], f"rp{rp}_p95_m": hi[k]})
    return row


def _site_task(args) -> list[dict]:
    out, sites, hc_gtsm, msl = args
    out = Path(out)
    b_re = np.load(out / "hourly_basis_re.npy", mmap_mode="r")
    b_im = np.load(out / "hourly_basis_im.npy", mmap_mode="r")
    surge_st, surge_time, surge = load_series(out, "surge")
    pos = dict(zip(surge_st.gtsm_index, range(len(surge_st))))
    surge_idx, _ = on_hourly_grid(surge_time.to_numpy(), np.zeros(len(surge_time)))
    rng = np.random.default_rng(0)
    rows = []
    for site, h in zip(sites, hc_gtsm):
        data = np.load(out / "gesla" / f"{site['file_name']}.npz")
        idx, z = on_hourly_grid(data["t"], data["z"])
        detrend = remove_annual_means if msl == "annual" else remove_linear_trend
        rows.append({"file_name": site["file_name"], "source": "gesla",
                     **(gev_levels(annual_maxima(idx, detrend(idx, z)), rng) if len(idx) else {})})
        if msl != "annual":
            continue
        x = station_series(surge, pos[site["gtsm_index"]])
        ok = np.isfinite(x)
        if np.isfinite(h).all() and ok.any():
            tide = np.asarray(b_re[surge_idx[ok]]) @ h.real + np.asarray(b_im[surge_idx[ok]]) @ h.imag
            rows.append({"file_name": site["file_name"], "source": "gtsm",
                         **gev_levels(annual_maxima(surge_idx[ok], tide + x[ok]), rng)})
    return rows


def coast_rp_levels(pairs: pd.DataFrame) -> pd.DataFrame:
    """RP levels of COAST-RP and its ETC/TC files at each site's COAST-RP station.

    The TC file has its own (smaller) station set, so every file is matched by coordinates to the
    station `levels_pairs.py` chose in COAST-RP.nc; a TC value is NaN where that station has none.
    """
    rows = {"file_name": pairs.file_name.to_numpy()}
    target = np.radians(pairs[["coast_rp_lat", "coast_rp_lon"]].to_numpy(dtype=float))
    for name, label in (("COAST-RP.nc", "coast_rp"), ("COAST-RP_ETC.nc", "coast_rp_etc"), ("COAST-RP_TC.nc", "coast_rp_tc")):
        with netCDF4.Dataset(INPUTS / name) as ds:
            ll = np.column_stack([ds["station_y_coordinate"][:], ds["station_x_coordinate"][:]]).astype(float)
            d, i = BallTree(np.radians(ll), metric="haversine").query(target, k=1)
            same = d[:, 0] * EARTH_RADIUS_KM <= SAME_STATION_KM
            for rp in COAST_RP_RPS:
                var = f"storm_tide_rp_{rp:04d}"
                if var in ds.variables:
                    rows[f"{label}_rp{rp}_m"] = np.where(same, np.ma.filled(ds[var][:], np.nan)[i[:, 0]], np.nan)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--msl", choices=("annual", "linear"), default="annual")
    args = parser.parse_args()
    out = Path(args.out_dir)
    pairs = pd.read_parquet(out / "pairs.parquet")
    hc = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    assert (hc["file_name"] == pairs.file_name.to_numpy()).all()
    sites = pairs[["file_name", "gtsm_index"]].to_dict("records")
    tasks = [(str(out), sites[i:i + SITES_PER_TASK], hc["gtsm_harm"][i:i + SITES_PER_TASK], args.msl)
             for i in range(0, len(sites), SITES_PER_TASK)]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(pool.map(_site_task, tasks)):
            rows += r
            if k % 10 == 0:
                print(f"  {min((k + 1) * SITES_PER_TASK, len(sites))}/{len(sites)} sites", flush=True)
    gev = pd.DataFrame(rows)
    if args.msl != "annual":
        gev.to_parquet(out / f"extremes_{args.msl}.parquet")
        return
    gev.to_parquet(out / "extremes_gev.parquet")
    table = gev.merge(coast_rp_levels(pairs), on="file_name", how="left")
    table.to_parquet(out / "extremes.parquet")
    print(table.groupby("source")[["n_annual_maxima", "rp100_m"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
