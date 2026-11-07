"""Skew surges at example stations before and after the storm-tide product's QA steps (§8.4).

For each station: the hourly FES tide and GTSM residual around its largest skew surge, and the
ranked skew surges under (a) every local tide maximum as a high water, raw residual, no TC mask;
(b) the product's rules: low-water bumps dropped (`skew_surge.HW_PROMINENCE`), despiked
residual, tropical-storm cycles removed (stations with a COAST-RP TC part). Reads the product
directory of `run_preparation.py storm_tide_product`; writes <product-dir>/qa_examples.npz for
the summary figure.

Usage:
    python boundary_test/level_product_qa.py --product-dir <storm_tide_product_dir> \
        [--config snakemake_workflow/config/config.yml]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from gtsm_series import load_series, station_series  # noqa: E402
from skew_surge import HW_PROMINENCE, despike, high_waters, skew_surges  # noqa: E402
from storm_tide_product import coast_rp_levels, tc_hours  # noqa: E402
from tide_prediction import HOURLY_START, fes2022_model, hourly_times, on_hourly_grid, predict, seasonal_columns  # noqa: E402

EXAMPLES = {18406: "Ungava Bay (double low waters)", 13187: "Arafura coast, New Guinea (GTSM spikes)",
            15320: "Mississippi delta (Katrina in the ETC record)"}
WINDOW_HR = 48
TOP_N = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--product-dir", required=True)
    parser.add_argument("--config", default=str(REPO / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--override", default=None, help="YAML merged over the config (e.g. a local ibtracs_path)")
    args = parser.parse_args()
    out = Path(args.product_dir)
    cfg = load_config(args.config, args.override)
    catalog = get_data_catalog(REPO / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    coast_rp = Path(catalog.get_source("coast_rp").path)
    stations = pd.read_csv(out / "stations.csv", index_col="station")
    st_tab, time, series = load_series(out / "cache" / "gtsm_surge_stations")
    pos = dict(zip(st_tab.gtsm_index, range(len(st_tab))))
    hc = np.load(out / "cache" / "fes_constants.npz")["fes2022"]
    hc[:, seasonal_columns(fes2022_model(out / "cache" / "fes2022"))] = 0.0
    b_re = np.load(out / "cache" / "fes2022_hourly_re.npy", mmap_mode="r")
    b_im = np.load(out / "cache" / "fes2022_hourly_im.npy", mmap_mode="r")
    idx, _ = on_hourly_grid(time.to_numpy(), np.zeros(len(time)))
    tc_ids = set(coast_rp_levels(coast_rp.parent / "COAST-RP_TC.nc").dropna().index)
    with_tc = stations.loc[[k for k in EXAMPLES if stations.station_id[k] in tc_ids]]
    masks = tc_hours(with_tc, len(hourly_times()), Path(cfg["storm_tide_product"]["ibtracs_path"]), HOURLY_START)
    result = {"stations": np.array(list(EXAMPLES)), "labels": np.array(list(EXAMPLES.values()))}
    for k in EXAMPLES:
        th = predict(b_re, b_im, hc[k])
        g = np.full(len(th), np.nan)
        g[idx] = station_series(series, pos[int(stations.gtsm_index[k])])
        hw, h = high_waters(th)
        old = skew_surges(g + th, hw, h)
        hw_new, h_new = high_waters(th, HW_PROMINENCE)
        new = skew_surges(despike(g) + th, hw_new, h_new)
        new = np.where(np.isin(hw_new, masks.get(k, [])), np.nan, new)
        top = hw[np.nanargmax(old)]
        win = slice(top - WINDOW_HR, top + WINDOW_HR + 1)
        result[f"{k}_tide"], result[f"{k}_residual"] = th[win], g[win]
        result[f"{k}_hw_old"] = hw[(hw >= win.start) & (hw < win.stop)] - top
        result[f"{k}_hw_new"] = hw_new[(hw_new >= win.start) & (hw_new < win.stop)] - top
        for name, s in (("old", old), ("new", new)):
            result[f"{k}_ranked_{name}"] = np.sort(s[np.isfinite(s)])[::-1][:TOP_N]
        print(f"{stations.station_id[k]}: largest skew surge {np.nanmax(old):.2f} -> {np.nanmax(new):.2f} m", flush=True)
    np.savez(out / "qa_examples.npz", **result)


if __name__ == "__main__":
    main()
