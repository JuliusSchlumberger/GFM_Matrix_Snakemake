"""Skew surges at example stations before and after the level product's QA steps (§8.4).

For each station: the hourly FES tide and GTSM residual around its largest skew surge, and the
ranked skew surges under (a) every local tide maximum as a high water, raw residual, no TC mask;
(b) prominent high waters only (HW_PROMINENCE), despiked residual, tropical-storm cycles removed
(stations with a COAST-RP TC part). Writes <out-dir>/qa_examples.npz for the summary figure.

Usage:
    python boundary_test/level_product_qa.py --out-dir .cleanup_baseline/level_product
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyTMD.io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from build_level_product import HW_PROMINENCE, coast_rp_levels, despike, tc_hours  # noqa: E402
from extract_gtsm_series import load_series, station_series  # noqa: E402
from fes_tide_stats import FES_MODEL, ensure_local_fes  # noqa: E402
from levels_tides import FES_CACHE, SEASONAL, on_hourly_grid  # noqa: E402
from surge_correction import high_waters, skew_surges  # noqa: E402

EXAMPLES = {18406: "Ungava Bay (double low waters)", 13187: "Arafura coast, New Guinea (GTSM spikes)",
            15320: "Mississippi delta (Katrina in the ETC record)"}
WINDOW_HR = 48
TOP_N = 300


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sources-dir", default=".cleanup_baseline/level_sources")
    args = parser.parse_args()
    out, sources = Path(args.out_dir), Path(args.sources_dir)
    stations = pd.read_csv(out / "stations.csv", index_col="station")
    st_tab, time, series = load_series(out, "surge")
    pos = dict(zip(st_tab.gtsm_index, range(len(st_tab))))
    hc = np.load(out / "fes_constants.npz")["fes2022"]
    cons = list(pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL).constituents)
    hc[:, [cons.index(c) for c in SEASONAL if c in cons]] = 0.0
    b_re = np.load(sources / "hourly_basis_re.npy", mmap_mode="r")
    b_im = np.load(sources / "hourly_basis_im.npy", mmap_mode="r")
    idx, _ = on_hourly_grid(time.to_numpy(), np.zeros(len(time)))
    tc_ids = set(coast_rp_levels("COAST-RP_TC.nc").dropna().index)
    masks = tc_hours(stations.loc[[k for k in EXAMPLES if stations.station_id[k] in tc_ids]], b_re.shape[0])
    result = {"stations": np.array(list(EXAMPLES)), "labels": np.array(list(EXAMPLES.values()))}
    for k in EXAMPLES:
        th = np.asarray(b_re) @ hc[k].real + np.asarray(b_im) @ hc[k].imag
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
