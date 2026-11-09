"""Depth reliability of the eikonal against USGS CoSMoS in SF Bay (exposure/depth assessment 3.2).

From validate_sf_bay.py's saved model-grid rasters (agreement, model - CoSMoS depth) and the
merged model depth: on cells both flood above exposure.exceedance_threshold_m, the share within
10/20/30/50 cm, bias, MAE, RMSE and P90 per CoSMoS depth class, area- and population-weighted
(exposure_depth.py's statistics, 1 cm bins). Includes DEM error, unlike the SFINCS comparison.

Usage:
    python validation/sf_bay_depth_reliability.py --config <config.yml> --validation-dir <dir>/USA --out <csv>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from population_grid import cell_area_km2, population_per_pixel  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT / "sfincs_tiles"))
from flood_agreement import DIFF_EDGES, SFINCS_CLASS_EDGES, SFINCS_CLASS_LABELS, diff_stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox, _mosaic_read  # noqa: E402

TAG = "USA_sf_bay_RP100_SLR_0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-dir", required=True, help="the USA folder of validate_sf_bay.py's --out-dir")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    threshold = float(cfg["exposure"]["exceedance_threshold_m"])
    paths = cfg["paths"]["hydromt_data_catalog"]
    catalog = get_data_catalog([_REPO_ROOT / p for p in ([paths] if isinstance(paths, str) else paths)],
                               root=cfg["paths"]["root"])

    vdir = Path(args.validation_dir)
    with rasterio.open(vdir / f"agreement_{TAG}.tif") as src:
        agreement, transform, shape = src.read(1), src.transform, src.shape
        bbox = list(src.bounds)
    with rasterio.open(vdir / f"depth_diff_{TAG}.tif") as src:
        diff = src.read(1).astype(np.float64)
    chunks = _chunks_overlapping_bbox(bbox, float(cfg["postprocessing"]["chunk_size_deg"]),
                                      Path(cfg["postprocessing"]["merged_outputs"]) / "chunks", "RP100", "SLR_0")
    mosaic, mtr = _mosaic_read(chunks, bbox)
    model = np.full(shape, np.nan)
    reproject(np.where(mosaic < 0, np.nan, mosaic).astype(np.float64), model, src_transform=mtr,
              src_crs="EPSG:4326", dst_transform=transform, dst_crs="EPSG:4326", resampling=Resampling.nearest,
              src_nodata=np.nan, dst_nodata=np.nan)
    bench = model - diff
    both = (agreement == 1) & np.isfinite(diff) & (model > threshold) & (bench > threshold)

    lat = transform.f + transform.e * (np.arange(shape[0]) + 0.5)
    area = np.array([cell_area_km2(t - transform.e / 2, t + transform.e / 2, abs(transform.a)) for t in lat])
    weights = {"area": np.broadcast_to(area[:, None], shape),
               "pop": population_per_pixel(catalog.get_source("population").path, shape, transform, "EPSG:4326")[0]}
    full = np.concatenate(([-np.inf], DIFF_EDGES, [np.inf]))
    cls = np.digitize(bench[both], SFINCS_CLASS_EDGES)
    d = diff[both]
    rows = []
    for name, w in weights.items():
        wb = w[both]
        for c, label in enumerate(SFINCS_CLASS_LABELS):
            sel = cls == c
            rows.append({"weight": name, "cosmos_class_m": label,
                         **diff_stats(np.histogram(d[sel], bins=full, weights=wb[sel])[0])})
        rows.append({"weight": name, "cosmos_class_m": "all",
                     **diff_stats(np.histogram(d, bins=full, weights=wb)[0])})
    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    # Cumulative |model - CoSMoS| per denominator; a cell dry on one side (< threshold) is
    # compared at that side's own (sub-threshold) depth, exact from the difference raster.
    model_wet, bench_wet = model > threshold, bench > threshold
    masks = {"both_wet": both, "reference_wet": bench_wet & np.isfinite(diff),
             "either_wet": (model_wet | bench_wet) & np.isfinite(diff)}
    taus = np.round(np.arange(0.0, 2.001, 0.01), 2)
    cdf = {"tau_m": taus}
    for mname, mask in masks.items():
        for wname, w in weights.items():
            err, ww = np.abs(diff[mask]), w[mask]
            order = np.argsort(err)
            cum = np.cumsum(ww[order]) / ww.sum()
            cdf[f"{mname}_{wname}"] = np.interp(taus, err[order], cum, left=0.0, right=1.0)
    pd.DataFrame(cdf).to_csv(Path(args.out).with_name("sf_bay_depth_cdf.csv"), index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 220):
        print(f"{int(both.sum())} both-wet cells (> {threshold:g} m)")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
