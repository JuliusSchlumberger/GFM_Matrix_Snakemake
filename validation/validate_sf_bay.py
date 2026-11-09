"""Validate SF Bay flood depth against USGS CoSMoS (RP100, SLR000) at two grids.

A) "coarsen": CoSMoS 2m EPSG:26910 depth averaged onto the model grid (~30m, EPSG:4326).
B) "refine": model depth bilinear-resampled onto the CoSMoS 2m grid.
Each gives HT/FAR/CSI/bias at primary_threshold_m and RMSE/MAE (m) over the whole
domain and over benchmark-wet pixels only. Grid A also gets validate_country.py's
exposure columns (v.exposure_columns; both depth maps thresholded at each threshold).

Domain: union of the 9 counties' rectangular raster extents
(sf_bay_county_bbox_union.gpkg). CoSMoS only has values inside the flood extent,
so its nodata inside the union counts as dry (depth 0). Permanent water is
excluded: CoSMoS depth includes Bay bathymetry.

With --out-dir, also writes the model-grid (A) agreement raster (validate_country.py's
codes), the model-minus-CoSMoS depth difference (m), and the metrics CSV. --out-dir has
NO default - a custom --bbox/--benchmark-path sample must never silently collide with
production's own {validation.output_dir}/USA/<same filenames> (TODO.md incident,
2026-11-07); pass --out-dir explicitly for both the true production run and any sample.
Without --out-dir, only the metrics are printed (and written to --out if given). Plot
the model-grid rasters with plot_sf_bay.py.

Defaults reproduce the full 9-county production behaviour exactly. --bbox/--benchmark-path
override the scoring window and benchmark raster (e.g. a county-clipped CoSMoS extent for a
cheap local-input sample); --native-merged-path reads the model depth directly from one
native-CRS merged file (merge_england_local_tiles.py's output) instead of the standard
EPSG:4326 merged chunks - for a local-input scenario whose DEM (and so its merge) is not in
EPSG:4326 (e.g. SF Bay's NOAA LIDAR DEM, EPSG:26910). Because that merge can be at a far
finer resolution than the 2m benchmark over a county-sized extent (e.g. 1m NOAA LIDAR over
all of Alameda - a ~57000x47000 array, ~21 GiB as float64), --native-merged-path scores both
grids window by window (_iter_subwindows) and accumulates confusion/error sums rather than
ever holding a full-resolution array - the plain chunk-based default path stays as a single
eager read (it is already small: DeltaDTM's own ~30m grid).

Usage:
    python validation/validate_sf_bay.py --out-dir <dir> [--config path/to/config.yml] [--out path.csv]
    python validation/validate_sf_bay.py --config <sf_bay_local_input_materialized.yml> \\
        --bbox -122.355 37.45 -121.99 37.91 \\
        --benchmark-path "zip://.../alameda_flooding_slr000.zip!flooding/v2.1/county_alameda_flddepth_slr000_w100.tif" \\
        --benchmark-scale-m 0.01 \\
        --native-merged-path <sf_bay_local_input merged_results/local_native/waterdepth_RP100_SLR_0.tif>
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.crs
import rasterio.transform
import rasterio.windows
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject, transform_bounds

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from population_grid import population_per_pixel  # noqa: E402
import validation as v  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _chunks import _chunks_overlapping_bbox, _mosaic_read  # noqa: E402
from validate_country import _CAT_AGREE, _CAT_DRY, _CAT_OVER, _CAT_UNDER  # noqa: E402


def _rasterize_surveyed_domain(surveyed_domain_path: Path, out_transform, out_shape, out_crs) -> np.ndarray:
    gdf = gpd.read_file(surveyed_domain_path).to_crs(out_crs)
    return rasterize(
        [(geom, 1) for geom in gdf.geometry], out_shape=out_shape,
        transform=out_transform, fill=0, dtype="uint8",
    ).astype(bool)


def _permanent_water_mask_lean(land_use: np.ndarray, exclude_codes) -> np.ndarray:
    """Like validation.permanent_water_mask without np.isin, which runs out of memory at 2m."""
    out = np.zeros(land_use.shape, dtype=bool)
    for code in exclude_codes:
        out |= land_use == code
    return out

NODATA_SENTINEL = -9999.0
# EPSG:4326, footprint of the v2.1 sfb_fldhazd benchmark.
BAY_BBOX_4326 = (-122.64, 37.40, -121.88, 38.28)
OUTPUT_TAG = "USA_sf_bay_RP100_SLR_0"
AGREEMENT_NODATA = 255
# Window side (px) for the native-merge windowed scorer - keeps every per-window array a
# few tens of MB regardless of the merge's full extent (TODO.md, 2026-11-07 OOM incident).
_WINDOW_PX = 2048


def _iter_subwindows(window: rasterio.windows.Window, size: int = _WINDOW_PX):
    row0, col0 = int(window.row_off), int(window.col_off)
    height, width = int(window.height), int(window.width)
    for r in range(0, height, size):
        h = min(size, height - r)
        for c in range(0, width, size):
            w = min(size, width - c)
            yield rasterio.windows.Window(col0 + c, row0 + r, w, h)


class _Accumulator:
    """Running confusion-matrix counts and squared/absolute depth-error sums.

    Summing disjoint windows this way is exact (not an approximation): HT/FAR/CSI/bias
    depend only on the aggregate tp/fp/fn/tn, and RMSE/MAE on the aggregate sum of
    squared/absolute error and pixel count.
    """

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.n_domain = 0
        self.sq_all = self.abs_all = 0.0
        self.n_all = 0
        self.sq_fl = self.abs_fl = 0.0
        self.n_fl = 0
        self.exposure: dict[str, float] = defaultdict(float)

    def add(self, model_depth: np.ndarray, bench_depth: np.ndarray, model_wet: np.ndarray,
            bench_wet: np.ndarray, domain: np.ndarray) -> None:
        d = domain
        self.tp += float((model_wet & bench_wet & d).sum())
        self.fp += float((model_wet & ~bench_wet & d).sum())
        self.fn += float((~model_wet & bench_wet & d).sum())
        self.tn += float((~model_wet & ~bench_wet & d).sum())
        self.n_domain += int(d.sum())

        err = model_depth - bench_depth
        err_all = err[d]
        self.sq_all += float(np.sum(err_all ** 2))
        self.abs_all += float(np.sum(np.abs(err_all)))
        self.n_all += int(err_all.size)

        flooded = d & bench_wet
        err_fl = err[flooded]
        self.sq_fl += float(np.sum(err_fl ** 2))
        self.abs_fl += float(np.sum(np.abs(err_fl)))
        self.n_fl += int(err_fl.size)

    def add_exposure(self, model_depth: np.ndarray, bench_depth: np.ndarray, model_wet: np.ndarray,
                     bench_wet: np.ndarray, domain: np.ndarray, area_km2, people: np.ndarray,
                     exposure_threshold_m: float) -> None:
        for k, val in _exposure_counts(model_depth, bench_depth, model_wet, bench_wet, domain, area_km2,
                                       people, exposure_threshold_m).items():
            self.exposure[k] += val

    def metrics(self) -> dict:
        m = v.metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        rmse_all = float(np.sqrt(self.sq_all / self.n_all)) if self.n_all else np.nan
        mae_all = float(self.abs_all / self.n_all) if self.n_all else np.nan
        rmse_fl = float(np.sqrt(self.sq_fl / self.n_fl)) if self.n_fl else np.nan
        mae_fl = float(self.abs_fl / self.n_fl) if self.n_fl else np.nan
        return {
            "n_domain_px": self.n_domain, "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            # Output column is HT (hit rate); validation.py calls it HR.
            "HT": m["HR"], "FAR": m["FAR"], "CSI": m["CSI"], "bias": m["bias"],
            "RMSE_all_m": rmse_all, "MAE_all_m": mae_all, "n_all_px": self.n_all,
            "RMSE_flooded_m": rmse_fl, "MAE_flooded_m": mae_fl, "n_flooded_px": self.n_fl,
            **(v.exposure_columns(self.exposure) if self.exposure else {}),
        }


def _exposure_counts(model_depth, bench_depth, model_wet, bench_wet, domain, area_km2, people,
                     exposure_threshold_m: float) -> dict[str, float]:
    # CoSMoS is a depth benchmark, so at the exposure threshold both maps are thresholded.
    return v.exposure_counts(
        model_wet, model_depth > exposure_threshold_m, bench_wet.astype("float64"), domain, area_km2, people,
        exposure_fraction=(bench_depth > exposure_threshold_m).astype("float64"),
    )


def _model_grid_weights(population_path, shape, transform, crs):
    """(km2 per pixel, people per pixel) on a model grid in any CRS."""
    if rasterio.crs.CRS.from_user_input(crs).is_geographic:
        area_km2 = pixel_area_km2_grid(transform, shape[1], shape[0])
    else:
        area_km2 = abs(transform.a * transform.e) / 1e6
    people = population_per_pixel(population_path, shape, transform, crs,
                                  open_raster=partial(retry_transient_io, rasterio.open))[0]
    return area_km2, people


def _write_model_grid_outputs(out_dir: Path, model_depth, bench_depth, model_wet, bench_wet, domain, transform, crs) -> None:
    """Write agreement categories and model-minus-CoSMoS depth (m) on the model grid.

    The depth difference is NaN where both are dry, so the map shows the background there.
    """
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    agreement = np.full(domain.shape, AGREEMENT_NODATA, dtype="uint8")
    agreement[domain] = _CAT_DRY
    agreement[domain & model_wet & bench_wet] = _CAT_AGREE
    agreement[domain & ~model_wet & bench_wet] = _CAT_UNDER
    agreement[domain & model_wet & ~bench_wet] = _CAT_OVER
    diff = np.where(domain & (model_wet | bench_wet), model_depth - bench_depth, np.nan).astype("float32")
    base = {"driver": "GTiff", "count": 1, "crs": crs, "transform": transform,
            "width": domain.shape[1], "height": domain.shape[0], "compress": "lzw"}
    for name, arr, extra in (
        ("agreement", agreement, {"dtype": "uint8", "nodata": AGREEMENT_NODATA}),
        ("depth_diff", diff, {"dtype": "float32", "nodata": float("nan")}),
    ):
        path = out_dir / f"{name}_{OUTPUT_TAG}.tif"
        with retry_transient_io(rasterio.open, path, "w", **base, **extra) as dst:
            dst.write(arr, 1)
        print(f"wrote {path}")


def _read_model_bay(chunk_paths: list[Path], bbox):
    if not chunk_paths:
        raise FileNotFoundError(f"no merged waterdepth chunk overlaps the SF Bay bbox {bbox}")
    if len(chunk_paths) == 1:
        return _read_model_window(chunk_paths[0], bbox)
    mosaic, transform = _mosaic_read(chunk_paths, list(bbox))
    with rasterio.open(chunk_paths[0]) as src:
        crs = src.crs
    return mosaic.astype("float64"), transform, crs, -9999.0


def _model_native_window(path: Path, bbox_4326):
    """Return the native-merge dataset's read window/transform/crs/nodata for `bbox_4326`.

    No pixel data is read - the windowed scorers below read the file block by block.
    """
    with rasterio.open(path) as src:
        native_bbox = bbox_4326 if str(src.crs) == "EPSG:4326" else transform_bounds("EPSG:4326", src.crs, *bbox_4326)
        window = rasterio.windows.from_bounds(*native_bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        # bbox can extend past what the merge actually covers (e.g. a county-wide benchmark
        # bbox vs. a local-input sample's smaller simulated-tile footprint) - clip to the
        # dataset's real extent so every later sub-window read comes back full-sized.
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        crs = src.crs
        nodata = src.nodata
    return window, crs, nodata, (int(window.height), int(window.width))


def _score_coarsen_windowed(
    model_path: Path, ds_window: rasterio.windows.Window, model_shape_native: tuple[int, int],
    bench_depth_2m: np.ndarray, bench_transform_2m, bench_crs_2m,
    gfm_catalog, permanent_water_source, permanent_water_codes,
    surveyed_domain_path: Path, threshold_m: float, out_dir: Path | None,
    population_path: str, exposure_threshold_m: float,
):
    """Grid A ("coarsen"), windowed over the model's own native-resolution grid."""
    acc = _Accumulator()
    with rasterio.open(model_path) as model_src:
        model_crs = model_src.crs
        model_nodata = model_src.nodata
        model_transform_native = model_src.window_transform(ds_window)

        agree_dst = diff_dst = None
        if out_dir is not None:
            retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
            base = {"driver": "GTiff", "count": 1, "crs": model_crs, "transform": model_transform_native,
                    "width": model_shape_native[1], "height": model_shape_native[0], "compress": "lzw",
                    "tiled": True, "blockxsize": 512, "blockysize": 512}
            agree_dst = retry_transient_io(rasterio.open, out_dir / f"agreement_{OUTPUT_TAG}.tif", "w",
                                            **base, dtype="uint8", nodata=AGREEMENT_NODATA)
            diff_dst = retry_transient_io(rasterio.open, out_dir / f"depth_diff_{OUTPUT_TAG}.tif", "w",
                                           **base, dtype="float32", nodata=float("nan"))
        try:
            for win in _iter_subwindows(ds_window):
                depth = model_src.read(1, window=win).astype("float64")
                win_transform = model_src.window_transform(win)
                shape = (win.height, win.width)
                valid = depth != model_nodata if model_nodata is not None else np.isfinite(depth)
                depth = np.where(valid, depth, NODATA_SENTINEL)

                bench_on_win = np.full(shape, NODATA_SENTINEL, dtype="float32")
                reproject(
                    source=bench_depth_2m, destination=bench_on_win,
                    src_transform=bench_transform_2m, src_crs=bench_crs_2m,
                    dst_transform=win_transform, dst_crs=model_crs,
                    src_nodata=NODATA_SENTINEL, dst_nodata=NODATA_SENTINEL,
                    resampling=Resampling.average,
                )
                win_bbox_4326 = transform_bounds(
                    model_crs, "EPSG:4326", *rasterio.transform.array_bounds(win.height, win.width, win_transform),
                )
                water_win = _permanent_water_on_grid(
                    gfm_catalog, permanent_water_source, permanent_water_codes,
                    list(win_bbox_4326), win_transform, shape, model_crs,
                )
                surveyed_win = _rasterize_surveyed_domain(surveyed_domain_path, win_transform, shape, model_crs)
                bench_on_win = np.where((bench_on_win == NODATA_SENTINEL) & surveyed_win, 0.0, bench_on_win)
                domain_win = (depth != NODATA_SENTINEL) & surveyed_win & ~water_win
                model_wet_win = depth > threshold_m
                bench_wet_win = bench_on_win > threshold_m
                acc.add(depth, bench_on_win, model_wet_win, bench_wet_win, domain_win)
                area_km2, people = _model_grid_weights(population_path, shape, win_transform, model_crs)
                acc.add_exposure(depth, bench_on_win, model_wet_win, bench_wet_win, domain_win, area_km2, people,
                                 exposure_threshold_m)

                if agree_dst is not None:
                    out_win = rasterio.windows.Window(
                        win.col_off - ds_window.col_off, win.row_off - ds_window.row_off, win.width, win.height,
                    )
                    agreement = np.full(shape, AGREEMENT_NODATA, dtype="uint8")
                    agreement[domain_win] = _CAT_DRY
                    agreement[domain_win & model_wet_win & bench_wet_win] = _CAT_AGREE
                    agreement[domain_win & ~model_wet_win & bench_wet_win] = _CAT_UNDER
                    agreement[domain_win & model_wet_win & ~bench_wet_win] = _CAT_OVER
                    diff = np.where(domain_win & (model_wet_win | bench_wet_win), depth - bench_on_win, np.nan).astype("float32")
                    agree_dst.write(agreement, 1, window=out_win)
                    diff_dst.write(diff, 1, window=out_win)
        finally:
            if agree_dst is not None:
                agree_dst.close()
                diff_dst.close()
                print(f"wrote {out_dir / f'agreement_{OUTPUT_TAG}.tif'}")
                print(f"wrote {out_dir / f'depth_diff_{OUTPUT_TAG}.tif'}")
    return acc, model_transform_native


def _score_refine_windowed(
    model_path: Path, model_crs, model_nodata,
    bench_depth_2m: np.ndarray, bench_transform_2m, bench_crs_2m, bench_shape_2m: tuple[int, int],
    gfm_catalog, permanent_water_source, permanent_water_codes,
    surveyed_domain_path: Path, threshold_m: float,
):
    """Grid B ("refine"), windowed over the benchmark's own 2m grid."""
    acc = _Accumulator()
    full_win = rasterio.windows.Window(0, 0, bench_shape_2m[1], bench_shape_2m[0])
    with rasterio.open(model_path) as model_src:
        model_band = rasterio.band(model_src, 1)
        for win in _iter_subwindows(full_win):
            win_transform = rasterio.windows.transform(win, bench_transform_2m)
            shape = (win.height, win.width)
            bench_depth_win = bench_depth_2m[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]

            model_on_win = np.full(shape, NODATA_SENTINEL, dtype="float32")
            reproject(
                source=model_band, destination=model_on_win,
                dst_transform=win_transform, dst_crs=bench_crs_2m,
                src_nodata=model_nodata, dst_nodata=NODATA_SENTINEL,
                resampling=Resampling.bilinear,
            )
            win_bbox_4326 = transform_bounds(
                bench_crs_2m, "EPSG:4326", *rasterio.transform.array_bounds(win.height, win.width, win_transform),
            )
            water_win = _permanent_water_on_grid(
                gfm_catalog, permanent_water_source, permanent_water_codes,
                list(win_bbox_4326), win_transform, shape, bench_crs_2m,
            )
            surveyed_win = _rasterize_surveyed_domain(surveyed_domain_path, win_transform, shape, bench_crs_2m)
            bench_depth_win = np.where((bench_depth_win == NODATA_SENTINEL) & surveyed_win, 0.0, bench_depth_win)
            domain_win = (model_on_win != NODATA_SENTINEL) & surveyed_win & ~water_win
            model_wet_win = model_on_win > threshold_m
            bench_wet_win = bench_depth_win > threshold_m
            acc.add(model_on_win, bench_depth_win, model_wet_win, bench_wet_win, domain_win)
    return acc


def _read_model_window(path: Path, bbox):
    with rasterio.open(path) as src:
        window = rasterio.windows.from_bounds(*bbox, transform=src.transform)
        window = window.round_offsets().round_lengths()
        depth = src.read(1, window=window).astype("float64")
        nodata = src.nodata
        transform = src.window_transform(window)
        crs = src.crs
    return depth, transform, crs, nodata


def _read_benchmark_full(path: Path, scale_m: float = 1.0):
    """Read the whole benchmark depth raster, converted to metres via `scale_m`.

    The full 9-county mosaic is already float32 metres (scale_m=1.0); USGS CoSMoS's
    per-county extract files (e.g. the Alameda zip) are int32 centimetres - pass
    scale_m=0.01 for those (confirmed empirically, TODO.md 2026-11-07: the mosaic's
    values near Alameda are ~1.33/1.19/..., the per-county file's are ~133/119/...
    at the same locations).
    """
    with rasterio.open(path) as src:
        depth = src.read(1).astype("float32")
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
    is_nodata = np.isnan(depth) if nodata is None or np.isnan(nodata) else depth == nodata
    depth = np.where(is_nodata, NODATA_SENTINEL, depth * scale_m)
    return depth, transform, crs


def _permanent_water_on_grid(gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, out_shape, out_crs):
    """Like validation.read_permanent_water_mask, which hardcodes EPSG:4326, for any CRS (float32, lean mask)."""
    dst = v._read_categorical_on_grid(
        gfm_catalog, permanent_water_source, bbox, out_transform, out_shape, dst_crs=out_crs, dtype="float32",
    )
    if dst is None:
        return np.zeros(out_shape, dtype=bool)
    return _permanent_water_mask_lean(dst, tuple(permanent_water_codes))


def _confusion_metrics(model_wet: np.ndarray, bench_wet: np.ndarray, domain: np.ndarray) -> dict:
    d = domain
    tp = float((model_wet & bench_wet & d).sum())
    fp = float((model_wet & ~bench_wet & d).sum())
    fn = float((~model_wet & bench_wet & d).sum())
    tn = float((~model_wet & ~bench_wet & d).sum())
    m = v.metrics_from_counts(tp, fp, fn, tn)
    # Output column is HT (hit rate); validation.py calls it HR.
    return {"n_domain_px": int(d.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "HT": m["HR"], "FAR": m["FAR"], "CSI": m["CSI"], "bias": m["bias"]}


def _depth_error_metrics(model_depth: np.ndarray, bench_depth: np.ndarray, domain: np.ndarray, bench_wet: np.ndarray) -> dict:
    err_all = (model_depth - bench_depth)[domain]
    rmse_all = float(np.sqrt(np.mean(err_all ** 2))) if err_all.size else np.nan
    mae_all = float(np.mean(np.abs(err_all))) if err_all.size else np.nan

    flooded = domain & bench_wet
    err_fl = (model_depth - bench_depth)[flooded]
    rmse_fl = float(np.sqrt(np.mean(err_fl ** 2))) if err_fl.size else np.nan
    mae_fl = float(np.mean(np.abs(err_fl))) if err_fl.size else np.nan

    return {
        "RMSE_all_m": rmse_all, "MAE_all_m": mae_all, "n_all_px": int(err_all.size),
        "RMSE_flooded_m": rmse_fl, "MAE_flooded_m": mae_fl, "n_flooded_px": int(err_fl.size),
    }


def validate_sf_bay(
    config_path: str, threshold_m: float | None = None, out_dir: Path | None = None,
    bbox: tuple[float, float, float, float] = BAY_BBOX_4326,
    benchmark_path: str | Path | None = None,
    benchmark_scale_m: float = 1.0,
    native_merged_path: str | Path | None = None,
) -> dict[str, dict]:
    """Score both directions; `threshold_m` defaults to validation.primary_threshold_m.

    With `out_dir`, also writes the model-grid rasters (_write_model_grid_outputs). `bbox`/
    `benchmark_path` default to the full 9-county window (already metres - leave
    `benchmark_scale_m` at 1.0); pass 0.01 for a USGS CoSMoS per-county extract file
    (int32 centimetres - see `_read_benchmark_full`). `native_merged_path` reads the
    model depth from one native-CRS merged file instead of the standard EPSG:4326 chunks,
    windowed to bound memory (see module docstring).
    """
    cfg = load_config(Path(config_path))
    val_cfg = cfg["validation"]
    threshold_m = threshold_m if threshold_m is not None else val_cfg["primary_threshold_m"]
    _catalog_paths = cfg["paths"]["hydromt_data_catalog"]
    _catalog_paths = (
        _REPO_ROOT / _catalog_paths if isinstance(_catalog_paths, str)
        else [_REPO_ROOT / p for p in _catalog_paths]
    )
    gfm_catalog = get_data_catalog(_catalog_paths, root=cfg["paths"]["root"])
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    population_path = gfm_catalog.get_source("population").path
    exposure_threshold_m = float(cfg["exposure"]["exceedance_threshold_m"])

    benchmark_root = Path(val_cfg["benchmark_root"])
    surveyed_domain_path = benchmark_root / "California" / "flooding" / "sf_bay_county_bbox_union.gpkg"
    if benchmark_path is None:
        benchmark_path = benchmark_root / "California" / "flooding" / "sf_bay_flddepth_slr000_w100_mosaic.tif"

    if native_merged_path is not None:
        native_merged_path = Path(native_merged_path)
        ds_window, model_crs, model_nodata, model_shape_native = _model_native_window(native_merged_path, bbox)
        bench_depth_2m, bench_transform_2m, bench_crs_2m = _read_benchmark_full(benchmark_path, benchmark_scale_m)
        bench_shape_2m = bench_depth_2m.shape

        acc_a, model_transform_native = _score_coarsen_windowed(
            native_merged_path, ds_window, model_shape_native,
            bench_depth_2m, bench_transform_2m, bench_crs_2m,
            gfm_catalog, permanent_water_source, permanent_water_codes,
            surveyed_domain_path, threshold_m, Path(out_dir) if out_dir is not None else None,
            population_path, exposure_threshold_m,
        )
        acc_b = _score_refine_windowed(
            native_merged_path, model_crs, model_nodata,
            bench_depth_2m, bench_transform_2m, bench_crs_2m, bench_shape_2m,
            gfm_catalog, permanent_water_source, permanent_water_codes,
            surveyed_domain_path, threshold_m,
        )
        return {
            "A_coarsen_to_model_grid": {
                "resolution": f"{abs(model_transform_native.a):.4f} m (~model native, {model_shape_native})",
                **acc_a.metrics(),
            },
            "B_refine_model_to_2m": {
                "resolution": f"2m ({bench_crs_2m}, {bench_shape_2m})",
                **acc_b.metrics(),
            },
        }

    model_chunk_paths = _chunks_overlapping_bbox(
        list(bbox), cfg["postprocessing"]["chunk_size_deg"],
        Path(cfg["postprocessing"]["merged_outputs"]) / "chunks", "RP100", "SLR_0",
    )
    model_depth_native, model_transform_native, model_crs, model_nodata = _read_model_bay(model_chunk_paths, bbox)

    bench_depth_2m, bench_transform_2m, bench_crs_2m = _read_benchmark_full(benchmark_path, benchmark_scale_m)
    bench_shape_2m = bench_depth_2m.shape

    model_shape_native = model_depth_native.shape
    model_valid_native = model_depth_native != model_nodata if model_nodata is not None else np.isfinite(model_depth_native)
    model_depth_native = np.where(model_valid_native, model_depth_native, NODATA_SENTINEL)

    results: dict[str, dict] = {}

    bench_on_model = np.full(model_shape_native, NODATA_SENTINEL, dtype="float64")
    reproject(
        source=bench_depth_2m, destination=bench_on_model,
        src_transform=bench_transform_2m, src_crs=bench_crs_2m,
        dst_transform=model_transform_native, dst_crs=model_crs,
        src_nodata=NODATA_SENTINEL, dst_nodata=NODATA_SENTINEL,
        resampling=Resampling.average,
    )
    water_on_model = _permanent_water_on_grid(
        gfm_catalog, permanent_water_source, permanent_water_codes,
        list(bbox), model_transform_native, model_shape_native, model_crs,
    )
    surveyed_a = _rasterize_surveyed_domain(surveyed_domain_path, model_transform_native, model_shape_native, model_crs)
    # CoSMoS nodata inside the surveyed domain = modeled dry.
    bench_on_model = np.where((bench_on_model == NODATA_SENTINEL) & surveyed_a, 0.0, bench_on_model)
    domain_a = (model_depth_native != NODATA_SENTINEL) & surveyed_a & ~water_on_model
    model_wet_a = model_depth_native > threshold_m
    bench_wet_a = bench_on_model > threshold_m
    results["A_coarsen_to_model_grid"] = {
        "resolution": f"{abs(model_transform_native.a):.7f} deg (~model native, {model_shape_native})",
        **_confusion_metrics(model_wet_a, bench_wet_a, domain_a),
        **_depth_error_metrics(model_depth_native, bench_on_model, domain_a, bench_wet_a),
        **v.exposure_columns(_exposure_counts(
            model_depth_native, bench_on_model, model_wet_a, bench_wet_a, domain_a,
            *_model_grid_weights(population_path, model_shape_native, model_transform_native, model_crs),
            exposure_threshold_m,
        )),
    }
    if out_dir is not None:
        _write_model_grid_outputs(Path(out_dir), model_depth_native, bench_on_model, model_wet_a, bench_wet_a,
                                  domain_a, model_transform_native, model_crs)

    model_on_2m = np.full(bench_shape_2m, NODATA_SENTINEL, dtype="float32")
    reproject(
        source=model_depth_native, destination=model_on_2m,
        src_transform=model_transform_native, src_crs=model_crs,
        dst_transform=bench_transform_2m, dst_crs=bench_crs_2m,
        src_nodata=NODATA_SENTINEL, dst_nodata=NODATA_SENTINEL,
        resampling=Resampling.bilinear,
    )
    water_on_2m = _permanent_water_on_grid(
        gfm_catalog, permanent_water_source, permanent_water_codes,
        list(bbox), bench_transform_2m, bench_shape_2m, bench_crs_2m,
    )
    surveyed_b = _rasterize_surveyed_domain(surveyed_domain_path, bench_transform_2m, bench_shape_2m, bench_crs_2m)
    bench_depth_2m = np.where((bench_depth_2m == NODATA_SENTINEL) & surveyed_b, 0.0, bench_depth_2m)
    domain_b = (model_on_2m != NODATA_SENTINEL) & surveyed_b & ~water_on_2m
    model_wet_b = model_on_2m > threshold_m
    bench_wet_b = bench_depth_2m > threshold_m
    results["B_refine_model_to_2m"] = {
        "resolution": f"2m (EPSG:26910, {bench_shape_2m})",
        **_confusion_metrics(model_wet_b, bench_wet_b, domain_b),
        **_depth_error_metrics(model_on_2m, bench_depth_2m, domain_b, bench_wet_b),
    }

    return results


def main() -> None:
    """Print the SF Bay results; with --out-dir, also write rasters + metrics there."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out-dir", default=None,
                        help="write the model-grid rasters + metrics CSV here - no default "
                             "(a custom --bbox/--benchmark-path sample must never silently "
                             "land on production's own {validation.output_dir}/USA)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                        help="default: the full 9-county BAY_BBOX_4326")
    parser.add_argument("--benchmark-path", default=None,
                        help="default: the full 9-county mosaic - pass a county-clipped "
                             "CoSMoS raster (a zip:// path works) for a cheap sample")
    parser.add_argument("--benchmark-scale-m", type=float, default=1.0,
                        help="multiply --benchmark-path's raw values by this to get metres - "
                             "0.01 for a USGS CoSMoS per-county extract (int32 centimetres); "
                             "default 1.0 is correct for the full 9-county mosaic (float32 metres)")
    parser.add_argument("--native-merged-path", default=None,
                        help="read the model depth from one native-CRS merged file instead "
                             "of the standard EPSG:4326 merged chunks - see module docstring")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    kwargs = {"out_dir": out_dir}
    if args.bbox is not None:
        kwargs["bbox"] = tuple(args.bbox)
    if args.benchmark_path is not None:
        kwargs["benchmark_path"] = args.benchmark_path
    kwargs["benchmark_scale_m"] = args.benchmark_scale_m
    if args.native_merged_path is not None:
        kwargs["native_merged_path"] = args.native_merged_path
    results = validate_sf_bay(args.config, **kwargs)

    import pandas as pd
    df = pd.DataFrame(results).T
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(df.to_string())

    if out_dir is not None:
        metrics_path = out_dir / f"metrics_{OUTPUT_TAG}.csv"
        df.to_csv(metrics_path)
        print(f"\nwrote {metrics_path}")

    if args.out:
        df.to_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
