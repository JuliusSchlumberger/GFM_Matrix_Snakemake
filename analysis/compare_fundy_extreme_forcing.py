"""One-off experiment: does forcing with a real observed extreme water level
(instead of COAST-RP's own RP100 value) close the severe under-prediction
already found for New Brunswick's Bay of Fundy benchmark
(`new_brunswick_present_day_1in100yr` in data_catalog_validation.yml,
CSI ~ 0.002 for both regions, attributed to COAST-RP's RP100 capping at an
"implausibly low" ~4.4-4.77m near the Fundy head)?

Compares CSI against the same NB government benchmark for three sources, per
region (`cumberland_basin`, `petitcodiac` - the benchmark's own two bboxes):
  (A) production COAST-RP RP100/SLR_0 forcing, unmodified.
  (C) every boundary station's water level replaced with a single flat value
      - the real observed maximum from the nearest relevant GESLA tide-gauge
      station (Joggins for Cumberland Basin, Moncton for Petitcodiac).
  (D) same flat-value substitution, but using a detided/scaled extreme
      water level instead of the plain historical max: harmonic tidal fit
      + real multi-decade annual-maxima GEV surge extreme from the two
      nearest LONG-record stations (Saint John, Eastport - 100/88 usable
      years respectively), scaled to the target site via the M2
      tidal-amplitude ratio against Joggins/Moncton, then added to each
      target site's own harmonic-only extreme-tide estimate. Using the
      HIGHER (Saint-John-based) of the two reference-station estimates, per
      direct user confirmation - see FORCING_SOURCES below for the real
      numbers and conversation history for the full derivation (not
      reproduced in code - this is a one-off experiment script, not a
      reusable detiding/EVA pipeline).

Both tiles covering this area (571 north, 891 south - the tile-grid seam
runs through both bboxes) are solved directly via flood_model.flood_depth_dense,
reading inputs straight from model_outputs/{tile}/inputs/ (read-only - this
never touches production's own model_outputs/{tile}/results/ tree, same
bypass pattern calibration_studies/test_*_calibration.py already use).
Neither tile has a production RP100 result yet, so (A) is a fresh run too -
this keeps A and C perfectly comparable (identical code path/config, only
the boundary input differs).

CSI itself reuses the SAME confusion-count machinery
validate_country_national_coverage() uses in production
(validation.confusion_counts_soft/metrics_from_counts/benchmark_fraction_from_
vector/read_permanent_water_mask/read_country_mask) - not reimplemented here.

Usage:
    python compare_fundy_extreme_forcing.py
    python compare_fundy_extreme_forcing.py --config ../snakemake_workflow/config/config.yml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402
from plotting import pixel_area_km2_grid  # noqa: E402
from rasters import decode_dem_cm, decode_friction_int16, decode_waterlevel_cm  # noqa: E402
import validation as v  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"
OCEAN_CODE = 1
RIVER_CODE = 3
BENCHMARK_KEY = "new_brunswick_present_day_1in100yr"
COUNTRY_ISO = "CAN"

# Both tiles whose bbox overlaps either study region (confirmed directly
# against domain_tiles_global.gpkg - the tile-grid seam runs through both
# bboxes, so a faithful score needs both tiles summed together, not either
# alone) - both hop_distance==0, both already preprocessed (dem/mask/friction/
# boundaries on disk), neither has a results/ dir yet.
TILES = [571, 891]

# The two benchmark regions are exactly data_catalog_validation.yml's own
# meta.regions for BENCHMARK_KEY - duplicated here as plain bboxes so this
# script doesn't need a full BenchmarkSpec just to read two numbers back out.
BBOXES = {
    "cumberland_basin": [-64.42, 45.85, -64.22, 46.05],
    "petitcodiac": [-64.83, 45.92, -64.42, 46.13],
}

# Real observed maxima, pulled directly from GESLA4.1 station files (NULL
# value -99.9999 excluded) - the nearest geographically-relevant station per
# region (both short, century-old records):
#   Joggins  (45.68N -64.47W, near Cumberland Basin): 744 valid hourly
#            readings, 1919 only, max = 7.43 m.
#   Moncton  (46.08N -64.77W, at the head of the Petitcodiac estuary): 719
#            valid hourly readings, 1898-1920 (only 15 usable continuous
#            days), max = 8.65 m.
GESLA_MAX_BY_BBOX = {
    "cumberland_basin": 7.43,
    "petitcodiac": 8.65,
}

# Detided + scaled extreme water level (see module docstring, source D):
# harmonic-only extreme tide (Joggins/Moncton 5-constituent fit, predicted
# over a synthetic 19-yr window) + 100-yr surge return level from Saint
# John's real 100-year GEV fit (shape=-0.236, loc=1.125, scale=0.228 on 100
# usable annual maxima), scaled by the M2 amplitude ratio to the target site
# (Joggins M2=3.441m / Saint John M2=3.039m = 1.132; Moncton M2=3.155m /
# Saint John M2=3.039m = 1.038). Using the HIGHER of the two reference
# stations tried (Saint John vs Eastport) per direct user confirmation.
GESLA_DETIDED_EVA_BY_BBOX = {
    "cumberland_basin": 10.22,
    "petitcodiac": 10.66,
}

# {source_key: {bbox_name: forcing_m}} - every entry here gets its own solve
# per tile, scored against the benchmark exactly like source A.
OVERRIDE_SOURCES = {
    "C_gesla_max": GESLA_MAX_BY_BBOX,
    "D_detided_eva": GESLA_DETIDED_EVA_BY_BBOX,
}


def run_tile(tile_id: int, model_outputs: Path, flood_cfg: dict, override_wl: float | None) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Solves flood_depth_dense for one tile, reading inputs directly from
    model_outputs/{tile_id}/inputs/ - never writes there. `override_wl`, if
    given (plain metres), replaces every boundary station's own water level
    with that single flat value before solving; `None` runs production's
    real, unmodified boundaries as-is.
    """
    inputs_dir = model_outputs / str(tile_id) / "inputs"
    with rasterio.open(inputs_dir / "dem.tif") as src:
        dem = decode_dem_cm(src.read(1))
        transform = src.transform
    with rasterio.open(inputs_dir / "mask.tif") as src:
        mask = src.read(1).astype(np.int8)
    with rasterio.open(inputs_dir / "friction.tif") as src:
        friction = decode_friction_int16(src.read(1)) * np.float32(flood_cfg["friction_scale_factor"])
        friction = np.where(friction > 0, friction, friction.dtype.type(0.001))

    boundaries = gpd.read_file(inputs_dir / f"boundaries_{RETURN_PERIOD}_{WATERLEVEL_NAME}.gpkg")
    boundaries[WATERLEVEL_NAME] = decode_waterlevel_cm(boundaries[WATERLEVEL_NAME].to_numpy())
    if override_wl is not None:
        boundaries[WATERLEVEL_NAME] = override_wl

    waterdepth, diagnostics = flood_depth_dense(
        dem, mask, friction, transform,
        boundaries=boundaries, resolution=float(flood_cfg["resolution"]), k=int(flood_cfg["knn"]),
        variable=WATERLEVEL_NAME, ocean_code=OCEAN_CODE, river_code=RIVER_CODE,
        obstacle_coupling=bool(flood_cfg["obstacle_coupling"]["enabled"]),
        max_outer_iterations=int(flood_cfg["obstacle_coupling"]["max_outer_iterations"]),
        max_rounds=int(flood_cfg["max_rounds"]),
        outer_convergence_pct=float(flood_cfg["obstacle_coupling"]["outer_convergence_pct"]),
        waterlevel_epsilon_m=float(flood_cfg["waterlevel_epsilon_m"]),
    )
    tag = "A (COAST-RP RP100)" if override_wl is None else f"override={override_wl:.2f}m"
    print(f"  tile {tile_id} [{tag}]: {int((waterdepth > 0).sum()):,} wet cell(s), "
          f"max depth {float(waterdepth.max()):.3f} m "
          f"({'obstacle_coupling outer=' + str(diagnostics.get('outer_iterations_used')) if diagnostics.get('obstacle_coupling') else 'no obstacle coupling'})")
    return waterdepth, transform


def write_raster(path: Path, arr: np.ndarray, transform, crs="EPSG:4326") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="float32", crs=crs, transform=transform, nodata=-9999.0,
        compress="deflate",
    ) as dst:
        dst.write(arr.astype("float32"), 1)


def score_region(
    region_bbox: list[float], tile_paths: list[Path], gfm_catalog, cfg: dict,
    iso_lookup: dict, geogunit_ids: list[int], bench_gdf: gpd.GeoDataFrame,
) -> tuple[float, float, float, float]:
    """Sums (tp, fp, fn, tn), area-weighted km2, across every tile that
    overlaps region_bbox - additive by construction, so this is exactly how
    the two tiles sharing the seam combine into one region-level score."""
    val_cfg = cfg["validation"]
    threshold = float(val_cfg["primary_threshold_m"])
    supersample = int(val_cfg["benchmark_supersample"])
    permanent_water_source = val_cfg["permanent_water_source"]
    permanent_water_codes = val_cfg["permanent_water_codes"]
    geogunit_source = val_cfg["geogunit_source"]

    tp_sum = fp_sum = fn_sum = tn_sum = 0.0
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as src:
            window = rasterio.windows.from_bounds(*region_bbox, transform=src.transform)
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            window = window.round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                continue
            depth = src.read(1, window=window)
            out_transform = src.window_transform(window)

        bbox = list(rasterio.transform.array_bounds(depth.shape[0], depth.shape[1], out_transform))
        candidate = bench_gdf.iloc[list(bench_gdf.sindex.query(box(*bbox), predicate="intersects"))]
        fraction = v.benchmark_fraction_from_vector(candidate, out_transform, "EPSG:4326", depth.shape, supersample=supersample)

        water_mask = v.read_permanent_water_mask(
            gfm_catalog, permanent_water_source, permanent_water_codes, bbox, out_transform, depth.shape,
        )
        in_country = v.read_country_mask(
            gfm_catalog, geogunit_source, iso_lookup, COUNTRY_ISO, bbox, out_transform, depth.shape,
            geogunit_ids=geogunit_ids,
        )
        not_water = ~water_mask & in_country
        domain_mask = not_water  # whole tile is real computed data - no model_domain_mask nodata gap here

        model_wet = (depth > threshold) & not_water
        area_km2 = pixel_area_km2_grid(out_transform, depth.shape[1], depth.shape[0])

        tp, fp, fn, tn = v.confusion_counts_soft(model_wet, fraction, domain_mask, area_km2)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

    return tp_sum, fp_sum, fn_sum, tn_sum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_outputs = Path(cfg["simulation"]["model_outputs"])
    flood_cfg = cfg["simulation"]["flooding"]
    scratch_dir = Path(cfg["paths"]["root"]) / "fundy_extreme_forcing_experiment"

    print(f"=== Running tiles {TILES} for source A (COAST-RP RP100) and overrides {list(OVERRIDE_SOURCES)} ===")
    tile_paths: dict[str, dict[int, Path]] = {"A": {}}
    for tile_id in TILES:
        waterdepth_a, transform = run_tile(tile_id, model_outputs, flood_cfg, override_wl=None)
        path_a = scratch_dir / str(tile_id) / "waterdepth_A.tif"
        write_raster(path_a, waterdepth_a, transform)
        tile_paths["A"][tile_id] = path_a
        depth_a_wet = int((waterdepth_a > 0).sum())

        # Each bbox within an override source gets its own forcing value -
        # both bboxes get scored against both tiles, so each tile is solved
        # once per DISTINCT (source, bbox) override value actually needed.
        for source_key, wl_by_bbox in OVERRIDE_SOURCES.items():
            for bbox_name, wl in wl_by_bbox.items():
                waterdepth_o, transform = run_tile(tile_id, model_outputs, flood_cfg, override_wl=wl)
                path_o = scratch_dir / str(tile_id) / f"waterdepth_{source_key}_{bbox_name}.tif"
                write_raster(path_o, waterdepth_o, transform)
                assert depth_a_wet <= int((waterdepth_o > 0).sum()), (
                    f"tile {tile_id}: {source_key} forcing ({wl}m) produced FEWER wet cells "
                    f"({int((waterdepth_o > 0).sum())}) than COAST-RP RP100 ({depth_a_wet}) - "
                    f"a strictly higher, spatially-flat forcing should never flood less; investigate before trusting results."
                )
                tile_paths.setdefault(f"{source_key}_{bbox_name}", {})[tile_id] = path_o

    print(f"\n=== Loading NB benchmark + country/geogunit lookups ===")
    gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    val_cfg = cfg["validation"]
    bench_catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    v.fix_catalog_meta_encoding(bench_catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])
    spec = v.load_benchmark_spec(bench_catalog, BENCHMARK_KEY)
    full_gdf = v.load_benchmark_full(bench_catalog, spec)
    if full_gdf.crs is None or full_gdf.crs.to_epsg() != 4326:
        full_gdf = full_gdf.to_crs(4326)
    full_gdf = full_gdf.set_geometry("geometry")
    full_gdf.sindex  # build once, reused per region below
    iso_lookup = v.load_iso_lookup(gfm_catalog, val_cfg["iso_lookup_source"])

    print(f"\n=== Scoring ===")
    rows = []
    for bbox_name, region_bbox in BBOXES.items():
        # Clip the benchmark to this region's own bbox (+ small margin) first -
        # data_catalog_validation.yml's own comment explains why: the
        # whole-province region was abandoned after a 3+ hour rasterize() hang,
        # these two small regions pull only a few thousand candidate features
        # and score in seconds.
        clip_box = box(region_bbox[0] - 0.05, region_bbox[1] - 0.05, region_bbox[2] + 0.05, region_bbox[3] + 0.05)
        region_gdf = full_gdf.iloc[list(full_gdf.sindex.query(clip_box, predicate="intersects"))]

        sources_to_score = [("A_coastrp_rp100", tile_paths["A"], None)]
        for source_key, wl_by_bbox in OVERRIDE_SOURCES.items():
            sources_to_score.append((source_key, tile_paths[f"{source_key}_{bbox_name}"], wl_by_bbox[bbox_name]))

        for source_label, paths_by_tile, forcing_m in sources_to_score:
            tp, fp, fn, tn = score_region(
                region_bbox, list(paths_by_tile.values()), gfm_catalog, cfg,
                iso_lookup, spec.geogunit_ids, region_gdf,
            )
            metrics = v.metrics_from_counts(tp, fp, fn, tn)
            rows.append({
                "region": bbox_name, "source": source_label, "forcing_m": forcing_m,
                "tp_km2": round(tp, 4), "fp_km2": round(fp, 4), "fn_km2": round(fn, 4), "tn_km2": round(tn, 4),
                "HR": metrics["HR"], "FAR": metrics["FAR"], "CSI": metrics["CSI"], "bias": metrics["bias"],
            })

    df = pd.DataFrame(rows)
    out_csv = scratch_dir / "csi_comparison.csv"
    retry_transient_io(df.to_csv, out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
