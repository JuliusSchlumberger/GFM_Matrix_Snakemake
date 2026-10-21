"""Compute per-delta RP100 flooded area (km2, % of modelled outline) per SSP and SLR.

Flood hazard only: no protection standards are applied, so a cell counts as flooded
wherever the merged RP100 depth is > 0 inside the delta polygon. Tiles are merged in
memory by per-cell maximum on the delta's bbox grid (as merge.py does for chunks), and
the percentage is relative to the tile-covered part of the polygon
(delta_area_km2_modeled), not the shapefile Area, so partial tile coverage does not
read as low flooding (see pct_delta_covered_by_model).

Usage:
    python compute_delta_flood_extent.py
    python compute_delta_flood_extent.py --scenarios ssp126
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402
from merge import EIKONAL_NODATA, _make_chunk_transform, _open_overlapping_tiles, _tile_id_from_path  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from rasters import decode_waterdepth_cm, polygon_domain_mask  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RETURN_PERIOD = "RP100"
SLR_SCENARIOS = ["SLR_0", "SLR_500", "SLR_1000", "SLR_1500", "SLR_2000"]
SCENARIOS_DEFAULT = ["ssp126", "ssp245"]
VOLGA_ID = 44  # zero tile coverage in both scenarios

# Best-effort names matched by hand to centroids; a label, never a join key (delta_id is authoritative).
DELTA_NAME_INFERRED = {
    1: "Amazon", 2: "Amur", 3: "Fitzroy (Australia)", 4: "Colorado (Mexico/US)",
    5: "Congo", 6: "Chao Phraya", 7: "Danube", 8: "Dnieper", 9: "Ebro",
    10: "Fly", 11: "Ganges-Brahmaputra", 12: "Godavari", 13: "Grijalva-Usumacinta",
    14: "Han", 15: "Indus", 16: "Irrawaddy", 17: "Krishna", 18: "Lena",
    19: "Limpopo", 20: "Mackenzie", 21: "Magdalena", 22: "Mahakam", 23: "Mahanadi",
    24: "Mekong", 25: "Mississippi", 26: "Moulouya", 27: "Niger", 28: "Nile",
    29: "Orinoco", 30: "Parana-Rio de la Plata", 31: "Pearl", 32: "Po",
    33: "Red River", 34: "Rio Grande", 35: "Rhine-Meuse", 36: "Rhone",
    37: "Sebou", 38: "Senegal", 39: "Sao Francisco", 40: "Tana", 41: "Shatt al-Arab",
    42: "Tone", 43: "Vistula", 44: "Volga", 45: "Volta", 46: "Yangtze",
    47: "Yellow River", 48: "Yukon-Kuskokwim", 49: "Zambezi",
}


def build_tile_delta_join(tile_grid: gpd.GeoDataFrame, deltas: gpd.GeoDataFrame) -> dict[int, list[int]]:
    """Map delta Id -> tile ids intersecting its REAL polygon (not the selection bbox buffer)."""
    joined = gpd.sjoin(tile_grid[["tile_id", "geometry"]], deltas[["Id", "geometry"]],
                        how="inner", predicate="intersects")
    return joined.groupby("Id")["tile_id"].apply(list).to_dict()


def _tile_geometry(tile_rasters: list[Path], bbox: tuple[float, float, float, float]):
    # Reused for every SLR: all tiles share origin and pixel size. Tiles not reaching bbox are dropped.
    with rasterio.open(tile_rasters[0]) as ref:
        ref_transform = ref.transform
    out_transform, out_w, out_h = _make_chunk_transform(bbox, ref_transform)
    metas = _open_overlapping_tiles(tile_rasters, out_transform, out_w, out_h)
    geometry = [
        (_tile_id_from_path(tm.path), tm.row_off, tm.col_off, tm.n_rows, tm.n_cols, tm.src_r0, tm.src_c0)
        for tm in metas
    ]
    for tm in metas:
        tm.src.close()
    return out_transform, out_w, out_h, geometry


def merge_in_memory(model_outputs: Path, slr: str, out_w: int, out_h: int,
                    geometry: list[tuple]) -> np.ndarray | None:
    """Merge overlapping tile depths by per-cell MAXIMUM, in memory.

    Same rule as merge.merge_tile_rasters_chunk (tiles overlap; summing double
    counts). Safe in memory only because a delta bbox is small (Amazon ~287M px
    at 30 m). int16-cm tiles are decoded, so their 'not computed' value is nodata, not
    flooded. NaN where no tile reaches a cell; None if no tile has this SLR.
    """
    best = np.full((out_h, out_w), -np.inf, dtype="float64")
    covered = np.zeros((out_h, out_w), dtype=bool)
    any_found = False
    for tile_id, row_off, col_off, n_rows, n_cols, src_r0, src_c0 in geometry:
        path = model_outputs / str(tile_id) / "results" / f"waterdepth_{RETURN_PERIOD}_{slr}.tif"
        if not path.exists():
            print(f"    WARNING: {path} missing (had SLR_0 but not {slr}) - excluded from this merge", flush=True)
            continue
        any_found = True
        with rasterio.open(path) as src:
            raw = src.read(1, window=Window(src_c0, src_r0, n_cols, n_rows))
        patch = (decode_waterdepth_cm(raw) if raw.dtype == np.int16 else raw).astype("float64")
        valid = patch < EIKONAL_NODATA
        sl = (slice(row_off, row_off + n_rows), slice(col_off, col_off + n_cols))
        sub_best = best[sl]
        wins = valid & (patch > sub_best)
        sub_best[wins] = patch[wins]
        best[sl] = sub_best
        covered[sl] |= valid
    if not any_found:
        return None
    return np.where(covered, best, np.nan).astype("float32")


def score_delta(delta_row, candidate_tiles: list[int], model_outputs: Path) -> dict:
    """Score one delta's flooded area per SLR on its bbox merge grid.

    Flooded = merged depth > 0 inside the tile-covered delta polygon. n_tiles
    and the modelled domain come from SLR_0, so every SLR is scored on the
    same cells.
    """
    delta_id = int(delta_row["Id"])
    delta_gdf = gpd.GeoDataFrame({"Id": [delta_id]}, geometry=[delta_row.geometry], crs="EPSG:4326")
    bbox = tuple(delta_row.geometry.bounds)

    slr0_rasters = [
        model_outputs / str(tid) / "results" / f"waterdepth_{RETURN_PERIOD}_SLR_0.tif"
        for tid in candidate_tiles
    ]
    slr0_rasters = [p for p in slr0_rasters if p.exists()]
    if not slr0_rasters:
        return {"n_tiles": 0, "delta_area_km2_modeled": 0.0,
                "delta_area_km2_shapefile": float(delta_row["Area"]),
                "pct_delta_covered_by_model": 0.0, "flooded_km2_by_slr": {slr: 0.0 for slr in SLR_SCENARIOS}}

    out_transform, out_w, out_h, geometry = _tile_geometry(slr0_rasters, bbox)

    per_slr_flooded_km2: dict[str, float] = {}
    n_tiles = len(geometry)
    modeled_km2 = 0.0
    domain = None
    area_grid = None
    for slr in SLR_SCENARIOS:
        depth = merge_in_memory(model_outputs, slr, out_w, out_h, geometry)
        if depth is None:
            per_slr_flooded_km2[slr] = 0.0
            continue

        if domain is None:
            # Not model_domain_mask: nodata here is NaN and NaN != NaN is always True.
            covered = ~np.isnan(depth)
            domain = polygon_domain_mask(delta_gdf, out_transform, "EPSG:4326", (out_h, out_w)) & covered
            area_grid = pixel_area_km2_grid(out_transform, out_w, out_h)
            modeled_km2 = float((domain * area_grid).sum())

        wet = (depth > 0) & domain  # any depth (not the 0.1 m exposure threshold); False on NaN
        per_slr_flooded_km2[slr] = float((wet * area_grid).sum())

    shapefile_km2 = float(delta_row["Area"])
    pct_covered = 100.0 * modeled_km2 / shapefile_km2 if shapefile_km2 > 0 else float("nan")

    return {
        "n_tiles": n_tiles,
        "delta_area_km2_modeled": modeled_km2,
        "delta_area_km2_shapefile": shapefile_km2,
        "pct_delta_covered_by_model": pct_covered,
        "flooded_km2_by_slr": per_slr_flooded_km2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=str(_REPO_ROOT / "snakemake_workflow" / "config"))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--delta-ids", nargs="+", type=int, default=None, help="only these delta Ids (testing)")
    args = parser.parse_args()

    prod_cfg = load_config(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")

    deltas = retry_transient_io(gpd.read_file, Path(prod_cfg["paths"]["deltas_shapefile"]))
    deltas = deltas[deltas["Id"] != VOLGA_ID].reset_index(drop=True)
    if args.delta_ids:
        deltas = deltas[deltas["Id"].isin(args.delta_ids)].reset_index(drop=True)
    centroids = deltas.geometry.centroid
    deltas["centroid_lon"] = centroids.x
    deltas["centroid_lat"] = centroids.y
    print(f"{len(deltas)} delta(s) (Id={VOLGA_ID} Volga excluded - zero tile coverage)")

    first_cfg = load_config(Path(args.config_dir) / f"deltas_{args.scenarios[0]}_materialized.yml")
    tile_grid = retry_transient_io(gpd.read_file, Path(first_cfg["tile_grid"]["path"]))
    if tile_grid.crs != deltas.crs:
        tile_grid = tile_grid.to_crs(deltas.crs)
    tiles_by_delta = build_tile_delta_join(tile_grid, deltas)
    n_unmatched = sum(1 for _, row in deltas.iterrows() if int(row["Id"]) not in tiles_by_delta)
    if n_unmatched:
        print(f"WARNING: {n_unmatched} delta(s) matched zero tiles in the shared tile grid")

    root = Path(first_cfg["paths"]["root"])
    out_path = Path(args.out) if args.out else root / "deltas_floodmaps" / "delta_flood_extent.csv"

    rows = []
    t_start = time.time()
    n_deltas = len(deltas)
    for scenario in args.scenarios:
        cfg = load_config(Path(args.config_dir) / f"deltas_{scenario}_materialized.yml")
        model_outputs = Path(cfg["simulation"]["model_outputs"])
        print(f"\n=== {scenario} (model_outputs={model_outputs}) ===", flush=True)

        for i, (_, delta_row) in enumerate(deltas.iterrows(), start=1):
            delta_id = int(delta_row["Id"])
            delta_name = DELTA_NAME_INFERRED.get(delta_id, f"delta_{delta_id}")
            candidate_tiles = tiles_by_delta.get(delta_id, [])
            print(f"  [{i}/{n_deltas}] delta {delta_id:2d} ({delta_name}): "
                  f"{len(candidate_tiles)} candidate tile(s) - merging...", flush=True)
            t0 = time.time()
            result = score_delta(delta_row, candidate_tiles, model_outputs)
            elapsed = time.time() - t0
            modeled_km2 = result["delta_area_km2_modeled"]

            print(f"    -> {result['n_tiles']} tile(s), "
                  f"modeled {modeled_km2:.1f}/{result['delta_area_km2_shapefile']:.1f} km2 "
                  f"({result['pct_delta_covered_by_model']:.1f}% covered) - {elapsed:.1f}s "
                  f"[{(time.time() - t_start) / 60:.1f}min elapsed]", flush=True)

            for slr in SLR_SCENARIOS:
                flooded_km2 = result["flooded_km2_by_slr"][slr]
                # vs the tile-covered outline, not the shapefile Area (see pct_delta_covered_by_model)
                pct_flooded = 100.0 * flooded_km2 / modeled_km2 if modeled_km2 > 0 else float("nan")
                rows.append({
                    "delta_id": delta_id,
                    "delta_name_inferred": delta_name,
                    "centroid_lon": round(float(delta_row["centroid_lon"]), 4),
                    "centroid_lat": round(float(delta_row["centroid_lat"]), 4),
                    "ssp_scenario": scenario,
                    "waterlevel_name": slr,
                    "return_period": RETURN_PERIOD,
                    "n_tiles": result["n_tiles"],
                    "flooded_area_km2": round(flooded_km2, 4),
                    "delta_area_km2_modeled": round(modeled_km2, 4),
                    "delta_area_km2_shapefile": round(result["delta_area_km2_shapefile"], 4),
                    "pct_delta_covered_by_model": round(result["pct_delta_covered_by_model"], 2),
                    "pct_flooded_of_modeled_area": round(pct_flooded, 4),
                })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retry_transient_io(df.to_csv, out_path, index=False)
    print(f"\nWrote {len(df)} row(s) to {out_path}")

    low_coverage = df[["delta_id", "delta_name_inferred", "ssp_scenario", "pct_delta_covered_by_model"]].drop_duplicates()
    low_coverage = low_coverage[low_coverage["pct_delta_covered_by_model"] < 90]
    if not low_coverage.empty:
        print(f"\nWARNING: {len(low_coverage)} (delta, scenario) pair(s) below 90% modeled coverage:")
        print(low_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
