"""Compute per-delta RP100 flooded area (km2, % of modelled outline) per SSP and SLR.

Flooding is FLOPROS-protection-adjusted; flooded_area_km2_unprotected is the raw value.

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
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from merge import EIKONAL_NODATA, _make_chunk_transform, _open_overlapping_tiles, _tile_id_from_path  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from rasters import polygon_domain_mask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sfincs_tiles"))
from build_station_boundary_lines import MASK_VRT, OCEAN_CODE  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
RETURN_PERIOD = "RP100"
SLR_SCENARIOS = ["SLR_0", "SLR_500", "SLR_1000", "SLR_1500", "SLR_2000"]
SCENARIOS_DEFAULT = ["ssp126", "ssp245"]
VOLGA_ID = 44  # zero tile coverage in both scenarios
# Rhine-Meuse: Dutch standards (1/1,000-1/30,000 yr) are capped at COAST-RP's RP1000, whose tail is far
# too flat here (RP1000-RP100 ~0.2 m vs ~1.1-1.3 m between the official 1/100 and 1/10,000 levels at Hoek
# van Holland), so the failure rule would fail the dikes at the first SLR step.
NEVER_FAILING_DELTA_IDS = {35}
PROTECTION_THRESHOLD_YEARS = 100.0  # areas protected to at least this design RP do not flood
CONNECT_PAD_DEG = (0.25, 0.5, 1.0)  # sea-connectivity grid = merge grid + this on every side (first with ocean)
SIMULATED_RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)  # COAST-RP return periods a design standard snaps to


def snap_rp(rp):
    """Floor a design standard (yr) to a simulated RP (above the largest -> the largest).

    Floored, not rounded up as in compute_exposure_analysis.py: delta FLOPROS
    standards sit just above a COAST-RP step (101, 112, 134, 150, 156, 500, 527 yr).
    """
    steps = np.asarray(SIMULATED_RPS)
    out = steps[np.clip(np.searchsorted(steps, rp, side="right") - 1, 0, len(steps) - 1)]
    return int(out) if np.ndim(out) == 0 else out.astype("int32")
GEOGUNIT_SOURCE = "geogunit_protection_units"

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


def merge_in_memory(model_outputs: Path, slr: str, out_w: int, out_h: int, geometry: list[tuple],
                    rp: str = RETURN_PERIOD) -> np.ndarray | None:
    """Merge overlapping tile depths by per-cell MAXIMUM, in memory.

    Same rule as merge.merge_tile_rasters_chunk (tiles overlap; summing double
    counts). Safe in memory only because a delta bbox is small (Amazon ~287M px
    at 30 m). NaN where no tile reaches a cell; None if no tile has this SLR.
    """
    best = np.full((out_h, out_w), -np.inf, dtype="float64")
    covered = np.zeros((out_h, out_w), dtype=bool)
    any_found = False
    for tile_id, row_off, col_off, n_rows, n_cols, src_r0, src_c0 in geometry:
        path = model_outputs / str(tile_id) / "results" / f"waterdepth_{rp}_{slr}.tif"
        if not path.exists():
            print(f"    WARNING: {path} missing (had SLR_0 but not {slr}) - excluded from this merge", flush=True)
            continue
        any_found = True
        with rasterio.open(path) as src:
            patch = src.read(1, window=Window(src_c0, src_r0, n_cols, n_rows)).astype("float64")
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


def _read_mask_on_grid(root: Path, transform, width: int, height: int) -> np.ndarray:
    # Mask codes: 0 land, 1 ocean, 2 lake, 3 river. The VRT has no tiles over open ocean or far inland:
    # a no-data block whose valid rim is mostly ocean is ocean, else land (as build_station_boundary_lines).
    with rasterio.open(root / MASK_VRT) as src:
        nodata = src.nodata if src.nodata is not None else 255
        arr = np.full((height, width), nodata, dtype="uint8")
        reproject(rasterio.band(src, 1), arr, dst_transform=transform, dst_crs="EPSG:4326",
                  resampling=Resampling.nearest, src_nodata=nodata, dst_nodata=nodata)
    missing = arr == nodata
    if missing.any():
        labels, n = ndimage.label(missing)
        rim = ndimage.binary_dilation(missing) & ~missing
        lab_rim = ndimage.grey_dilation(labels, size=3) * rim
        n_rim = ndimage.sum(rim, lab_rim, index=np.arange(1, n + 1))
        n_ocean = ndimage.sum(rim & (arr == OCEAN_CODE), lab_rim, index=np.arange(1, n + 1))
        ocean_blocks = np.arange(1, n + 1)[(n_rim > 0) & (n_ocean > 0.5 * n_rim)]
        arr[missing] = 0
        arr[np.isin(labels, ocean_blocks)] = OCEAN_CODE
    return arr


def _protection_rp(catalog, rp_by_geogunit: pd.Series, default_rp: float, transform, width: int, height: int):
    # Cells without a geogunit unit get default_rp.
    left, top = transform.c, transform.f
    bbox = [left, top + height * transform.e, left + width * transform.a, top]
    try:
        geo = catalog.get_rasterdataset(GEOGUNIT_SOURCE, bbox=bbox, variables=["Geogunits"]).squeeze(drop=True)
    except Exception as e:  # noqa: BLE001 - no geogunit coverage: production's default standard everywhere
        print(f"    (no geogunit coverage: {e}) - default protection {default_rp:g} yr everywhere", flush=True)
        return np.full((height, width), default_rp, dtype="float32")
    gid = np.full((height, width), -1, dtype="int32")
    reproject(np.nan_to_num(geo.values.astype("float64"), nan=-1).astype("int32"), gid,
              src_transform=geo.raster.transform, src_crs=geo.raster.crs, dst_transform=transform,
              dst_crs="EPSG:4326", resampling=Resampling.nearest, src_nodata=-1, dst_nodata=-1)
    ids, inv = np.unique(gid, return_inverse=True)
    mapped = pd.Series(ids).map(rp_by_geogunit).to_numpy(dtype="float64")
    return np.where(np.isnan(mapped), default_rp, mapped)[inv].reshape(height, width).astype("float32")


class ProtectionGrid:
    """Hold one delta's FLOPROS protection (Coastal, else Riverine, else default_rp) on the merge grid.

    Padded by CONNECT_PAD_DEG until it holds ocean (Amur's bbox has none), cropped back after.
    floodable(failed): cells below PROTECTION_THRESHOLD_YEARS or in a failed class (floored design
    RP), 8-connected to DeltaDTM open ocean through such cells (land and water alike), ~30 m grid.
    """

    def __init__(self, prot_inputs: dict, domain: np.ndarray, transform, width: int, height: int):
        catalog, rp_by_geogunit = prot_inputs["catalog"], prot_inputs["rp_by_geogunit"]
        default_rp, root = prot_inputs["default_rp"], prot_inputs["root"]
        for pad_deg in CONNECT_PAD_DEG:
            pad = int(round(pad_deg / abs(transform.a)))
            tr = transform * Affine.translation(-pad, -pad)
            H, W = height + 2 * pad, width + 2 * pad
            ocean = _read_mask_on_grid(root, tr, W, H) == OCEAN_CODE
            if ocean.any():
                break
        rp = _protection_rp(catalog, rp_by_geogunit, default_rp, tr, W, H)
        self.ocean = ocean
        self.rp_class = np.where(rp >= PROTECTION_THRESHOLD_YEARS,
                                 snap_rp(rp), 0).astype("int32")
        self.crop = (slice(pad, pad + height), slice(pad, pad + width))
        self._cache: dict[frozenset, np.ndarray] = {}
        rp_c = rp[self.crop]
        self.land = domain & ~ocean[self.crop]
        vals, counts = np.unique(rp_c[self.land], return_counts=True)
        self.main_rp = float(vals[np.argmax(counts)]) if len(vals) else float("nan")
        cls_c = self.rp_class[self.crop]
        self.classes = sorted(int(c) for c in np.unique(cls_c[self.land]) if c > 0)
        self.class_masks = {c: self.land & (cls_c == c) for c in self.classes}
        self.n_land = max(int(self.land.sum()), 1)
        self.pct_protected = 100.0 * float((self.land & (cls_c > 0)).sum()) / self.n_land

    def floodable(self, failed: frozenset) -> np.ndarray:
        if failed not in self._cache:
            low = (self.rp_class == 0) | np.isin(self.rp_class, list(failed))
            labels, _ = ndimage.label(low | self.ocean, structure=np.ones((3, 3)))
            sea_ids = np.unique(labels[self.ocean])
            self._cache[failed] = np.isin(labels[self.crop], sea_ids[sea_ids > 0]) & low[self.crop]
        return self._cache[failed]


def score_delta(delta_row, candidate_tiles: list[int], model_outputs: Path, prot_inputs: dict | None = None,
                design_outputs: Path | None = None) -> dict:
    """Score one delta's flooded area per SLR on its bbox merge grid.

    A protection class fails once the RP100 flood covers more of its area than
    its design event did at SLR_0 (design floods > RP100 from design_outputs);
    failed cells count as unprotected. n_tiles/modeled area come from SLR_0.
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
        zero = {slr: 0.0 for slr in SLR_SCENARIOS}
        return {"n_tiles": 0, "delta_area_km2_modeled": 0.0,
                "delta_area_km2_shapefile": float(delta_row["Area"]),
                "pct_delta_covered_by_model": 0.0, "flooded_km2_by_slr": zero, "raw_km2_by_slr": dict(zero),
                "protection": None, "per_slr_protection": {}}

    out_transform, out_w, out_h, geometry = _tile_geometry(slr0_rasters, bbox)

    per_slr_flooded_km2: dict[str, float] = {}
    per_slr_raw_km2: dict[str, float] = {}
    n_tiles = len(geometry)
    prot = None
    design_km2: dict[int, float] = {}  # class -> its area flooded by the design event at SLR_0
    per_slr_prot: dict[str, dict] = {}
    modeled_km2 = 0.0
    domain = None
    area_grid = None
    for slr in SLR_SCENARIOS:
        depth = merge_in_memory(model_outputs, slr, out_w, out_h, geometry)
        if depth is None:
            per_slr_flooded_km2[slr] = per_slr_raw_km2[slr] = 0.0
            continue

        if domain is None:
            # Not model_domain_mask: nodata here is NaN and NaN != NaN is always True.
            covered = ~np.isnan(depth)
            domain = polygon_domain_mask(delta_gdf, out_transform, "EPSG:4326", (out_h, out_w)) & covered
            area_grid = pixel_area_km2_grid(out_transform, out_w, out_h)
            modeled_km2 = float((domain * area_grid).sum())
            if prot_inputs is not None:
                prot = ProtectionGrid(prot_inputs, domain, out_transform, out_w, out_h)
                never_fails = int(delta_row["Id"]) in NEVER_FAILING_DELTA_IDS
                for c in prot.classes:
                    if never_fails:
                        design_km2[c] = float("inf")
                        continue
                    if c == 100:
                        d_wet = (depth > 0) & domain
                    else:
                        d_depth = (merge_in_memory(design_outputs, "SLR_0", out_w, out_h, geometry, f"RP{c}")
                                   if design_outputs is not None else None)
                        if d_depth is None:
                            print(f"    WARNING: no RP{c} SLR_0 design flood for this delta - its RP{c} "
                                  f"protection is treated as never failing", flush=True)
                            design_km2[c] = float("inf")
                            continue
                        d_wet = (d_depth > 0) & domain
                    design_km2[c] = float(((d_wet & prot.class_masks[c]) * area_grid).sum())

        wet = (depth > 0) & domain  # any depth (not the 0.1 m exposure threshold); False on NaN
        per_slr_raw_km2[slr] = float((wet * area_grid).sum())
        if prot is None:
            per_slr_flooded_km2[slr] = per_slr_raw_km2[slr]
            continue
        failed = frozenset(c for c in prot.classes
                           if float(((wet & prot.class_masks[c]) * area_grid).sum()) > design_km2[c])
        floodable = prot.floodable(failed)
        per_slr_flooded_km2[slr] = float(((wet & floodable) * area_grid).sum())
        per_slr_prot[slr] = {"failed": sorted(failed),
                             "pct_floodable": 100.0 * float((prot.land & floodable).sum()) / prot.n_land}

    shapefile_km2 = float(delta_row["Area"])
    pct_covered = 100.0 * modeled_km2 / shapefile_km2 if shapefile_km2 > 0 else float("nan")

    return {
        "n_tiles": n_tiles,
        "delta_area_km2_modeled": modeled_km2,
        "delta_area_km2_shapefile": shapefile_km2,
        "pct_delta_covered_by_model": pct_covered,
        "flooded_km2_by_slr": per_slr_flooded_km2,
        "raw_km2_by_slr": per_slr_raw_km2,
        "protection": None if prot is None else {"main_rp": prot.main_rp, "pct_protected": prot.pct_protected,
                                                 "classes": prot.classes, "design_km2": design_km2},
        "per_slr_protection": per_slr_prot,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=str(_REPO_ROOT / "snakemake_workflow" / "config"))
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS_DEFAULT)
    parser.add_argument("--out", default=None, help="default: {root}/deltas_floodmaps/delta_flood_extent.csv")
    parser.add_argument("--design-outputs", default=None,
                        help="model_outputs of the design-event runs (prepare_delta_design_runs.py); default: "
                             "{root}/deltas_floodmaps/design/model_outputs")
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

    default_rp = float(prod_cfg.get("protection", {}).get("default_rp", 5.0))
    catalog = get_data_catalog(_REPO_ROOT / prod_cfg["paths"]["hydromt_data_catalog"], root=prod_cfg["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")
    rp_by_geogunit = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)
    rp_by_geogunit.index = rp_by_geogunit.index.astype("int64")
    prot_inputs = {"catalog": catalog, "rp_by_geogunit": rp_by_geogunit, "default_rp": default_rp, "root": root}
    design_outputs = Path(args.design_outputs) if args.design_outputs else root / "deltas_floodmaps" / "design" / "model_outputs"
    if not design_outputs.exists():
        print(f"WARNING: {design_outputs} missing - protections above 100 yr are treated as never failing")
        design_outputs = None
    print(f"\nFLOPROS protection: coastal, else riverine, else {default_rp:g} yr; cells >= "
          f"{PROTECTION_THRESHOLD_YEARS:g} yr and enclosed low-standard cells do not flood", flush=True)

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
            result = score_delta(delta_row, candidate_tiles, model_outputs, prot_inputs, design_outputs)
            prot = result["protection"] or {"main_rp": float("nan"), "pct_protected": float("nan"), "classes": [],
                                             "design_km2": {}}
            elapsed = time.time() - t0
            modeled_km2 = result["delta_area_km2_modeled"]

            print(f"    -> {result['n_tiles']} tile(s), "
                  f"modeled {modeled_km2:.1f}/{result['delta_area_km2_shapefile']:.1f} km2 "
                  f"({result['pct_delta_covered_by_model']:.1f}% covered) - {elapsed:.1f}s "
                  f"[{(time.time() - t_start) / 60:.1f}min elapsed]", flush=True)
            if prot["classes"]:
                fails = {slr: v["failed"] for slr, v in result["per_slr_protection"].items() if v["failed"]}
                print(f"       protection: {prot['pct_protected']:.1f}% >= {PROTECTION_THRESHOLD_YEARS:g} yr, "
                      f"design classes {prot['classes']} (design-event flooded km2 "
                      f"{ {c: round(v, 1) for c, v in prot['design_km2'].items()} }); failed: {fails or 'never'}",
                      flush=True)

            for slr in SLR_SCENARIOS:
                sp = result["per_slr_protection"].get(slr, {"failed": [], "pct_floodable": 100.0})
                status = ("unprotected" if not prot["classes"] else
                          "protection_failed" if sp["failed"] else
                          "protected" if sp["pct_floodable"] == 0.0 else "partly_protected")
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
                    "flooded_area_km2_unprotected": round(result["raw_km2_by_slr"][slr], 4),
                    "protection_main_rp_years": prot["main_rp"],
                    "pct_area_protected_ge100": round(prot["pct_protected"], 2),
                    "pct_area_floodable": round(sp["pct_floodable"], 2),
                    "failed_design_rps": " ".join(str(c) for c in sp["failed"]),
                    "protection_status": status,
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
