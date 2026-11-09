"""Write a tile's calibration_result.json: every flood-model run vs SFINCS on the 30 m subgrid.

Needs hydromt-sfincs-dev (reads sfincs.msk through SfincsModel). Writes the exact depth-difference
histograms of every run, area- and population-weighted, to calibration_depth_accuracy.npz beside it.

Usage:
    python compute_calibration_result.py --config <config.yml> --base-dir-name calibration_roughness_SFINCS --tile-id 1122
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from hydromt_sfincs import SfincsModel
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import (  # noqa: E402
    DEPTH_ACCURACY_NAME, RUNS_NAME, WATERDEPTH_NODATA_INT16, WATERDEPTH_SCALE, decode_waterdepth_cm, run_path,
)
from calibration_split import split_group  # noqa: E402
from flood_agreement import (  # noqa: E402
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, WET_THRESHOLD_M, confusion_counts, depth_corr_sufficient_stats,
    depth_diff_hists, depth_joint_hist, metrics_from_counts,
)
from gfm_config import read_root, resolve_catalog_path  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

# Appended, not inserted: most of src/ needs the gfm env's hydromt; population_grid does not.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from population_grid import population_per_pixel  # noqa: E402

DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "data_catalog_gfm.yml"

LAND_CODE, OCEAN_CODE = 0, 1


def _depth_stats(depth: np.ndarray, wet: np.ndarray) -> dict:
    if not wet.any():
        return {"mean_m": None, "median_m": None, "max_m": None}
    v = depth[wet]
    return {"mean_m": float(v.mean()), "median_m": float(np.median(v)), "max_m": float(v.max())}


def _sfincs_active_on_subgrid(sfincs_dir: Path, sg_shape: tuple, sg_transform) -> tuple:
    # The subgrid shares the coarse grid's origin and rotation at exactly a quarter of its cell size.
    sf = SfincsModel(root=str(sfincs_dir), mode="r")
    retry_transient_io(sf.grid.read)
    msk = sf.grid.data["mask"].values
    tr = sf.grid.data.raster.transform
    f = int(round(tr.a / sg_transform.a))
    if not (np.isclose(tr.c, sg_transform.c) and np.isclose(tr.f, sg_transform.f)
            and (msk.shape[0] * f, msk.shape[1] * f) == tuple(sg_shape)):
        raise ValueError(f"sfincs grid {msk.shape} @ {tr} does not nest into subgrid {sg_shape} @ {sg_transform}")
    active = np.kron(msk > 0, np.ones((f, f), dtype=bool))
    return active, {"n_active_cells": int((msk > 0).sum()), "n_boundary_cells": int((msk == 2).sum())}, msk, tr


def _compare(depth: np.ndarray, wet: np.ndarray, ref: dict) -> dict:
    matched, model_only, sfincs_only = confusion_counts(wet, ref["sfincs_wet"], ref["score"], ref["weight"])
    out = {"matched_km2": matched, "model_only_km2": model_only, "sfincs_only_km2": sfincs_only,
           **metrics_from_counts(matched, model_only, sfincs_only), "depth_joint": None}
    if ref["people"] is not None:
        out.update(zip(("matched_pop", "model_only_pop", "sfincs_only_pop"),
                       confusion_counts(wet, ref["sfincs_wet"], ref["score"], ref["people"])))
    both = wet & ref["sfincs_wet"]
    if both.any():
        x = ref["sfincs_depth"][both].astype(np.float64)
        y = depth[both].astype(np.float64)
        out["depth_joint"] = {**depth_corr_sufficient_stats(x, y),
                              "hist_fine": depth_joint_hist(x, y, DEPTH_CORR_FINE_EDGES).tolist(),
                              "hist_category": depth_joint_hist(x, y, DEPTH_CATEGORY_EDGES).tolist()}
    return out


def _read_on_grid(path: Path, transform, crs, shape: tuple, int16_cm: bool) -> np.ndarray | None:
    """Read the part of a depth raster over the destination grid, reprojected onto it (NaN: no data)."""
    h, w = shape
    with retry_transient_io(rasterio.open, path) as src:
        left, bottom, right, top = transform_bounds(crs, src.crs, *rasterio.transform.array_bounds(h, w, transform),
                                                    densify_pts=21)
        # Affine arithmetic, not windows.from_bounds: its GDAL transformer crashes hydromt-sfincs-dev on Windows.
        inv = ~src.transform
        cols, rows = zip(*(inv * xy for xy in ((left, top), (right, top), (left, bottom), (right, bottom))))
        c0, c1 = max(0, int(np.floor(min(cols)))), min(src.width, int(np.ceil(max(cols))))
        r0, r1 = max(0, int(np.floor(min(rows)))), min(src.height, int(np.ceil(max(rows))))
        if c1 <= c0 or r1 <= r0:
            return None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        raw, src_tr, src_crs, nodata = src.read(1, window=win), src.window_transform(win), src.crs, src.nodata
    if int16_cm:
        arr = raw.astype(np.float32) / WATERDEPTH_SCALE
        arr[raw == (nodata if nodata is not None else WATERDEPTH_NODATA_INT16)] = np.nan
    else:
        arr = raw.astype(np.float32)
        if nodata is not None and not np.isnan(nodata):
            arr[raw == nodata] = np.nan
    out = np.full(shape, np.nan, dtype=np.float32)
    reproject(arr, out, src_transform=src_tr, src_crs=src_crs, dst_transform=transform, dst_crs=crs,
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def _max_merged(depth: np.ndarray, paths: list[Path], transform, crs, int16_cm: bool) -> np.ndarray:
    for p in paths:
        if p.exists():
            other = _read_on_grid(p, transform, crs, depth.shape, int16_cm)
            if other is not None:
                depth = np.fmax(depth, other)
    return depth


def compute_result(tile_dir: Path, flooding: dict, core_mask: np.ndarray | None = None,
                   pop_path: Path | None = None,
                   neighbor_dirs: list[Path] | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Build one tile's calibration_result dict vs SFINCS's hmax_subgrid.tif, and its depth histograms.

    Scoring area: real land (mask.tif == LAND_CODE) active in SFINCS
    (sfincs.msk > 0, each 120 m cell -> its 4x4 subgrid pixels); inactive land
    is left out of every count, the model rasters themselves are not masked.
    Every run of outputs/model_runs.json is scored as written (the connectivity rule is part
    of the model). `flooding` is copied verbatim into "settings".

    Parameters
    ----------
    core_mask : np.ndarray of bool, optional
        On the hmax_subgrid.tif grid. When given, the scoring area is also
        restricted to it (the tile's Voronoi core, so overlapping tiles do not
        count shared land twice) and the result gains ``"dedup": True`` and
        ``domain["core_fraction_of_active_land"]``.
    neighbor_dirs : list of Path, optional
        Tile dirs of the other calibration tiles overlapping this one. SFINCS and every run
        then take, per cell, the maximum over this tile and those (production's tile merge,
        merge.merge_tile_rasters_chunk); with `core_mask`, each place is scored once.
    pop_path : Path, optional
        WorldPop counts. When given, each run also gets population-weighted extent counts
        (``matched_pop`` ...) and its histograms a population weight (people per subgrid pixel,
        population_grid.population_per_pixel).

    The histograms map ``"<run key>__area"`` (km2) and ``"<run key>__pop"`` (people) to
    flood_agreement.depth_diff_hists over the scoring area.
    """
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    with retry_transient_io(rasterio.open, sfincs_dir / "hmax_subgrid.tif") as src:
        sfincs_depth = src.read(1).astype(np.float32)
        nodata, transform, crs, shape = src.nodata, src.transform, src.crs, src.shape
    if nodata is not None and not np.isnan(nodata):
        sfincs_depth[sfincs_depth == nodata] = np.nan
    neighbors = neighbor_dirs or []
    sfincs_depth = _max_merged(sfincs_depth, [n / "sfincs_model" / "hmax_subgrid.tif" for n in neighbors],
                               transform, crs, int16_cm=False)

    mog = np.empty(shape, dtype=np.float32)
    with retry_transient_io(rasterio.open, tile_dir / "inputs" / "mask.tif") as src:
        reproject(source=rasterio.band(src, 1), destination=mog, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest)
    land = mog == LAND_CODE
    active, counts, msk, coarse_tr = _sfincs_active_on_subgrid(sfincs_dir, shape, transform)
    score = land & active
    if core_mask is not None:
        score = score & core_mask
    cell_km2 = abs(transform.a * transform.e - transform.b * transform.d) / 1e6
    people = (population_per_pixel(pop_path, shape, transform, crs,
                                   open_raster=partial(retry_transient_io, rasterio.open))[0]
              if pop_path is not None else None)
    ref = {"score": score, "weight": np.full(shape, cell_km2, dtype=np.float64), "people": people,
           "sfincs_depth": sfincs_depth,
           "sfincs_wet": score & np.isfinite(sfincs_depth) & (sfincs_depth > WET_THRESHOLD_M)}

    domain = {"land_km2": float(land.sum() * cell_km2), "sfincs_active_land_km2": float(score.sum() * cell_km2),
              "sfincs_inactive_land_km2": float((land & ~active).sum() * cell_km2)}
    if core_mask is not None:
        domain["core_fraction_of_active_land"] = float(score.sum() / max(1, (land & active).sum()))
    domain.update(counts)
    forcing = sfincs_dir / "matched_boundary_points.gpkg"
    domain["n_forcing_points"] = int(len(gpd.read_file(forcing))) if forcing.exists() else None
    land_c = land[::4, ::4][: msk.shape[0], : msk.shape[1]]
    bnd = msk == 2
    if land_c.any() and bnd.any():
        d = ndimage.distance_transform_edt(~land_c, sampling=abs(coarse_tr.a))[bnd] / 1000.0
        domain.update(boundary_dist_mean_km=float(d.mean()), boundary_dist_median_km=float(np.median(d)),
                      boundary_dist_max_km=float(d.max()))

    result = {
        "tile_id": tile_dir.name, "group": split_group(tile_dir.name), "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        **({"dedup": True} if core_mask is not None else {}),
        **({"max_merged_neighbors": [n.name for n in neighbors]} if neighbor_dirs is not None else {}),
        "settings": {"wet_threshold_m": WET_THRESHOLD_M, "subgrid_res_m": float(abs(transform.a)),
                     "flooding": flooding},
        "domain": domain,
        "sfincs": {"flooded_km2": float(ref["sfincs_wet"].sum() * cell_km2),
                   "depth": _depth_stats(sfincs_depth, ref["sfincs_wet"])},
        "runs": {},
    }

    if people is not None:
        result["sfincs"]["exposed_pop"] = float(people[ref["sfincs_wet"]].sum())
    hists = {}
    runs_path = out_dir / RUNS_NAME
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    for key, meta in runs.items():
        p = run_path(out_dir, key)
        if not p.exists():
            continue
        depth, _, _, rshape = decode_waterdepth_cm(p)
        if rshape != shape:
            continue
        depth = _max_merged(depth, [run_path(n / "outputs", key) for n in neighbors], transform, crs, int16_cm=True)
        wet = score & np.isfinite(depth) & (depth > WET_THRESHOLD_M)
        result["runs"][key] = {"variant": meta["variant"], "parameter": meta["parameter"], "alpha": meta["alpha"],
                               "flooded_km2": float(wet.sum() * cell_km2), "depth": _depth_stats(depth, wet),
                               "vs_sfincs": _compare(depth, wet, ref), "solver": meta.get("solver"),
                               "runtime_s": meta.get("runtime_s")}
        for name, w in (("area", ref["weight"]), ("pop", people)):
            if w is not None:
                hists[f"{key}__{name}"] = depth_diff_hists(depth, sfincs_depth, wet, ref["sfincs_wet"], w)
    return result, hists


def write_result(out_dir: Path, result: dict, hists: dict[str, np.ndarray]) -> None:
    """Write calibration_result.json and the depth histograms to `out_dir`."""
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((out_dir / "calibration_result.json").write_text, json.dumps(result, indent=1), encoding="utf-8")
    retry_transient_io(np.savez_compressed, out_dir / DEPTH_ACCURACY_NAME, **hists)


def population_path(config_path: Path, catalog_path: Path = DEFAULT_CATALOG) -> Path:
    """Return the data catalog's ``population`` raster (WorldPop counts)."""
    return resolve_catalog_path(catalog_path, read_root(config_path), "population")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="data catalog holding the population raster")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))
    tile_dir = root / args.base_dir_name / args.tile_id
    result, hists = compute_result(tile_dir, config["simulation"]["flooding"],
                                   pop_path=population_path(Path(args.config), Path(args.catalog)))
    out = tile_dir / "outputs" / "calibration_result.json"
    write_result(out.parent, result, hists)
    csi = ", ".join(f"{k}:{v['vs_sfincs']['CSI']:.3f}" for k, v in result["runs"].items())
    print(f"tile {args.tile_id} ({result['group']}): sfincs {result['sfincs']['flooded_km2']:.1f} km2 "
          f"(inactive land {result['domain']['sfincs_inactive_land_km2']:.1f} km2 excluded); CSI {csi} -> {out}")


if __name__ == "__main__":
    main()
