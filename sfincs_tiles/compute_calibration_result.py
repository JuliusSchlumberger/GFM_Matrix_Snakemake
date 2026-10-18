"""Write a tile's calibration_result.json: SFINCS vs bathtub vs eikonal on the 30 m subgrid.

Needs hydromt-sfincs-dev (reads sfincs.msk through SfincsModel).

Usage:
    python compute_calibration_result.py --config <config.yml> --base-dir-name calibration_roughness_SFINCS --tile-id 1122
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from hydromt_sfincs import SfincsModel
from rasterio.warp import Resampling, reproject
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import (  # noqa: E402
    FRICTION_SCALE_FACTORS, WATERLEVEL_NAME, bathtub_path, decode_waterdepth_cm, eikonal_path,
)
from flood_agreement import (  # noqa: E402
    DEPTH_CATEGORY_EDGES, DEPTH_CORR_FINE_EDGES, WET_THRESHOLD_M, confusion_counts, depth_corr_sufficient_stats,
    depth_joint_hist, metrics_from_counts, prune_to_ocean_connected,
)
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

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
    both = wet & ref["sfincs_wet"]
    if both.any():
        x = ref["sfincs_depth"][both].astype(np.float64)
        y = depth[both].astype(np.float64)
        out["depth_joint"] = {**depth_corr_sufficient_stats(x, y),
                              "hist_fine": depth_joint_hist(x, y, DEPTH_CORR_FINE_EDGES).tolist(),
                              "hist_category": depth_joint_hist(x, y, DEPTH_CATEGORY_EDGES).tolist()}
    return out


def compute_result(tile_dir: Path, fsfs: list[float], flooding: dict, river_code: int,
                   core_mask: np.ndarray | None = None) -> dict:
    """Build one tile's calibration_result dict vs SFINCS's hmax_subgrid.tif.

    Scoring area: real land (mask.tif == LAND_CODE) active in SFINCS
    (sfincs.msk > 0, each 120 m cell -> its 4x4 subgrid pixels); inactive land
    is left out of every count, the model rasters themselves are not masked.
    Bathtub is pruned to coast-connected flooding first; eikonal and SFINCS are
    connected by construction. `flooding` is copied verbatim into "settings".

    Parameters
    ----------
    core_mask : np.ndarray of bool, optional
        On the hmax_subgrid.tif grid. When given, the scoring area is also
        restricted to it (the tile's Voronoi core, so overlapping tiles do not
        count shared land twice) and the result gains ``"dedup": True`` and
        ``domain["core_fraction_of_active_land"]``.
    """
    sfincs_dir = tile_dir / "sfincs_model"
    out_dir = tile_dir / "outputs"
    with retry_transient_io(rasterio.open, sfincs_dir / "hmax_subgrid.tif") as src:
        sfincs_depth = src.read(1).astype(np.float32)
        nodata, transform, crs, shape = src.nodata, src.transform, src.crs, src.shape
    if nodata is not None and not np.isnan(nodata):
        sfincs_depth[sfincs_depth == nodata] = np.nan

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
    ref = {"score": score, "weight": np.full(shape, cell_km2, dtype=np.float64), "sfincs_depth": sfincs_depth,
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
        "tile_id": tile_dir.name, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        **({"dedup": True} if core_mask is not None else {}),
        "settings": {"wet_threshold_m": WET_THRESHOLD_M, "subgrid_res_m": float(abs(transform.a)),
                     "friction_scale_factors": list(fsfs), "flooding": flooding},
        "domain": domain,
        "sfincs": {"flooded_km2": float(ref["sfincs_wet"].sum() * cell_km2),
                   "depth": _depth_stats(sfincs_depth, ref["sfincs_wet"])},
        "bathtub": None, "eikonal": {},
    }

    bp = bathtub_path(out_dir)
    if bp.exists():
        depth, _, _, bshape = decode_waterdepth_cm(bp)
        if bshape == shape:
            wet = prune_to_ocean_connected(land & np.isfinite(depth) & (depth > WET_THRESHOLD_M), mog,
                                           ocean_code=OCEAN_CODE, land_code=LAND_CODE, river_code=river_code) & score
            boundaries = gpd.read_file(tile_dir / "inputs" / "boundaries_RP100_SLR_0.gpkg")
            # SLR_0 is int16 cm (rasters.decode_waterlevel_cm, not importable in this env)
            result["bathtub"] = {"level_m": float(boundaries[WATERLEVEL_NAME].max()) / 100.0,
                                 "flooded_km2": float(wet.sum() * cell_km2), "depth": _depth_stats(depth, wet),
                                 "vs_sfincs": _compare(depth, wet, ref)}

    diag_path = out_dir / "eikonal_solver_diagnostics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    for fsf in fsfs:
        p = eikonal_path(out_dir, fsf)
        if not p.exists():
            continue
        depth, _, _, eshape = decode_waterdepth_cm(p)
        if eshape != shape:
            continue
        wet = score & np.isfinite(depth) & (depth > WET_THRESHOLD_M)
        key = f"fsf_{fsf:g}"
        result["eikonal"][key] = {"flooded_km2": float(wet.sum() * cell_km2), "depth": _depth_stats(depth, wet),
                                  "vs_sfincs": _compare(depth, wet, ref), "solver": diag.get(key)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=list(FRICTION_SCALE_FACTORS))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    root = read_root(Path(args.config))
    tile_dir = root / args.base_dir_name / args.tile_id
    result = compute_result(tile_dir, args.friction_scale_factors, config["simulation"]["flooding"],
                            int(config["tile_generation"]["river_code"]))
    out = tile_dir / "outputs" / "calibration_result.json"
    retry_transient_io(out.write_text, json.dumps(result, indent=1), encoding="utf-8")
    e = result["eikonal"]
    csi = ", ".join(f"{k[4:]}:{v['vs_sfincs']['CSI']:.3f}" for k, v in e.items())
    print(f"tile {args.tile_id}: sfincs {result['sfincs']['flooded_km2']:.1f} km2 "
          f"(inactive land {result['domain']['sfincs_inactive_land_km2']:.1f} km2 excluded); "
          f"bathtub CSI {result['bathtub']['vs_sfincs']['CSI'] if result['bathtub'] else None}; eikonal CSI {csi} -> {out}")


if __name__ == "__main__":
    main()
