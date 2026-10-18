"""Set up the extra delta simulations the protection-failure rule needs: for every delta that is (partly)
protected to >= PROTECTION_THRESHOLD_YEARS by FLOPROS, the flood of its design event(s) at today's sea
level (SLR_0). compute_delta_flood_extent.py then counts a protection as failed once the delta's RP100
flood at a higher sea level covers more of the area protected to that standard than the design event
did at SLR_0 (same idea as the main pipeline's baseline protection, compute_exposure_analysis.py, with
the delta-wide flooded area as the proxy instead of 1 km flood fractions).

Design RP per protected area: its FLOPROS standard (Coastal, else Riverine, else protection.default_rp)
floored to the nearest simulated COAST-RP return period at or below it (100/250/500/1000; anything
above 1000 -> 1000) - compute_delta_flood_extent.snap_rp. RP100 at SLR_0 already exists in
the delta runs, so only the design RPs above 100 are simulated here.

Writes
  {processed_inputs}/mask/deltas_design_tiles.gpkg   tiles of the selected deltas plus, for every
                                                    hop>=1 tile, its lower-hop neighbours (recursively)
                                                    - the tiles it takes its boundary water levels from
  snakemake_workflow/config/deltas_design_materialized.yml
                                                    deltas_ssp245_materialized.yml (so every model
                                                    setting is identical to the delta runs) with only
                                                    tile_grid, return periods (design RPs), sea level
                                                    (SLR_0 only) and the output folders changed
  {root}/deltas_floodmaps/design/design_rps_by_delta.csv

Then run the simulations with
    python run_delta_floodmaps.py --scenarios design --stages boundary_conditions preprocess simulate

Usage:
    python analysis/prepare_delta_design_runs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from validation import tri_domain_mask  # noqa: E402

from compute_delta_flood_extent import (  # noqa: E402
    DELTA_NAME_INFERRED, GEOGUNIT_SOURCE, PROTECTION_THRESHOLD_YEARS, VOLGA_ID, build_tile_delta_join,
    snap_rp,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _REPO_ROOT / "snakemake_workflow" / "config"


def design_rps_for_delta(delta_geometry, catalog, rp_by_geogunit: pd.Series, default_rp: float) -> list[int]:
    """Snapped design RPs (>= threshold) of the FLOPROS units covering the delta polygon."""
    try:
        geo = catalog.get_rasterdataset(GEOGUNIT_SOURCE, bbox=list(delta_geometry.bounds),
                                        variables=["Geogunits"]).squeeze(drop=True)
    except Exception:  # noqa: BLE001
        return []
    arr = np.nan_to_num(geo.values.astype("float64"), nan=-1)
    inside = tri_domain_mask(gpd.GeoDataFrame(geometry=[delta_geometry], crs="EPSG:4326"),
                             geo.raster.transform, geo.raster.crs, arr.shape)
    ids = np.unique(arr[inside & (arr >= 0)].astype("int64"))
    rps = [float(rp_by_geogunit.get(int(i), default_rp)) for i in ids]
    return sorted({snap_rp(r) for r in rps if r >= PROTECTION_THRESHOLD_YEARS})


def with_upstream_neighbours(tile_grid: gpd.GeoDataFrame, tile_ids: set[int]) -> set[int]:
    """`tile_ids` plus, recursively, every lower-hop tile intersecting a hop>=1 tile already in the set
    (the neighbours a hop>=1 tile seeds its water levels from)."""
    grid = tile_grid.set_index("tile_id")
    out, todo = set(tile_ids), list(tile_ids)
    while todo:
        t = todo.pop()
        hop = int(grid.at[t, "hop_distance"])
        if hop == 0:
            continue
        geom = grid.at[t, "geometry"]
        cand = grid[(grid["hop_distance"] < hop) & grid.intersects(geom)].index
        for c in cand:
            if int(c) not in out:
                out.add(int(c))
                todo.append(int(c))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-config", default=str(_CONFIG_DIR / "deltas_ssp245_materialized.yml"),
                        help="delta run config the design runs copy every model setting from")
    args = parser.parse_args()

    base = yaml.safe_load(open(args.base_config, encoding="utf-8"))
    cfg = load_config(args.base_config)
    prod = load_config(_CONFIG_DIR / "config.yml")
    default_rp = float(prod.get("protection", {}).get("default_rp", 5.0))
    catalog = get_data_catalog(_REPO_ROOT / prod["paths"]["hydromt_data_catalog"], root=prod["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")
    rp_by_geogunit = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)
    rp_by_geogunit.index = rp_by_geogunit.index.astype("int64")

    deltas = retry_transient_io(gpd.read_file, Path(prod["paths"]["deltas_shapefile"]))
    deltas = deltas[deltas["Id"] != VOLGA_ID].reset_index(drop=True)
    rows = []
    for _, d in deltas.iterrows():
        rps = design_rps_for_delta(d.geometry, catalog, rp_by_geogunit, default_rp)
        if rps:
            rows.append({"delta_id": int(d["Id"]), "delta_name_inferred": DELTA_NAME_INFERRED.get(int(d["Id"]), ""),
                         "design_rps": " ".join(str(r) for r in rps)})
    sel = pd.DataFrame(rows)
    print(f"{len(sel)} delta(s) with areas protected to >= {PROTECTION_THRESHOLD_YEARS:g} yr:")
    print(sel.to_string(index=False))
    rps_to_run = sorted({int(r) for s in sel["design_rps"] for r in s.split() if int(r) > 100})
    print(f"design return periods to simulate at SLR_0 (RP100 exists already): {rps_to_run}")

    tile_grid = retry_transient_io(gpd.read_file, Path(cfg["tile_grid"]["path"]))
    tile_grid["tile_id"] = tile_grid["tile_id"].astype(int)
    # deltas whose classes are all RP100 need no extra run (RP100 SLR_0 exists already)
    to_run = sel[sel["design_rps"].map(lambda s: any(int(r) > 100 for r in s.split()))]
    joined = build_tile_delta_join(tile_grid.to_crs(deltas.crs), deltas[deltas["Id"].isin(to_run["delta_id"])])
    direct = {int(t) for ts in joined.values() for t in ts}
    tiles = with_upstream_neighbours(tile_grid, direct)
    print(f"{len(to_run)} delta(s) need design runs ({', '.join(to_run['delta_name_inferred'])}); "
          f"{len(direct)} tile(s) intersect them, {len(tiles)} with their lower-hop neighbours")

    root = Path(cfg["paths"]["root"])
    tiles_path = Path(cfg["tile_grid"]["path"]).with_name("deltas_design_tiles.gpkg")
    tile_grid[tile_grid["tile_id"].isin(tiles)].to_file(tiles_path, driver="GPKG")
    out_dir = root / "deltas_floodmaps" / "design"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out_dir / "design_rps_by_delta.csv", index=False)

    design = dict(base)
    design["tile_grid"] = {**base["tile_grid"], "path": base["tile_grid"]["path"].replace(
        "deltas_global_tiles.gpkg", "deltas_design_tiles.gpkg")}
    design["boundary_conditions"] = {**base["boundary_conditions"], "return_periods": rps_to_run,
                                     "slr_scenarios": ["SLR_0"],
                                     "waterlevel_nc_dir": "{root}/deltas_floodmaps/design/WL_scenarios"}
    design["simulation"] = {**base["simulation"], "model_outputs": "{root}/deltas_floodmaps/design/model_outputs"}
    design["postprocessing"] = {**base["postprocessing"], "merged_outputs": "{root}/deltas_floodmaps/design/merged_results"}
    design["hpc"] = {**base.get("hpc", {}), "jobs_dir": "{root}/deltas_floodmaps/design/model_outputs/hpc_jobs"}
    if "visualization" in base:
        design["visualization"] = {**base["visualization"], "output_dir": "{root}/deltas_floodmaps/design/figures"}
    out_cfg = _CONFIG_DIR / "deltas_design_materialized.yml"
    with open(out_cfg, "w", encoding="utf-8") as f:
        f.write("# Generated by analysis/prepare_delta_design_runs.py from "
                f"{Path(args.base_config).name} - design-event floods (SLR_0) for the protected deltas.\n")
        yaml.safe_dump(design, f, sort_keys=False)
    print(f"Wrote {tiles_path}\nWrote {out_cfg}\nWrote {out_dir / 'design_rps_by_delta.csv'}")
    print("Next: python run_delta_floodmaps.py --scenarios design --stages boundary_conditions preprocess simulate")


if __name__ == "__main__":
    main()
