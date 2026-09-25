"""Select tiles for the SFINCS-vs-eikonal validation batch: a real ocean
edge (hop_distance==0), a minimum ocean fraction, antimeridian-excluded,
a high-latitude cutoff, spatially stratified for global scatter, no
area-based pre-filter. Reuses select_test_tiles.py's eligibility logic
(evaluate_tile/build_coast_hg_tree: hop_distance==0, mask.tif size cap,
<=1% river, real COAST-HG match).

High-latitude exclusion: tiles far from the equator suffer worse UTM-
reprojection distortion (meridian convergence curves a tile's straight
lon/lat edges once reprojected to SFINCS's own axis-aligned UTM grid).
`--max-abs-lat-deg` (default 55) drops any tile whose centroid falls
outside that band.

Works directly off model_outputs/{tile_id}/inputs/ as it currently exists
on disk, not a fresh catalog re-derivation. A tile without a real
model_outputs/<id>/inputs/mask.tif yet is excluded up front, explicitly
and reported.

Writes a single tile_ids.txt (every selected tile ID).

Antimeridian exclusion: tiles straddling +-180 deg break
hydromt_sfincs's own water_level.create() with a GEOS topology exception
(see build_sfincs_tile.py's own _classify_water_level_create_error). A
tile whose own geometry bounds span an implausibly wide longitude range
(only possible for a real coastal tile if its polygon wraps the
dateline) is flagged and dropped.

Also writes, from the same selected sample:
  - tile_locations_map.png: global Equal Earth map of tile locations
    (light grey land, no figure title)
  - tile_selection_histograms.png: the selected sample's own size/ocean-
    fraction distribution against the eligible pool it was drawn from
  - tile_selection_statistics.xlsx: summary statistics tables (tile
    counts, size stats, ocean-fraction stats)

Usage:
    python select_validation_tiles.py
    python select_validation_tiles.py --n-tiles 520
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_test_tiles import (  # noqa: E402
    MAX_CELLS_DEFAULT,
    MAX_RIVER_FRAC_DEFAULT,
    OCEAN_CODE,
    GRID_DEG,
    build_coast_hg_tree,
    evaluate_tile,
)
from gfm_config import read_root  # noqa: E402

N_TILES_DEFAULT = 520
MIN_OCEAN_FRAC_DEFAULT = 0.05
ANTIMERIDIAN_BBOX_WIDTH_DEG_DEFAULT = 60.0  # real GFM tiles are never anywhere near this wide
MAX_ABS_LAT_DEG_DEFAULT = 55.0  # see module docstring's "High-latitude exclusion" note
BASE_DIR_NAME_DEFAULT = "validation_sfincs_v4"

LAND_COLOR = "#d8d8d4"
OCEAN_COLOR = "#fcfcfb"
COAST_COLOR = "#b8b8b3"
SELECTED_COLOR = "#2a78d6"
POPULATION_COLOR = "#b3b3ad"


def read_ocean_frac(tile_id: int, root: Path) -> float | None:
    """Unconditional mask.tif read for the histogram's population - unlike
    evaluate_tile() (select_test_tiles.py), which returns None before
    computing ocean_frac for any tile exceeding max_cells, this always
    reads the full array so the population comparison covers every tile.
    Returns None if the tile has no mask.tif at all."""
    mask_path = root / "model_outputs" / str(tile_id) / "inputs" / "mask.tif"
    if not mask_path.exists():
        return None
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return float(np.mean(mask == OCEAN_CODE))


def is_antimeridian_tile(geom, max_width_deg: float) -> bool:
    """True if this tile's own geometry bounds span an implausibly wide
    longitude range - the real signature of a polygon that wraps the dateline
    (naive min/max longitude across a dateline-crossing feature spans nearly
    360 deg, vastly wider than any real GFM tile)."""
    minx, _, maxx, _ = geom.bounds
    return (maxx - minx) > max_width_deg


def stratified_sample_exact(df: pd.DataFrame, n_target: int, grid_deg: float, seed: int) -> pd.DataFrame:
    """Round-robins one tile at a time across coarse global lon/lat bins
    until n_target is met or the pool is exhausted. Guarantees exactly
    min(n_target, len(df)) tiles, still spatially stratified."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["_bin"] = (
        (np.floor(df["lon"] / grid_deg).astype(int)).astype(str) + "_" +
        (np.floor(df["lat"] / grid_deg).astype(int)).astype(str)
    )
    bin_pools = {b: list(g.index) for b, g in df.groupby("_bin")}
    for b in bin_pools:
        rng.shuffle(bin_pools[b])
    bins = list(bin_pools.keys())
    rng.shuffle(bins)

    picked = []
    while len(picked) < n_target and any(bin_pools.values()):
        for b in bins:
            if len(picked) >= n_target:
                break
            if bin_pools[b]:
                picked.append(bin_pools[b].pop())
    return df.loc[picked].drop(columns="_bin").sort_values("tile_id").reset_index(drop=True)


def _area_km2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Lat-corrected bbox area (km2), straight from geometry - same
    convention used throughout this pipeline."""
    b = gdf.geometry.bounds
    lon_span = b["maxx"] - b["minx"]
    lat_span = b["maxy"] - b["miny"]
    lat_mid = (b["maxy"] + b["miny"]) / 2.0
    w_km = lon_span * 111.32 * np.cos(np.radians(lat_mid))
    h_km = lat_span * 110.54
    return (w_km * h_km).to_numpy()


def plot_tile_locations_map(selected_df: pd.DataFrame, out_path: Path) -> None:
    """Global Equal Earth map of selected tile centroids - light grey
    land, no figure title."""
    proj = ccrs.EqualEarth()
    fig = plt.figure(figsize=(14, 7.5), facecolor=OCEAN_COLOR)
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor=COAST_COLOR, linewidth=0.4, zorder=1)
    ax.scatter(
        selected_df["lon"], selected_df["lat"], transform=ccrs.PlateCarree(),
        s=16, c=SELECTED_COLOR, alpha=0.85, linewidths=0.3, edgecolors="white", zorder=3,
    )
    ax.spines["geo"].set_edgecolor(COAST_COLOR)
    ax.spines["geo"].set_linewidth(0.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=OCEAN_COLOR)
    plt.close(fig)


def plot_selection_histograms(selected_df: pd.DataFrame, population_df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel histogram: the selected sample's own area/ocean-fraction
    distribution against the population it was drawn from.

    `population_df` is every hop_distance==0 tile with a real
    model_outputs/<id>/inputs/mask.tif, after only the antimeridian/
    high-latitude/missing-inputs filters - not narrowed further to
    eligible_df's own max_cells/river-frac/COAST-HG-match criteria, since
    those are eligibility constraints for this validation run, not a
    property of what tiles actually exist. Needs both `area_km2_bbox` and
    `ocean_frac` columns - see main()'s own hop0 construction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=OCEAN_COLOR)

    ax = axes[0]
    area_pop = population_df["area_km2_bbox"].to_numpy()
    area_sel = selected_df["area_km2_bbox"].to_numpy()
    bins = np.logspace(np.log10(max(area_pop.min(), 0.1)), np.log10(area_pop.max()), 30)
    ax.hist(area_pop, bins=bins, density=True, color=POPULATION_COLOR, alpha=0.75, label=f"All available tiles (n={len(area_pop)})")
    ax.hist(area_sel, bins=bins, density=True, color=SELECTED_COLOR, alpha=0.55, label=f"Selected (n={len(area_sel)})")
    ax.set_xscale("log")
    ax.set_xlabel("Tile area (km2)")
    ax.set_ylabel("Density")
    ax.set_title("Tile size")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ocean_pop = population_df["ocean_frac"].dropna().to_numpy()
    ocean_sel = selected_df["ocean_frac"].to_numpy()
    bins = np.linspace(0, 1, 26)
    ax.hist(ocean_pop, bins=bins, density=True, color=POPULATION_COLOR, alpha=0.75, label=f"All available tiles (n={len(ocean_pop)})")
    ax.hist(ocean_sel, bins=bins, density=True, color=SELECTED_COLOR, alpha=0.55, label=f"Selected (n={len(ocean_sel)})")
    ax.set_xlabel("Ocean fraction")
    ax.set_ylabel("Density")
    ax.set_title("Ocean fraction")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=OCEAN_COLOR)
    plt.close(fig)


def _stats_row(label: str, area: np.ndarray, ocean: np.ndarray | None = None) -> dict:
    row = {
        "Population": label, "n_tiles": len(area),
        "area_km2_min": float(np.min(area)), "area_km2_median": float(np.median(area)),
        "area_km2_mean": float(np.mean(area)), "area_km2_max": float(np.max(area)),
    }
    if ocean is not None:
        row.update({
            "ocean_frac_min": float(np.min(ocean)), "ocean_frac_median": float(np.median(ocean)),
            "ocean_frac_mean": float(np.mean(ocean)), "ocean_frac_max": float(np.max(ocean)),
        })
    return row


def write_statistics_excel(
    tiles_gdf: gpd.GeoDataFrame, hop0: gpd.GeoDataFrame,
    eligible_df: pd.DataFrame, selected_df: pd.DataFrame, out_path: Path,
) -> None:
    """Summary tables: tile counts, size stats, ocean-fraction stats - the
    same tables reported for this selection in conversation."""
    counts_df = pd.DataFrame([
        {"Population": "Full production grid", "n_tiles": len(tiles_gdf)},
        {"Population": "hop_distance==0 (own real ocean edge)", "n_tiles": len(hop0)},
        {"Population": "Eligible pool (size/river/COAST-HG criteria met)", "n_tiles": len(eligible_df)},
        {"Population": "Selected for SFINCS validation", "n_tiles": len(selected_df)},
    ])

    size_df = pd.DataFrame([
        _stats_row("Full production grid", tiles_gdf["area_km2"].to_numpy()),
        _stats_row("hop_distance==0 subset", hop0["area_km2"].to_numpy()),
        _stats_row("Eligible pool", eligible_df["area_km2_bbox"].to_numpy()),
        _stats_row("Selected", selected_df["area_km2_bbox"].to_numpy()),
    ])[["Population", "n_tiles", "area_km2_min", "area_km2_median", "area_km2_mean", "area_km2_max"]]

    ocean_df = pd.DataFrame([
        _stats_row("Eligible pool", eligible_df["area_km2_bbox"].to_numpy(), eligible_df["ocean_frac"].to_numpy()),
        _stats_row("Selected", selected_df["area_km2_bbox"].to_numpy(), selected_df["ocean_frac"].to_numpy()),
    ])[["Population", "n_tiles", "ocean_frac_min", "ocean_frac_median", "ocean_frac_mean", "ocean_frac_max"]]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        counts_df.to_excel(writer, sheet_name="Tile counts", index=False)
        size_df.to_excel(writer, sheet_name="Size (km2)", index=False)
        ocean_df.to_excel(writer, sheet_name="Ocean fraction", index=False)

    from openpyxl import load_workbook
    from openpyxl.styles import Font
    wb = load_workbook(out_path)
    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for col_cells in ws.columns:
            width = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(width, 10), 40)
    wb.save(out_path)


def main() -> None:
    import argparse

    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--max-cells", type=int, default=MAX_CELLS_DEFAULT)
    parser.add_argument("--max-river-frac", type=float, default=MAX_RIVER_FRAC_DEFAULT)
    parser.add_argument("--n-tiles", type=int, default=N_TILES_DEFAULT)
    parser.add_argument("--min-ocean-frac", type=float, default=MIN_OCEAN_FRAC_DEFAULT)
    parser.add_argument("--antimeridian-width-deg", type=float, default=ANTIMERIDIAN_BBOX_WIDTH_DEG_DEFAULT)
    parser.add_argument(
        "--max-abs-lat-deg", type=float, default=MAX_ABS_LAT_DEG_DEFAULT,
        help="exclude tiles whose centroid latitude falls outside +-this value - see module "
             "docstring's 'High-latitude exclusion' note",
    )
    parser.add_argument("--base-dir-name", default=BASE_DIR_NAME_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    tiles_gdf = gpd.read_file(root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg")
    n_all = len(tiles_gdf)
    hop0 = tiles_gdf[tiles_gdf["hop_distance"] == 0].copy()
    print(f"{n_all} tiles total, {len(hop0)} with hop_distance == 0")

    hop0["antimeridian"] = hop0.geometry.apply(lambda g: is_antimeridian_tile(g, args.antimeridian_width_deg))
    n_antimeridian = int(hop0["antimeridian"].sum())
    hop0 = hop0[~hop0["antimeridian"]].copy()
    print(f"{n_antimeridian} antimeridian-crossing tile(s) excluded, {len(hop0)} remain")

    centroid = hop0.geometry.centroid
    hop0["lat_centroid"] = centroid.y.to_numpy()
    n_high_lat = int((hop0["lat_centroid"].abs() > args.max_abs_lat_deg).sum())
    hop0 = hop0[hop0["lat_centroid"].abs() <= args.max_abs_lat_deg].copy()
    print(f"{n_high_lat} tile(s) excluded for |lat_centroid| > {args.max_abs_lat_deg}, {len(hop0)} remain")

    # Explicit model_outputs existence check, reported up front rather than left as an
    # implicit side-effect of evaluate_tile()'s own per-tile reject reasons.
    model_outputs_root = root / "model_outputs"
    has_inputs = hop0["tile_id"].apply(lambda t: (model_outputs_root / str(int(t)) / "inputs" / "mask.tif").is_file())
    n_missing_inputs = int((~has_inputs).sum())
    hop0 = hop0[has_inputs].copy()
    print(f"{n_missing_inputs} tile(s) excluded - no model_outputs/<id>/inputs/mask.tif yet, {len(hop0)} remain")

    tiles_gdf["area_km2"] = _area_km2(tiles_gdf)
    hop0["area_km2"] = _area_km2(hop0)
    hop0["area_km2_bbox"] = hop0["area_km2"]

    # ocean_frac for the FULL hop0 population - see plot_selection_histograms's own docstring.
    print(f"Reading ocean_frac for all {len(hop0)} available tile(s) (for the histogram's own "
          f"population, not just the eligible subset)...", flush=True)
    hop0["ocean_frac"] = hop0["tile_id"].apply(lambda t: read_ocean_frac(int(t), root))

    coast_hg_tree = build_coast_hg_tree(root / "inputs" / "COAST_HG" / "COAST-HG_RP100.nc")

    records = []
    n_checked = 0
    for tile_id in hop0["tile_id"]:
        n_checked += 1
        rec = evaluate_tile(int(tile_id), root, coast_hg_tree, args.max_cells, args.max_river_frac)
        if rec is not None:
            records.append(rec)
        if n_checked % 200 == 0:
            print(f"  checked {n_checked}/{len(hop0)}, {len(records)} eligible so far", flush=True)

    eligible_df = pd.DataFrame(records)
    eligible_df = eligible_df.merge(
        hop0[["tile_id", "lat_centroid", "area_km2_bbox"]], on="tile_id", how="left",
    )
    print(f"\n{len(eligible_df)} of {len(hop0)} hop_distance==0, non-antimeridian tiles eligible "
          f"(<= {args.max_cells:,} cells, <= {args.max_river_frac:.1%} river, real COAST-HG match)")

    out_dir = root / args.base_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible_df.to_csv(out_dir / "eligible_tile_pool.csv", index=False)
    print(f"Wrote {out_dir / 'eligible_tile_pool.csv'} (full eligible pool, for audit)")

    # -- single selection: ocean_frac floor, antimeridian + high-latitude + missing-inputs
    # already excluded, NO area-based pre-filter, spatially stratified --
    pool = eligible_df[eligible_df["ocean_frac"] >= args.min_ocean_frac].copy()
    print(f"  {len(pool)} of {len(eligible_df)} eligible tiles have ocean_frac >= {args.min_ocean_frac:.0%}")
    selected = stratified_sample_exact(pool, min(args.n_tiles, len(pool)), GRID_DEG, args.seed)
    print(f"\nSelected: {len(selected)} tiles (ocean_frac >= {args.min_ocean_frac:.0%}, "
          f"antimeridian-excluded, |lat| <= {args.max_abs_lat_deg}, stratified)")

    selected.to_csv(out_dir / "tile_selection_metadata.csv", index=False)
    (out_dir / "tile_ids.txt").write_text("\n".join(str(t) for t in selected["tile_id"]) + "\n")
    print(f"\nWrote {out_dir / 'tile_selection_metadata.csv'}, {out_dir / 'tile_ids.txt'} ({len(selected)})")

    map_path = out_dir / "tile_locations_map.png"
    plot_tile_locations_map(selected, map_path)
    print(f"Wrote {map_path}")

    hist_path = out_dir / "tile_selection_histograms.png"
    plot_selection_histograms(selected, hop0, hist_path)
    print(f"Wrote {hist_path}")

    xlsx_path = out_dir / "tile_selection_statistics.xlsx"
    write_statistics_excel(tiles_gdf, hop0, eligible_df, selected, xlsx_path)
    print(f"Wrote {xlsx_path}")


if __name__ == "__main__":
    main()
