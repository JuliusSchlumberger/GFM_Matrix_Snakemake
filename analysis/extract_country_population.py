"""Extract total 2020 population per country from the WorldPop raster.

Sums the FULL, unclipped WorldPop population raster (`population` catalog
source) per country, using the WRI geogunit_109 country-unit raster
(`geogunit_country_units`) for pixel attribution - resolved the same way
prepare_exposure_grid_chunk()/protection.load_geogunit_ids() already resolve
geogunit IDs everywhere else in this pipeline (nearest-neighbour
reproject_like onto the population grid, since geogunit IDs are categorical
region codes, not a continuous field - grid misalignment between the
WorldPop and WRI products is handled automatically, not assumed away).

Unlike the exposure pipeline's own exposure_population_grid_{chunk_id}.tif
(prepare_exposure_grid_chunk() in src/exposure.py), which is clipped to each
chunk's flood-simulation bbox (coastal/floodable buffer only - see
src/tile_chunking.py's module docstring), this reads the population raster's
FULL global extent in latitude bands, so it can serve as a true
country-population denominator for "% of population exposed" figures
(analysis/plot_burning_ember.py, analysis/plot_timeseries.py) - not just
"% of the modeled coastal buffer".

Country attribution reuses visualization.build_geo109_to_iso_lookup - the
same geogunit_109 ID -> ISO-3 lookup plot_world_map.py already builds for
its choropleths - and shares its cache file so whichever script runs first
builds it once.

Output CSV columns: ISO, total_population_2020.

Usage:
    python snakemake_workflow/analysis/extract_country_population.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--output D:/GFM/processed_inputs/country_population_2020.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from protection import load_geogunit_ids  # noqa: E402
from visualization import build_geo109_to_iso_lookup  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# WorldPop's own documented coverage (data_catalog_gfm.yml: "lat -72..84"),
# with a small buffer either side.
_GLOBAL_LAT_RANGE = (-75.0, 85.0)


def _geo109_iso_lookup(catalog, cfg: dict, viz: dict) -> dict[int, str]:
    """Reuse plot_world_map.py's cached geo109->ISO lookup if present, else build+cache it there.

    Same cache path plot_world_map.py itself writes to (out_dir /
    "geo109_iso_lookup.csv" under its --outdir, default {output_dir}/world_maps)
    - whichever of the two scripts runs first builds it, the other reuses it.
    """
    out_dir = Path(viz.get("output_dir", f"{cfg['paths']['root']}/figures")) / "world_maps"
    cache_path = out_dir / "geo109_iso_lookup.csv"
    if cache_path.exists():
        print(f"  Reusing cached geo109->ISO lookup: {cache_path}")
        cache_df = pd.read_csv(cache_path, index_col=0)
        return dict(zip(cache_df.index.astype(int), cache_df["ISO"].astype(str)))

    print("  Building geogunit_109->ISO lookup (one-time, shared with plot_world_map.py)...")
    geo109_nc = Path(catalog.get_source("geogunit_country_units").path)
    geo107_nc = Path(catalog.get_source("geogunit_protection_units").path)
    flopros_xlsx = Path(catalog.get_source("flopros_protection_standards").path)
    lookup = build_geo109_to_iso_lookup(
        geo109_nc, geo107_nc, flopros_xlsx, subsample=int(viz.get("geo109_subsample", 10)),
    )
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    pd.DataFrame({"ISO": lookup}).to_csv(cache_path)
    return lookup


def sum_population_by_iso(
    catalog, geo109_to_iso: dict[int, str], row_band_deg: float = 5.0,
) -> tuple[dict[str, float], int, float]:
    """Zonal-sum the FULL population raster per geogunit_109 ID, then per ISO.

    Reads the population raster in latitude bands (row_band_deg wide) via
    the data catalog, bounding memory on the full ~1km global raster - every
    pixel's population is summed exactly (never spatially subsampled, unlike
    the coarser geo109->ISO ID lookup itself, where subsampling only affects
    a majority vote over already country-sized units).

    Returns (pop_by_iso, n_dropped_ids, dropped_population) - dropped =
    geo109 IDs present in the raster with no ISO mapping (ocean/unassigned).
    """
    lat_min, lat_max = _GLOBAL_LAT_RANGE
    totals: dict[int, float] = {}
    lat = lat_min
    while lat < lat_max:
        lat_top = min(lat + row_band_deg, lat_max)
        bbox = [-180.0, lat, 180.0, lat_top]
        try:
            pop_da = retry_transient_io(
                catalog.get_rasterdataset, "population", bbox=bbox,
            ).squeeze(drop=True)
        except Exception:
            pop_da = None
        if pop_da is None or pop_da.size == 0:
            lat = lat_top
            continue

        pop = pop_da.values.astype("float64")
        nodata = pop_da.raster.nodata
        if nodata is not None:
            pop[pop == nodata] = 0.0
        pop[~np.isfinite(pop)] = 0.0

        # Nearest-neighbour reproject onto pop_da's own grid - same helper
        # (and same alignment-safety guarantee) prepare_exposure_grid_chunk
        # already relies on for every chunk's own geogunit attribution.
        geo_ids = load_geogunit_ids(catalog, "geogunit_country_units", pop_da)

        ids_flat = geo_ids.ravel().astype("int64")
        pop_flat = pop.ravel()
        # geogunit ID 0 = no country (same convention as
        # visualization.build_geo109_to_iso_lookup's own "> 0" mask);
        # load_geogunit_ids' own GEOGUNIT_INVALID sentinel (-1) is excluded too.
        valid = ids_flat > 0
        if np.any(valid):
            band_sums = np.bincount(ids_flat[valid], weights=pop_flat[valid])
            for gid in np.nonzero(band_sums)[0]:
                totals[int(gid)] = totals.get(int(gid), 0.0) + float(band_sums[gid])

        print(f"  lat [{lat:.1f}, {lat_top:.1f}): {pop.sum():,.0f} people")
        lat = lat_top

    pop_by_iso: dict[str, float] = {}
    n_dropped = 0
    dropped_pop = 0.0
    for gid, total in totals.items():
        iso = geo109_to_iso.get(gid)
        if iso is None:
            n_dropped += 1
            dropped_pop += total
            continue
        pop_by_iso[iso] = pop_by_iso.get(iso, 0.0) + total

    return pop_by_iso, n_dropped, dropped_pop


def main() -> None:
    _default_cfg = str(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument(
        "--output", default=None,
        help="output CSV path (default: visualization.country_population_csv from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg.get("visualization", {})

    out_path = Path(args.output or viz.get(
        "country_population_csv", f"{cfg['paths']['root']}/processed_inputs/country_population_2020.csv",
    ))
    row_band_deg = float(viz.get("population_extract_row_band_deg", 5.0))

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    geo109_to_iso = _geo109_iso_lookup(catalog, cfg, viz)

    print("Summing WorldPop population by country (reads the full global raster in latitude bands)...")
    pop_by_iso, n_dropped, dropped_pop = sum_population_by_iso(catalog, geo109_to_iso, row_band_deg)

    if not pop_by_iso:
        print("ERROR: no population summed - check the 'population'/'geogunit_country_units' catalog entries.")
        sys.exit(1)

    result = pd.Series(pop_by_iso, name="total_population_2020").sort_index()
    result.index.name = "ISO"

    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    atomic_write(out_path, lambda f: result.reset_index().to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
    print(f"\nWritten: {out_path} ({len(result)} countries)")

    print(f"\nGlobal total population: {result.sum():,.0f}")
    print(f"Dropped {n_dropped} geo109 ID(s) with no ISO mapping ({dropped_pop:,.0f} people).")
    print("\nTop 10 countries by population:")
    print(result.sort_values(ascending=False).head(10).round(0).to_string())


if __name__ == "__main__":
    main()
