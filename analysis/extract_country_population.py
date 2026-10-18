"""Extract total 2020 population per country from the full WorldPop raster.

Unlike the chunk-clipped exposure population grids, this covers the full global
extent, so it serves as the country denominator for "% of population exposed".
Pixels are attributed via the WRI geogunit_109 raster (nearest-neighbour onto the
population grid) and the geo109->ISO lookup cached with plot_world_map.py.

Usage:
    python analysis/extract_country_population.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--out D:/GFM/processed_inputs/country_population_2020.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hydromt.exceptions import NoDataException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from protection import load_geogunit_ids  # noqa: E402
from analysis_figures import build_geo109_to_iso_lookup  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]

# WorldPop coverage (lat -72..84) plus a small buffer.
_GLOBAL_LAT_RANGE = (-75.0, 85.0)


def _geo109_iso_lookup(catalog, cfg: dict, viz: dict) -> dict[int, str]:
    """Load the geo109->ISO lookup from plot_world_map.py's cache, else build and cache it."""
    out_dir = Path(viz["output_dir"]) / "world_maps"
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
        geo109_nc, geo107_nc, flopros_xlsx, subsample=int(viz["geo109_subsample"]),
    )
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    pd.DataFrame({"ISO": lookup}).to_csv(cache_path)
    return lookup


def sum_population_by_iso(
    catalog, geo109_to_iso: dict[int, str], row_band_deg: float = 5.0,
) -> tuple[dict[str, float], int, float]:
    """Zonal-sum the full population raster per geogunit_109 ID, then per ISO.

    Read in `row_band_deg` latitude bands to bound memory; every pixel is summed (no
    subsampling). Returns (pop_by_iso, n_dropped_ids, dropped_population), where dropped
    IDs have no ISO mapping (ocean/unassigned).
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
        except NoDataException:
            # Band outside WorldPop coverage; any other read error must abort, since a
            # skipped band silently lowers the country totals.
            pop_da = None
        if pop_da is None or pop_da.size == 0:
            lat = lat_top
            continue

        pop = pop_da.values.astype("float64")
        nodata = pop_da.raster.nodata
        if nodata is not None:
            pop[pop == nodata] = 0.0
        pop[~np.isfinite(pop)] = 0.0

        # Nearest-neighbour: geogunit IDs are categorical.
        geo_ids = load_geogunit_ids(catalog, "geogunit_country_units", pop_da)

        ids_flat = geo_ids.ravel().astype("int64")
        pop_flat = pop.ravel()
        # ID 0 = no country; -1 = GEOGUNIT_INVALID.
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
        "--out", default=None,
        help="output CSV path (default: visualization.country_population_csv from config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg["visualization"]

    out_path = Path(args.out or viz["country_population_csv"])
    row_band_deg = float(viz["population_extract_row_band_deg"])

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
