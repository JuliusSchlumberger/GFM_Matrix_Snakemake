"""Per-country total flooded area + total exposed population, per (RP, SLR) - raw
model output, no FLOPROS protection or adaptation applied.

Reuses the exact same precomputed per-chunk artifacts and chunk-streaming approach
as compute_exposure_analysis.py (pre-computed flood-fraction/population/geogunit
chunk rasters at ~1 km population-grid resolution, no global mosaic - see that
script's own docstring for why), but computes a much simpler quantity: this is
NOT an EAI (no return-period integration, no protection standards, no adaptation
scenario) - just "how much area does the model flood, and how many people does it
expose, at this exact modelled (RP, SLR) combo, country-wide" at the single fixed
exposure threshold (exposure.exceedance_threshold_m) the flood-fraction chunks are
already built at.

Why this exists: flood_extent_validation compares the model against national
hazard-map benchmarks, but can only ever read a benchmark's own local working
window (see docs/flood_extent_validation_caveats.md §1.2) - it has no view of how
much the model floods across the WHOLE country. Rather than re-deriving that
country-wide total from raw depth inside validate_country.py (which needed its own
bbox-vs-real-country-boundary correctness fix and re-reads large chunks on every
validation run), it is computed HERE, once, from data this pipeline already
produces for the exposure analysis, and read as a plain CSV lookup by
validate_country.py.

Per-country aggregation uses the exact same WRI geogunit-107 raster + FLOPROS ISO
lookup as compute_exposure_analysis.py's `iso_lookup`/`country_sums` - the
established, exact (not bbox-approximated) way this pipeline already knows which
country a model cell belongs to.

Writes one CSV per country: {outdir}/flood_totals/flood_totals_{ISO}.csv, columns
iso, return_period, waterlevel_name, total_wet_km2, total_exposed_pop - one row per
modelled (RP, SLR) combo that has a real flood-fraction chunk for at least one of
this country's cells.

Usage:
    python compute_flood_totals.py \\
        [--config snakemake_workflow/config/config.yml] \\
        --outdir D:/GFM/merged_results/exposure
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from plotting import pixel_area_km2_grid  # noqa: E402
from exposure_analysis import _build_iso_index, _safe, country_sums  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_band(path: Path, dtype: str, fill) -> np.ndarray:
    """Read band 1, replacing the file's own nodata with `fill` - same helper as
    compute_exposure_analysis.py's own _read_band (see that docstring for why a
    plain `src.read(1)` is not enough - population's nodata sentinel is finite)."""
    with retry_transient_io(rasterio.open, path) as src:
        arr = src.read(1, masked=True)
        return arr.filled(fill).astype(dtype)


def _read_band_and_transform(path: Path, dtype: str, fill) -> tuple[np.ndarray, Affine]:
    """Same as _read_band, plus the file's own transform - one rasterio.open
    instead of two. Only the population chunk needs its transform (for
    pixel_area_km2_grid); geogunit/flood-fraction chunks are always read on
    that same grid (see analysis/compute_exposure_analysis.py's own chunk
    convention), so re-opening them just for a transform would be redundant."""
    with retry_transient_io(rasterio.open, path) as src:
        arr = src.read(1, masked=True)
        return arr.filled(fill).astype(dtype), src.transform


def _round_val(val: float) -> float | int:
    """Same magnitude-conditional rounding rule as validate_country.py's _round_row -
    whole integer, except abs(val) < 1 rounds to 2 decimals so a small real
    nonzero value doesn't silently read as "none"."""
    if val != val:  # NaN
        return val
    return round(val, 2) if abs(val) < 1 else int(round(val))


def main() -> None:
    _default_cfg = str(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bc = cfg["boundary_conditions"]
    return_periods = bc["return_periods"]
    slr_scenarios = bc["slr_scenarios"]

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    flood_frac_dir = merged_dir / "chunks" / "flood_fraction"
    chunks_dir = merged_dir / "chunks"
    out_dir = Path(args.outdir) / "flood_totals"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    all_chunk_ids = sorted(set(
        p.stem.split("_RP")[0].replace("flood_fraction_", "")
        for p in flood_frac_dir.glob("flood_fraction_*.tif")
    ))
    if not all_chunk_ids:
        print(f"ERROR: no flood fraction files in {flood_frac_dir}")
        sys.exit(1)

    chunk_ids = [
        cid for cid in all_chunk_ids
        if (chunks_dir / f"exposure_population_grid_{cid}.tif").exists()
        and (chunks_dir / f"exposure_population_grid_{cid}.tif").stat().st_size > 0
    ]
    if not chunk_ids:
        print("ERROR: no population chunk files found — run the Snakemake postprocess target first.")
        sys.exit(1)
    print(f"Using {len(chunk_ids)} populated chunks, {len(return_periods)} RPs, {len(slr_scenarios)} SLR scenarios.")

    print("Loading FLOPROS ISO lookup…")
    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")  # catalog key (data_catalog_gfm.yml)
    iso_lookup = {
        int(gid): str(row["ISO"]) for gid, row in flopros.iterrows()
        if pd.notna(row.get("ISO"))
    }

    # totals[iso][(rp, slr)] = {"total_wet_km2": ..., "total_exposed_pop": ...}
    totals: dict[str, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)

    for i, cid in enumerate(chunk_ids, 1):
        pop_path = chunks_dir / f"exposure_population_grid_{cid}.tif"
        pop, transform = _read_band_and_transform(pop_path, "float64", 0.0)
        pop[~np.isfinite(pop)] = 0.0
        geo = _read_band(chunks_dir / f"exposure_geogunit_grid_{cid}.tif", "int32", -1)
        geo[geo < 0] = -1
        iso_index = _build_iso_index(geo, iso_lookup)
        if not iso_index[0]:
            continue  # no resolvable country anywhere in this chunk

        area_km2 = pixel_area_km2_grid(transform, pop.shape[1], pop.shape[0])

        for rp in return_periods:
            for slr in slr_scenarios:
                p = flood_frac_dir / f"flood_fraction_{cid}_RP{rp}_{slr}.tif"
                if not (p.exists() and p.stat().st_size > 0):
                    continue
                ff = _safe(_read_band(p, "float64", -1.0))
                exposed_pop, flooded_area = country_sums(ff * pop, ff * area_km2, geo, iso_lookup, iso_index)
                for iso in exposed_pop:
                    key = (int(rp), slr)
                    cur = totals[iso].setdefault(key, {"total_exposed_pop": 0.0, "total_wet_km2": 0.0})
                    cur["total_exposed_pop"] += exposed_pop[iso]
                    cur["total_wet_km2"] += flooded_area[iso]

        if i % 100 == 0 or i == len(chunk_ids):
            print(f"  {i}/{len(chunk_ids)} chunks processed…")

    for iso, by_key in sorted(totals.items()):
        rows = [
            {
                "iso": iso, "return_period": rp, "waterlevel_name": slr,
                "total_wet_km2": _round_val(v["total_wet_km2"]),
                "total_exposed_pop": _round_val(v["total_exposed_pop"]),
            }
            for (rp, slr), v in sorted(by_key.items())
        ]
        df = pd.DataFrame(rows)
        out_path = out_dir / f"flood_totals_{iso}.csv"
        atomic_write(out_path, lambda f: df.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")

    print(f"\nWrote flood totals for {len(totals)} countries to {out_dir}")


if __name__ == "__main__":
    main()
