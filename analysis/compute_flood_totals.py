"""Compute per-country flooded area and exposed population per (RP, SLR).

Raw model output: no RP integration, FLOPROS protection or adaptation; at the single
exposure.exceedance_threshold_m the flood-fraction chunks (~1 km population grid) are
built at. Countries are assigned via the WRI geogunit-107 raster + FLOPROS ISO lookup,
as in compute_exposure_analysis.py. validate_country.py reads these CSVs for country-wide
totals instead of re-deriving them from raw depth.

Writes {outdir}/flood_totals/flood_totals_{ISO}.csv.

Usage:
    python compute_flood_totals.py \\
        [--config snakemake_workflow/config/config.yml] \\
        --out-dir D:/GFM/merged_results/exposure
"""

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _read_band, _read_band_and_transform, _round_val, discover_chunk_ids  # noqa: E402
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from exposure_analysis import _build_iso_index, _safe, country_sums  # noqa: E402
from validation import load_iso_lookup  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _process_chunk(
    cid: str, return_periods: list[int], slr_scenarios: list[str],
    flood_frac_dir: Path, chunks_dir: Path, iso_lookup: dict[int, str],
) -> dict[str, dict[tuple[int, str], dict[str, float]]]:
    chunk_totals: dict[str, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)

    pop_path = chunks_dir / f"exposure_population_grid_{cid}.tif"
    pop, transform = _read_band_and_transform(pop_path, "float64", 0.0)
    pop[~np.isfinite(pop)] = 0.0
    geo = _read_band(chunks_dir / f"exposure_geogunit_grid_{cid}.tif", "int32", -1)
    geo[geo < 0] = -1
    iso_index = _build_iso_index(geo, iso_lookup)
    if not iso_index[0]:
        return chunk_totals

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
                cur = chunk_totals[iso].setdefault(key, {"total_exposed_pop": 0.0, "total_wet_km2": 0.0})
                cur["total_exposed_pop"] += exposed_pop[iso]
                cur["total_wet_km2"] += flooded_area[iso]

    return chunk_totals


def main() -> None:
    """Write per-country flood totals; exit 1 if any chunk failed (CSVs exclude it)."""
    _default_cfg = str(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--out-dir", "--outdir", dest="outdir", required=True)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="chunks processed concurrently via a thread pool (default 1 = serial). Each chunk's "
             "own work is ~45 small, independent rasterio reads over the network share - I/O-wait "
             "bound, not CPU-bound (same reasoning run_simulation_batch.py already uses for its own "
             "thread pool) - so this is a genuine speedup, not oversubscription, up to the real "
             "cpu/network-connection budget available.",
    )
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

    chunk_ids, all_chunk_ids = discover_chunk_ids(flood_frac_dir, chunks_dir)
    if not all_chunk_ids:
        print(f"ERROR: no flood fraction files in {flood_frac_dir}")
        sys.exit(1)
    if not chunk_ids:
        print("ERROR: no population chunk files found — run the Snakemake postprocess target first.")
        sys.exit(1)
    print(f"Using {len(chunk_ids)} populated chunks, {len(return_periods)} RPs, {len(slr_scenarios)} SLR scenarios.")

    print("Loading FLOPROS ISO lookup…")
    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    iso_lookup = load_iso_lookup(catalog, "flopros_protection_standards")

    totals: dict[str, dict[tuple[int, str], dict[str, float]]] = defaultdict(dict)

    print(f"Processing {len(chunk_ids)} chunks with {args.workers} worker(s)…")
    t0 = time.time()
    failed_chunks: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_chunk, cid, return_periods, slr_scenarios, flood_frac_dir, chunks_dir, iso_lookup): cid
            for cid in chunk_ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            cid = futures[fut]
            try:
                chunk_totals = fut.result()
            except Exception as exc:
                # A persistent P:\ failure must not discard hours of completed chunks.
                print(f"  [FAILED] chunk {cid}: {type(exc).__name__}: {exc}", flush=True)
                failed_chunks.append(cid)
                continue

            for iso, by_key in chunk_totals.items():
                for key, v in by_key.items():
                    cur = totals[iso].setdefault(key, {"total_exposed_pop": 0.0, "total_wet_km2": 0.0})
                    cur["total_exposed_pop"] += v["total_exposed_pop"]
                    cur["total_wet_km2"] += v["total_wet_km2"]

            if i % 50 == 0 or i == len(chunk_ids):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                eta_min = (len(chunk_ids) - i) / rate / 60 if rate else float("nan")
                print(f"  {i}/{len(chunk_ids)} chunks processed… ({elapsed:.0f}s elapsed, "
                      f"{rate:.1f} chunks/s, ETA {eta_min:.1f} min)")

    if failed_chunks:
        print(f"\nWARNING: {len(failed_chunks)}/{len(chunk_ids)} chunk(s) failed after retries "
              f"(totals below EXCLUDE them - re-run and compare, or investigate): {failed_chunks}")

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
    if failed_chunks:
        sys.exit(1)


if __name__ == "__main__":
    main()
