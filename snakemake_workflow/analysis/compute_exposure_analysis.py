"""Run all four flood-exposure scenarios (baseline, protect, retreat, avoid) globally.

Reads coarse flood-fraction rasters produced by compute_flood_fraction_chunk and the
pre-computed per-chunk population and geogunit rasters, and runs exposure analysis at
~1 km population-grid resolution.

Every chunk is read directly, one at a time ("chunk streaming") - there is no global
mosaic of any kind. A single (RP, SLR) flood-fraction grid at global scale is roughly
750M pixels (~6 GB at float64); holding all of them (10 RPs x 10 SLRs = 100 grids) in
memory at once, as an earlier version of this script did via rasterio.merge, requires
hundreds of GB and cannot complete at all beyond a small regional test domain. Chunk
streaming keeps peak memory to about one chunk's worth of arrays (~1-2 MB) regardless of
how large the study area or how many scenarios are configured - see pass1_shares/
_stream_eai below and snakemake_workflow/memory.md for the derivation.

Baseline is treated as exactly "protect" with its design threshold calibrated at SLR_0
(protection.baseline_waterlevel_name) instead of an adaptation.slr_intensities entry - see
exposure_analysis.protect_exposure_grid for why. It therefore has its own base EAI curve
and File 1/2/3 outputs, same as protect/retreat.

Three CSVs per scenario (baseline, protect_{slr_int}, retreat_{slr_int}):
  File 1 - exposure_{label}_base.csv
      Per-country EAI at each discrete MODELLED SLR scenario, 2020 population,
      no growth applied. The basis the other two files derive from.
  File 2 - exposure_{label}_growth_matrix.csv
      File 1 linearly interpolated onto the dense visualization.slr_interp
      grid, then scaled by the scenario-neutral visualization.growth_rates
      axis (same factor for every ISO) - feeds plot_burning_ember.py's
      background heatmap. Exact for baseline/protect/retreat (exposure is
      linear in a uniform population scaling).
  File 3 - exposure_{label}_ssp.csv
      One column per (SSP, year): File 1 linearly interpolated to that SSP's
      real projected SLR at that year (from visualization.slr_trajectories_csv,
      itself linearly interpolated between modelled trajectory years), scaled
      by that country's real SSP/year population growth factor - feeds
      plot_timeseries.py, already fully resolved (no further interpolation
      needed downstream).

Avoid has no growth-free base curve (its "redirected growth" term is
inherently growth-dependent), so it only gets File 2 and File 3, each
genuinely recomputed per growth-rate/SSP-year rather than derived from a
shared File 1 - see _avoid_growth_matrix_worker_task / _avoid_ssp_worker_task.

Usage:
    python compute_exposure_analysis.py \\
        [--config snakemake_workflow/config/config.yml] \\
        --outdir D:/GFM/merged_results/exposure
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, merged_slr_scenarios  # noqa: E402
from population_growth import load_ssp_growth_factors, interpolate_growth_factor  # noqa: E402
from visualization import load_slr_trajectories  # noqa: E402
from exposure_analysis import (                                 # noqa: E402
    build_adapt_protection_fraction,
    protect_exposure_grid,
    compute_retreat,
    compute_avoid,
    compute_country_eai,
    interpolate_eai_linear,
    resolve_ssp_scenario_eai,
    apply_growth_rates_to_eai,
    country_sums,
    apply_country_shares,
    scatter_country_values,
    _build_iso_index,
    _safe,
)


def _expand(s: str, root: str, code_root: str = "") -> str:
    return str(s).replace("{root}", root).replace("{code_root}", code_root or root)


def _slr_mm(slr_name: str) -> int:
    """Parse SLR scenario name to millimetres (e.g. 'SLR_250' → 250)."""
    return int(slr_name.split("_")[1])


# ── Chunk streaming ────────────────────────────────────────────────────────────
# Every chunk is read directly from its own small (~600x600 px) GeoTIFF - no
# global mosaic. This is what makes peak memory independent of study-area size
# and (RP, SLR) scenario count: at any moment, at most one chunk's arrays are
# resident, whether processing 13 chunks or 1400+.

@dataclass
class ChunkData:
    cid: str
    pop: np.ndarray
    geo: np.ndarray
    ff: dict[tuple[int, str], np.ndarray]
    iso_index: tuple[list[str], np.ndarray, np.ndarray]


def _read_band(path: Path, dtype: str, fill) -> np.ndarray:
    """Read band 1, replacing the file's OWN nodata value (whatever it is -
    population chunks use float32-min, geogunit uses -1, flood_fraction uses
    -1.0, all different) with `fill`.

    A plain `src.read(1)` does NOT do this - rasterio.merge (used by the old
    global-mosaic version this replaces) applies each source's nodata masking
    internally, which is easy to silently lose when reading files directly.
    Confirmed by exact-reproduction testing: population's nodata sentinel
    (-3.4e38, np.finfo(np.float32).min) is a FINITE value, so a naive
    `~np.isfinite(arr)` cleanup step (which is all that's needed after
    rio_merge already replaced it) does not catch it on a raw read and it
    corrupts every downstream computation for any chunk containing it.
    """
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        return arr.filled(fill).astype(dtype)


def _load_chunk(
    cid: str,
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    iso_lookup: dict[int, str],
) -> ChunkData:
    """Load one chunk's population, geogunit, and available flood-fraction rasters.

    Replaces the old global mosaic (_mosaic/_load_flood_fractions). Each
    raster's nodata is replaced with the same fill value the old mosaic-based
    loading used (population: 0.0; geogunit: GEOGUNIT_INVALID/-1;
    flood_fraction: -1.0, then _safe() as before).
    """
    pop = _read_band(chunks_dir / f"exposure_population_grid_{cid}.tif", "float64", 0.0)
    pop[~np.isfinite(pop)] = 0.0  # defensive: catch any genuine NaN too

    geo = _read_band(chunks_dir / f"exposure_geogunit_grid_{cid}.tif", "int32", -1)
    geo[geo < 0] = -1

    ff: dict[tuple[int, str], np.ndarray] = {}
    for rp in return_periods:
        for slr in slr_scenarios:
            p = flood_frac_dir / f"flood_fraction_{cid}_RP{rp}_{slr}.tif"
            if p.exists() and p.stat().st_size > 0:
                ff[(rp, slr)] = _safe(_read_band(p, "float64", -1.0))

    return ChunkData(cid=cid, pop=pop, geo=geo, ff=ff, iso_index=_build_iso_index(geo, iso_lookup))


def pass1_shares(
    chunk_ids: list[str],
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    slr_int: str,
    rp_applied: dict[int, int],
    iso_lookup: dict[int, str],
) -> dict[str, float]:
    """Stream every chunk once to compute retreat's per-country redistribution share
    for one adaptation slr_intensity: `share[iso] = Σ(apf·population) / Σ(1-apf)`.

    `retreating = apf*population` (no growth) is identical for retreat and the
    growth-independent part of avoid — since a growth factor is always a
    per-country scalar, `avoiding_sum[iso] = retreating_sum[iso] * max(0, g-1)`
    (see compute_avoid_redirected's docstring). So this single streaming pass,
    run once per slr_intensity, is reused for BOTH retreat and every avoid task
    at that intensity — avoid tasks only need a cheap per-country scalar
    multiply on top of this result, not their own chunk-streaming pass.
    """
    amt_total: dict[str, float] = {}
    cap_total: dict[str, float] = {}
    for cid in chunk_ids:
        c = _load_chunk(cid, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup)
        apf = build_adapt_protection_fraction(c.ff, return_periods, slr_int, rp_applied, c.geo)
        retreating = apf * c.pop
        cap = np.maximum(0.0, 1.0 - apf)
        amt, cap_d = country_sums(retreating, cap, c.geo, iso_lookup, c.iso_index)
        for iso, v in amt.items():
            amt_total[iso] = amt_total.get(iso, 0.0) + v
        for iso, v in cap_d.items():
            cap_total[iso] = cap_total.get(iso, 0.0) + v
    return {iso: amt_total[iso] / cap_total[iso] for iso in cap_total if cap_total[iso] > 0.0}


def _stream_eai(
    chunk_ids: list[str],
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    iso_lookup: dict[int, str],
    exposure_fn,
) -> pd.DataFrame:
    """Stream every chunk once, compute `exposure_fn(chunk, ff)` for every (RP, SLR)
    pair, aggregate to per-country EAI, and sum across chunks.

    Exact, not an approximation: `_trapezoid_eai` is linear (fixed integration
    weights, independent of the exposure values), so
    `Σ_chunks trapezoid(chunk_grid) == trapezoid(Σ_chunks chunk_grid)` - summing
    per-chunk `compute_country_eai` DataFrames reproduces the old
    one-shot-on-a-global-mosaic result exactly. compute_country_eai/
    _trapezoid_eai themselves are unchanged.
    """
    total = pd.DataFrame()
    for cid in chunk_ids:
        c = _load_chunk(cid, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup)
        grids = {key: exposure_fn(c, ff) for key, ff in c.ff.items()}
        chunk_eai = compute_country_eai(
            grids, return_periods, slr_scenarios, c.geo, iso_lookup, iso_index=c.iso_index,
        )
        total = total.add(chunk_eai, fill_value=0.0)
    return total


def _scenario_done(label: str, out_dir: Path, expect_ssp: bool) -> bool:
    """Whether a scenario's File 1/2(/3) already exist in out_dir."""
    required = [out_dir / f"exposure_{label}_base.csv", out_dir / f"exposure_{label}_growth_matrix.csv"]
    if expect_ssp:
        required.append(out_dir / f"exposure_{label}_ssp.csv")
    return all(p.exists() for p in required)


def _avoid_done(slr_int: str, out_dir: Path, expect_ssp: bool) -> bool:
    """Whether an avoid slr_intensity's File 2(/3) already exist in out_dir."""
    required = [out_dir / f"exposure_avoid_{slr_int}_growth_matrix.csv"]
    if expect_ssp:
        required.append(out_dir / f"exposure_avoid_{slr_int}_ssp.csv")
    return all(p.exists() for p in required)


def _write_base_scenario_files(
    label: str, eai: pd.DataFrame, out_dir: Path,
    slr_mm_sorted: list[float], slr_interp_mm: np.ndarray, growth_rates: np.ndarray,
    growth_df, ssps: list[str], output_years: list[int], slr_traj,
) -> None:
    """Write File 1 (base), File 2 (growth matrix), File 3 (ssp) for one scenario."""
    eai.to_csv(out_dir / f"exposure_{label}_base.csv")
    print(f"  Written: exposure_{label}_base.csv")

    dense = interpolate_eai_linear(eai, slr_mm_sorted, slr_interp_mm)
    apply_growth_rates_to_eai(dense, growth_rates).to_csv(
        out_dir / f"exposure_{label}_growth_matrix.csv"
    )
    print(f"  Written: exposure_{label}_growth_matrix.csv")

    if growth_df is not None and slr_traj is not None:
        resolve_ssp_scenario_eai(
            eai, slr_mm_sorted, growth_df, ssps, output_years, slr_traj
        ).to_csv(out_dir / f"exposure_{label}_ssp.csv")
        print(f"  Written: exposure_{label}_ssp.csv")
    else:
        print(f"  Skipping exposure_{label}_ssp.csv: no SSP growth factors or SLR trajectory available.")


# ── AVOID worker process ──────────────────────────────────────────────────────
# Every (slr_intensity, SSP, year) / (slr_intensity, growth_rate) combination is
# an independent unit of work, parallelized across worker processes. Unlike the
# old mosaic-based version, the shared payload sent to each worker via the pool
# initializer is now tiny (chunk file paths + small dicts, not multi-GB arrays)
# - each worker reads chunk files itself as it streams through them, exactly
# like the main process does for baseline/protect/retreat.

_AVOID_CTX: dict = {}


def _init_avoid_worker(
    chunk_ids: list[str],
    chunks_dir: Path,
    flood_frac_dir: Path,
    iso_lookup: dict[int, str],
    rp_applied: dict[int, int],
    return_periods: list[int],
    slr_scenarios: list[str],
    growth_df: "pd.DataFrame | None",
    slr_mm_sorted: list[float],
    slr_interp_mm: np.ndarray,
    slr_traj: "pd.DataFrame | None",
) -> None:
    """ProcessPoolExecutor initializer: stash the shared (now tiny) read-only inputs."""
    _AVOID_CTX.update(
        chunk_ids=chunk_ids,
        chunks_dir=chunks_dir,
        flood_frac_dir=flood_frac_dir,
        iso_lookup=iso_lookup,
        rp_applied=rp_applied,
        return_periods=return_periods,
        slr_scenarios=slr_scenarios,
        growth_df=growth_df,
        slr_mm_sorted=slr_mm_sorted,
        slr_interp_mm=slr_interp_mm,
        slr_traj=slr_traj,
    )


def _avoid_exposure_fn(
    chunk: ChunkData, ff: np.ndarray, apf: np.ndarray,
    growth_by_iso: dict[str, float], redirected_share: dict[str, float],
) -> np.ndarray:
    """Shared per-chunk avoid exposure computation for one (RP, SLR) grid.

    `growth_by_iso`/`redirected_share` are per-country dicts, scattered onto
    this chunk's small grid via the existing `scatter_country_values`/
    `apply_country_shares` helpers - same per-country-scalar semantics as the
    old global-grid version, just evaluated at chunk scale.
    """
    iso_list, iso_idx, cell_idx = chunk.iso_index
    g = scatter_country_values(growth_by_iso, iso_list, iso_idx, cell_idx, chunk.geo.shape, default=1.0)
    cap = np.maximum(0.0, 1.0 - apf)
    redirected = apply_country_shares(cap, redirected_share, chunk.geo, _AVOID_CTX["iso_lookup"], chunk.iso_index)
    return compute_avoid(ff, apf, chunk.pop, redirected, g)


def _avoid_ssp_worker_task(
    slr_int: str, share_retreat: dict[str, float], ssp: str, yr: int,
) -> tuple[str, str, int, "pd.Series | None"]:
    """Compute one (slr_intensity, SSP, year) AVOID EAI value (File 3)."""
    ctx = _AVOID_CTX
    growth_df = ctx["growth_df"]
    slr_traj = ctx["slr_traj"]

    if slr_traj is None or ssp not in slr_traj.columns:
        return slr_int, ssp, yr, None

    iso_list = sorted(set(ctx["iso_lookup"].values()))
    growth_by_iso = {
        iso: (interpolate_growth_factor(growth_df, ssp, iso, yr, default=1.0)
              if growth_df is not None else 1.0)
        for iso in iso_list
    }
    share_avoid = {iso: share_retreat[iso] * max(0.0, growth_by_iso.get(iso, 1.0) - 1.0) for iso in share_retreat}

    def exposure_fn(c: ChunkData, ff: np.ndarray) -> np.ndarray:
        apf = build_adapt_protection_fraction(c.ff, ctx["return_periods"], slr_int, ctx["rp_applied"], c.geo)
        return _avoid_exposure_fn(c, ff, apf, growth_by_iso, share_avoid)

    avoid_eai = _stream_eai(
        ctx["chunk_ids"], ctx["chunks_dir"], ctx["flood_frac_dir"],
        ctx["return_periods"], ctx["slr_scenarios"], ctx["iso_lookup"], exposure_fn,
    )

    traj_years = slr_traj.index.to_numpy(dtype=float)
    traj_mm = slr_traj[ssp].to_numpy(dtype=float)
    slr_at_yr = float(np.interp(float(yr), traj_years, traj_mm))

    x = np.asarray(ctx["slr_mm_sorted"], dtype=float)
    resolved = {
        iso: float(np.clip(np.interp(slr_at_yr, x, avoid_eai.loc[iso].to_numpy(dtype=float)), 0.0, None))
        for iso in avoid_eai.index
    }
    return slr_int, ssp, yr, pd.Series(resolved, name=f"EAI_{ssp}_{yr}")


def _avoid_growth_matrix_worker_task(
    slr_int: str, share_retreat: dict[str, float], g_pct: int,
) -> tuple[str, int, pd.DataFrame]:
    """Compute one (slr_intensity, growth_rate) scenario-neutral AVOID EAI slice (File 2)."""
    ctx = _AVOID_CTX
    growth_factor = 1.0 + g_pct / 100.0
    share_avoid = {iso: v * max(0.0, growth_factor - 1.0) for iso, v in share_retreat.items()}
    growth_by_iso_const = {iso: growth_factor for iso in ctx["iso_lookup"].values()}

    def exposure_fn(c: ChunkData, ff: np.ndarray) -> np.ndarray:
        apf = build_adapt_protection_fraction(c.ff, ctx["return_periods"], slr_int, ctx["rp_applied"], c.geo)
        return _avoid_exposure_fn(c, ff, apf, growth_by_iso_const, share_avoid)

    avoid_eai = _stream_eai(
        ctx["chunk_ids"], ctx["chunks_dir"], ctx["flood_frac_dir"],
        ctx["return_periods"], ctx["slr_scenarios"], ctx["iso_lookup"], exposure_fn,
    )
    dense = interpolate_eai_linear(avoid_eai, ctx["slr_mm_sorted"], ctx["slr_interp_mm"])
    dense.columns = [f"EAI_{c}_g{g_pct}" for c in dense.columns]
    return slr_int, g_pct, dense


def main() -> None:
    _default_cfg = str(Path(__file__).resolve().parents[1] / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a scenario (baseline/protect_*/retreat_*) or an avoid "
             "slr_intensity entirely if its output CSVs already exist in "
             "--outdir, instead of recomputing and overwriting them. Useful "
             "for resuming after a partial failure without redoing already-"
             "correct, expensive work.",
    )
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    root = cfg.get("paths", {}).get("root", "")
    code_root = cfg.get("paths", {}).get("code_root", root)
    ex = lambda s: _expand(s, root, code_root)

    bc = cfg["boundary_conditions"]
    adapt_cfg = cfg.get("adaptation", {})
    pg_cfg = cfg.get("population_growth", {})
    prot_cfg = cfg.get("protection", {})
    viz = cfg.get("visualization", {})

    return_periods = bc["return_periods"]
    slr_scenarios = merged_slr_scenarios(bc, adapt_cfg)
    slr_mm_sorted = [float(_slr_mm(s)) for s in slr_scenarios]
    slr_baseline = prot_cfg.get("baseline_waterlevel_name", "SLR_0")
    slr_intensities = adapt_cfg.get("slr_intensities", [])
    default_rp = float(prot_cfg.get("default_rp", 5.0))
    sorted_rps = sorted(return_periods)

    ssps = pg_cfg.get("ssps", ["SSP1", "SSP2", "SSP3", "SSP5"])
    output_years = [int(y) for y in pg_cfg.get("output_years", [2030, 2050, 2100])]

    # Scenario-neutral growth-rate axis for the burning-ember background
    # (same config the plot's y-axis uses) - -50% to +150% by default, applied
    # uniformly (not per-country like SSP factors) to every ISO/SLR cell.
    gr_cfg = viz.get("growth_rates", {})
    growth_rates = np.arange(
        float(gr_cfg.get("min", -0.5)),
        float(gr_cfg.get("max", 1.5)) + float(gr_cfg.get("step", 0.1)) / 2,
        float(gr_cfg.get("step", 0.1)),
    )

    # Dense SLR grid (mm) File 2 (growth matrix) is interpolated onto -
    # linearly, from the discrete modelled SLR scenarios (sea level, and EAI
    # along with it, is assumed to change linearly between two modelled
    # steps). Same config the burning-ember plot's x-axis used to interpolate
    # onto at plot-render time - now baked into the data file instead.
    si_cfg = viz.get("slr_interp", {})
    slr_interp_mm = np.linspace(
        float(si_cfg.get("min_mm", 0)),
        float(si_cfg.get("max_mm", max(slr_mm_sorted))),
        int(si_cfg.get("n_points", 100)),
    )

    # SSP SLR trajectories (median/p50), needed to resolve File 3's
    # per-(SSP,year) values to a single real SLR at file-creation time.
    traj_path = ex(viz.get("slr_trajectories_csv", ""))
    slr_traj = None
    if traj_path and Path(traj_path).exists():
        traj_full = load_slr_trajectories(traj_path)
        p50_cols = {c: c.replace("_p50", "") for c in traj_full.columns if c.endswith("_p50")}
        slr_traj = traj_full[list(p50_cols)].rename(columns=p50_cols)
    else:
        print(f"  WARNING: SLR trajectories not found at {traj_path}; File 3 (*_ssp.csv) will be skipped.")

    merged_dir = Path(ex(cfg["postprocessing"]["merged_outputs"]))
    flood_frac_dir = merged_dir / "chunks" / "flood_fraction"
    chunks_dir = merged_dir / "chunks"
    out_dir = Path(ex(args.outdir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover chunks and restrict to those with valid population data ─────
    all_chunk_ids = sorted(set(
        p.stem.split("_RP")[0].replace("flood_fraction_", "")
        for p in flood_frac_dir.glob("flood_fraction_*.tif")
    ))
    if not all_chunk_ids:
        print(f"ERROR: no flood fraction files in {flood_frac_dir}")
        sys.exit(1)

    # Only use chunks with non-empty population files. Ocean/unpopulated chunks
    # have zero-byte placeholder files and contribute zero exposure.
    chunk_ids = [
        cid for cid in all_chunk_ids
        if (chunks_dir / f"exposure_population_grid_{cid}.tif").exists()
        and (chunks_dir / f"exposure_population_grid_{cid}.tif").stat().st_size > 0
    ]
    if not chunk_ids:
        print("ERROR: no population chunk files found — run the Snakemake postprocess target first.")
        sys.exit(1)
    skipped = len(all_chunk_ids) - len(chunk_ids)
    if skipped:
        print(f"  Skipping {skipped} unpopulated/ocean chunks (zero-byte population files).")
    print(f"Using {len(chunk_ids)} populated chunks, {len(return_periods)} RPs, {len(slr_scenarios)} SLR scenarios.")
    print("Processing chunk-by-chunk (no global mosaic) - peak memory stays flat regardless of study-area size.")

    # ── FLOPROS: per-geogunit RP_applied and ISO lookup (tiny, geogunit-indexed) ──
    print("Loading FLOPROS protection standards…")
    catalog = get_data_catalog(ex(cfg["paths"]["hydromt_data_catalog"]))
    flopros = catalog.get_dataframe(prot_cfg["flopros_source"])
    coastal_rp = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)

    def _snap_rp(flopros_rp: float) -> int:
        candidates = [r for r in sorted_rps if r >= flopros_rp]
        return int(min(candidates)) if candidates else int(max(sorted_rps))

    rp_applied = {int(gid): _snap_rp(float(rp)) for gid, rp in coastal_rp.items()}
    iso_lookup = {
        int(gid): str(row["ISO"]) for gid, row in flopros.iterrows()
        if pd.notna(row.get("ISO"))
    }

    # ── SSP growth factors ────────────────────────────────────────────────────
    print("Loading SSP growth factors…")
    xlsx_path = Path(ex(pg_cfg.get(
        "factors_xlsx", "{root}/inputs/SSPs/getting_SSP_population_growth_factors.xlsx",
    )))
    growth_df = load_ssp_growth_factors(xlsx_path) if xlsx_path.exists() else None
    if growth_df is None:
        print("  WARNING: SSP growth factors not found; File 3 (*_ssp.csv) will be skipped.")

    expect_ssp = growth_df is not None and slr_traj is not None

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.skip_existing and _scenario_done("baseline", out_dir, expect_ssp):
        print("\nSkipping BASELINE exposure (--skip-existing, files already present).")
    else:
        print("\nComputing BASELINE exposure…")

        def _baseline_fn(c: ChunkData, ff: np.ndarray) -> np.ndarray:
            apf = build_adapt_protection_fraction(c.ff, return_periods, slr_baseline, rp_applied, c.geo)
            return protect_exposure_grid(ff, apf, c.pop)

        baseline_eai = _stream_eai(
            chunk_ids, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup, _baseline_fn,
        )
        _write_base_scenario_files(
            "baseline", baseline_eai, out_dir, slr_mm_sorted, slr_interp_mm, growth_rates,
            growth_df, ssps, output_years, slr_traj,
        )

    # ── Protect ───────────────────────────────────────────────────────────────
    for slr_int in slr_intensities:
        label = f"protect_{slr_int}"
        if args.skip_existing and _scenario_done(label, out_dir, expect_ssp):
            print(f"\nSkipping PROTECT exposure ({slr_int}) (--skip-existing, files already present).")
            continue
        print(f"\nComputing PROTECT exposure ({slr_int})…")

        def _protect_fn(c: ChunkData, ff: np.ndarray, slr_int=slr_int) -> np.ndarray:
            apf = build_adapt_protection_fraction(c.ff, return_periods, slr_int, rp_applied, c.geo)
            return protect_exposure_grid(ff, apf, c.pop)

        protect_eai = _stream_eai(
            chunk_ids, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup, _protect_fn,
        )
        _write_base_scenario_files(
            label, protect_eai, out_dir, slr_mm_sorted, slr_interp_mm, growth_rates,
            growth_df, ssps, output_years, slr_traj,
        )

    # ── Retreat ───────────────────────────────────────────────────────────────
    # share_retreat depends only on this slr_int's design grid, not on (rp, slr)
    # or growth - computed once via pass1_shares and reused across retreat's own
    # pass 2 below AND every avoid task at this intensity (see Avoid section).
    shares_by_intensity: dict[str, dict[str, float]] = {}
    for slr_int in slr_intensities:
        label = f"retreat_{slr_int}"
        print(f"\nComputing pass-1 country shares ({slr_int})…")
        share_retreat = pass1_shares(
            chunk_ids, chunks_dir, flood_frac_dir, return_periods, slr_scenarios,
            slr_int, rp_applied, iso_lookup,
        )
        shares_by_intensity[slr_int] = share_retreat

        if args.skip_existing and _scenario_done(label, out_dir, expect_ssp):
            print(f"  Skipping RETREAT exposure ({slr_int}) (--skip-existing, files already present).")
            continue
        print(f"Computing RETREAT exposure ({slr_int})…")

        def _retreat_fn(c: ChunkData, ff: np.ndarray, slr_int=slr_int, share_retreat=share_retreat) -> np.ndarray:
            apf = build_adapt_protection_fraction(c.ff, return_periods, slr_int, rp_applied, c.geo)
            cap = np.maximum(0.0, 1.0 - apf)
            redistributed = apply_country_shares(cap, share_retreat, c.geo, iso_lookup, c.iso_index)
            eff_pop = c.pop * (1.0 - apf) + redistributed
            return compute_retreat(ff, apf, eff_pop)

        retreat_eai = _stream_eai(
            chunk_ids, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup, _retreat_fn,
        )
        _write_base_scenario_files(
            label, retreat_eai, out_dir, slr_mm_sorted, slr_interp_mm, growth_rates,
            growth_df, ssps, output_years, slr_traj,
        )

    # ── Avoid ─────────────────────────────────────────────────────────────────
    # Every (slr_intensity, SSP, year) / (slr_intensity, growth_rate) combination
    # is independent - parallelized across worker processes. Each worker's
    # payload is now tiny (chunk paths + small dicts, not multi-GB arrays), so
    # the old memory-budget worker-count capping essentially never binds anymore
    # - kept for safety but effectively a no-op at typical avoid_worker_memory_
    # budget_gb settings.
    avoid_slr_intensities = slr_intensities
    if args.skip_existing:
        avoid_slr_intensities = [s for s in slr_intensities if not _avoid_done(s, out_dir, expect_ssp)]
        for s in slr_intensities:
            if s not in avoid_slr_intensities:
                print(f"  Skipping AVOID exposure ({s}) (--skip-existing, files already present).")

    growth_pct_labels = [int(round(g * 100)) for g in growth_rates]
    ssp_tasks = [
        (slr_int, ssp, yr)
        for slr_int in avoid_slr_intensities
        for ssp in ssps
        for yr in output_years
    ] if (growth_df is not None and slr_traj is not None) else []
    growth_tasks = [
        (slr_int, g_pct)
        for slr_int in avoid_slr_intensities
        for g_pct in growth_pct_labels
    ]
    n_tasks = len(ssp_tasks) + len(growth_tasks)
    budget_gb = float(cfg.get("analysis", {}).get("avoid_worker_memory_budget_gb", 8))
    n_workers = max(1, min(os.cpu_count() or 1, max(n_tasks, 1)))
    print(
        f"\nComputing AVOID exposure: {len(avoid_slr_intensities)} slr_intensities x "
        f"{len(ssps)} SSPs x {len(output_years)} years = {len(ssp_tasks)} SSP tasks, plus "
        f"{len(avoid_slr_intensities)} slr_intensities x {len(growth_pct_labels)} growth rates = "
        f"{len(growth_tasks)} scenario-neutral tasks ({n_tasks} total). "
        f"Per-worker payload is now file paths + small dicts (not full grids), "
        f"budget {budget_gb:.1f} GiB no longer constrains worker count -> {n_workers} worker process(es)…"
    )

    ssp_results: dict[str, list[pd.Series]] = {slr_int: [] for slr_int in avoid_slr_intensities}
    growth_results: dict[str, list[pd.DataFrame]] = {slr_int: [] for slr_int in avoid_slr_intensities}
    expected_ssp = {slr_int: sum(1 for s, _, _ in ssp_tasks if s == slr_int) for slr_int in avoid_slr_intensities}
    expected_growth = {slr_int: sum(1 for s, _ in growth_tasks if s == slr_int) for slr_int in avoid_slr_intensities}
    written = {slr_int: False for slr_int in avoid_slr_intensities}

    def _maybe_write_avoid(slr_int: str) -> None:
        if written[slr_int]:
            return
        if len(ssp_results[slr_int]) < expected_ssp[slr_int]:
            return
        if len(growth_results[slr_int]) < expected_growth[slr_int]:
            return
        if ssp_results[slr_int]:
            ssp_df = pd.concat(ssp_results[slr_int], axis=1).fillna(0.0)
            ssp_df.index.name = "ISO"
            ssp_df.to_csv(out_dir / f"exposure_avoid_{slr_int}_ssp.csv")
            print(f"  Written: exposure_avoid_{slr_int}_ssp.csv")
        else:
            print(f"  Skipping exposure_avoid_{slr_int}_ssp.csv: no SSP growth factors or SLR trajectory available.")
        growth_df_out = pd.concat(growth_results[slr_int], axis=1).fillna(0.0)
        growth_df_out.index.name = "ISO"
        growth_df_out.to_csv(out_dir / f"exposure_avoid_{slr_int}_growth_matrix.csv")
        print(f"  Written: exposure_avoid_{slr_int}_growth_matrix.csv")
        written[slr_int] = True
        ssp_results[slr_int] = []
        growth_results[slr_int] = []

    if n_workers == 1:
        _init_avoid_worker(chunk_ids, chunks_dir, flood_frac_dir, iso_lookup, rp_applied,
                            return_periods, slr_scenarios, growth_df,
                            slr_mm_sorted, slr_interp_mm, slr_traj)
        for slr_int, ssp, yr in ssp_tasks:
            _, _, _, series = _avoid_ssp_worker_task(slr_int, shares_by_intensity[slr_int], ssp, yr)
            if series is not None:
                ssp_results[slr_int].append(series)
            print(f"  [{slr_int}] {ssp} {yr} done.")
            _maybe_write_avoid(slr_int)
        for slr_int, g_pct in growth_tasks:
            _, _, df = _avoid_growth_matrix_worker_task(slr_int, shares_by_intensity[slr_int], g_pct)
            growth_results[slr_int].append(df)
            print(f"  [{slr_int}] growth g{g_pct} done.")
            _maybe_write_avoid(slr_int)
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_avoid_worker,
            initargs=(chunk_ids, chunks_dir, flood_frac_dir, iso_lookup, rp_applied,
                      return_periods, slr_scenarios, growth_df,
                      slr_mm_sorted, slr_interp_mm, slr_traj),
        ) as pool:
            futures = {
                pool.submit(_avoid_ssp_worker_task, slr_int, shares_by_intensity[slr_int], ssp, yr): "ssp"
                for slr_int, ssp, yr in ssp_tasks
            }
            futures.update({
                pool.submit(_avoid_growth_matrix_worker_task, slr_int, shares_by_intensity[slr_int], g_pct): "growth"
                for slr_int, g_pct in growth_tasks
            })
            for future in as_completed(futures):
                if futures[future] == "ssp":
                    slr_int, ssp, yr, series = future.result()
                    if series is not None:
                        ssp_results[slr_int].append(series)
                    print(f"  [{slr_int}] {ssp} {yr} done.")
                else:
                    slr_int, g_pct, df = future.result()
                    growth_results[slr_int].append(df)
                    print(f"  [{slr_int}] growth g{g_pct} done.")
                _maybe_write_avoid(slr_int)

    for slr_int in avoid_slr_intensities:
        _maybe_write_avoid(slr_int)

    print(f"\nAll exposure CSVs written to: {out_dir}")


if __name__ == "__main__":
    main()
