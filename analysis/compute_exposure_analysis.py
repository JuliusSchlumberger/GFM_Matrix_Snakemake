"""Compute baseline/protect/retreat/avoid/protect+avoid flood exposure globally.

Streams per-chunk flood-fraction, population and geogunit rasters one chunk at a
time (no global mosaic) at ~1 km population-grid resolution.

Usage:
    python compute_exposure_analysis.py \\
        [--config snakemake_workflow/config/config.yml] \\
        --out-dir D:/GFM/merged_results/exposure
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _chunk_is_populated, _read_band, discover_chunk_ids  # noqa: E402
from config_utils import get_data_catalog, load_config, merged_waterlevel_names, retry_transient_io  # noqa: E402
from population_growth import load_ssp_growth_factors, interpolate_growth_factor  # noqa: E402
from validation import load_iso_lookup  # noqa: E402
from analysis_figures import load_slr_trajectories  # noqa: E402
from exposure_analysis import (                                 # noqa: E402
    build_adapt_protection_fraction,
    protect_exposure_grid,
    compute_retreat,
    compute_avoid,
    compute_country_eai,
    compute_country_eai_from_cells,
    interpolate_eai_linear,
    resolve_ssp_scenario_eai,
    apply_growth_rates_to_eai,
    country_sums,
    apply_country_shares,
    scatter_country_values,
    _build_iso_index,
    _safe,
)


_REPO_ROOT = Path(__file__).resolve().parent.parent

# label -> existing population keeps BASELINE protection (True: avoid, equals
# baseline at growth <= 0) or gets the design protection (False: protect + avoid,
# equals protect at growth <= 0).
AVOID_VARIANTS: dict[str, bool] = {"avoid": True, "protect_avoid": False}


def _slr_mm(slr_name: str) -> int:
    return int(slr_name.split("_")[1])


# Chunks (~600x600 px GeoTIFFs) are read one at a time, never mosaicked: a
# global (RP, SLR) grid is ~750M px (~6 GB float64), 100 of them would need
# hundreds of GB; streaming keeps peak memory at one chunk (~1-2 MB).

@dataclass
class ChunkData:
    """Hold one chunk's nodata-cleaned rasters plus its country-index arrays.

    pop: people per cell (nodata -> 0.0); geo: geogunit id (nodata -> -1);
    ff: flood fraction 0-1 per available (return_period, slr_scenario), missing
    files omitted; iso_index: `(iso_list, iso_idx, cell_idx)` from _build_iso_index.
    """

    cid: str
    pop: np.ndarray
    geo: np.ndarray
    ff: dict[tuple[int, str], np.ndarray]
    iso_index: tuple[list[str], np.ndarray, np.ndarray]


def _load_chunk(
    cid: str,
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    iso_lookup: dict[int, str],
) -> ChunkData:
    pop = _read_band(chunks_dir / f"exposure_population_grid_{cid}.tif", "float64", 0.0)
    pop[~np.isfinite(pop)] = 0.0

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
    """Compute retreat's per-country share `Σ(apf·population) / Σ(1-apf)`.

    Also reused by every avoid task at this intensity, since growth is a
    per-country scalar: `avoiding_sum[iso] = retreating_sum[iso] * max(0, g-1)`.
    rp_applied: geogunit id -> FLOPROS-snapped return period (years).
    Countries with zero capacity are omitted.
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
    # Exact: _trapezoid_eai has fixed weights, so Σ_chunks trapezoid == trapezoid(Σ_chunks).
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
    required = [out_dir / f"exposure_{label}_base.csv", out_dir / f"exposure_{label}_growth_matrix.csv"]
    if expect_ssp:
        required.append(out_dir / f"exposure_{label}_ssp.csv")
    return all(p.exists() for p in required)


def _avoid_done(variant: str, slr_int: str, out_dir: Path, expect_ssp: bool) -> bool:
    required = [out_dir / f"exposure_{variant}_{slr_int}_growth_matrix.csv"]
    if expect_ssp:
        required.append(out_dir / f"exposure_{variant}_{slr_int}_ssp.csv")
    return all(p.exists() for p in required)


def _write_base_scenario_files(
    label: str, eai: pd.DataFrame, out_dir: Path,
    slr_mm_sorted: list[float], slr_interp_mm: np.ndarray, growth_rates: np.ndarray,
    growth_df, ssps: list[str], output_years: list[int], slr_traj,
) -> None:
    """Write File 1 (base), File 2 (growth matrix), File 3 (ssp) for one scenario.

    File 1: per-country EAI at each modelled SLR, 2020 population, no growth.
    File 2: File 1 linearly interpolated onto visualization.slr_interp, scaled by
    the uniform visualization.growth_rates axis (exact: exposure is linear in a
    uniform population scaling). File 3: File 1 interpolated to each SSP's p50 SLR
    at each year, scaled by that country's SSP/year growth factor.
    Baseline is protect with its design threshold at protection.baseline_waterlevel_name.
    """
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


# Avoid has no growth-free File 1 (redirected growth is growth-dependent), so
# each (variant, slr_int, SSP, year) / (variant, slr_int, growth) is recomputed
# in a worker process; workers receive only paths + small dicts.

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
    slr_baseline: str = "SLR_0",
) -> None:
    _AVOID_CTX.update(
        slr_baseline=slr_baseline,
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
    existing_pf: "np.ndarray | None" = None,
) -> np.ndarray:
    # existing_pf: baseline protection for avoid, None (= design apf) for protect + avoid.
    iso_list, iso_idx, cell_idx = chunk.iso_index
    g = scatter_country_values(growth_by_iso, iso_list, iso_idx, cell_idx, chunk.geo.shape, default=1.0)
    cap = np.maximum(0.0, 1.0 - apf)
    redirected = apply_country_shares(cap, redirected_share, chunk.geo, _AVOID_CTX["iso_lookup"], chunk.iso_index)
    return compute_avoid(ff, apf, chunk.pop, redirected, g, existing_prot_frac=existing_pf)


def _avoid_apf_pair(c: ChunkData, variant: str, slr_int: str) -> tuple[np.ndarray, "np.ndarray | None"]:
    ctx = _AVOID_CTX
    apf = build_adapt_protection_fraction(c.ff, ctx["return_periods"], slr_int, ctx["rp_applied"], c.geo)
    if not AVOID_VARIANTS[variant]:
        return apf, None
    return apf, build_adapt_protection_fraction(c.ff, ctx["return_periods"], ctx["slr_baseline"], ctx["rp_applied"], c.geo)


def _avoid_ssp_worker_task(
    variant: str, slr_int: str, share_retreat: dict[str, float], ssp: str, yr: int,
) -> tuple[str, str, str, int, "pd.Series | None"]:
    ctx = _AVOID_CTX
    growth_df = ctx["growth_df"]
    slr_traj = ctx["slr_traj"]

    if slr_traj is None or ssp not in slr_traj.columns:
        return variant, slr_int, ssp, yr, None

    iso_list = sorted(set(ctx["iso_lookup"].values()))
    growth_by_iso = {
        iso: (interpolate_growth_factor(growth_df, ssp, iso, yr, default=1.0)
              if growth_df is not None else 1.0)
        for iso in iso_list
    }
    share_avoid = {iso: share_retreat[iso] * max(0.0, growth_by_iso.get(iso, 1.0) - 1.0) for iso in share_retreat}

    def exposure_fn(c: ChunkData, ff: np.ndarray) -> np.ndarray:
        apf, existing_pf = _avoid_apf_pair(c, variant, slr_int)
        return _avoid_exposure_fn(c, ff, apf, growth_by_iso, share_avoid, existing_pf)

    avoid_eai = _stream_eai(
        ctx["chunk_ids"], ctx["chunks_dir"], ctx["flood_frac_dir"],
        ctx["return_periods"], ctx["waterlevel_names"], ctx["iso_lookup"], exposure_fn,
    )

    traj_years = slr_traj.index.to_numpy(dtype=float)
    traj_mm = slr_traj[ssp].to_numpy(dtype=float)
    slr_at_yr = float(np.interp(float(yr), traj_years, traj_mm))

    x = np.asarray(ctx["slr_mm_sorted"], dtype=float)
    resolved = {
        iso: float(np.clip(np.interp(slr_at_yr, x, avoid_eai.loc[iso].to_numpy(dtype=float)), 0.0, None))
        for iso in avoid_eai.index
    }
    return variant, slr_int, ssp, yr, pd.Series(resolved, name=f"EAI_{ssp}_{yr}")


def _avoid_growth_matrix_worker_task(
    variant: str, slr_int: str, share_retreat: dict[str, float], g_pct: int,
) -> tuple[str, str, int, pd.DataFrame]:
    ctx = _AVOID_CTX
    growth_factor = 1.0 + g_pct / 100.0
    share_avoid = {iso: v * max(0.0, growth_factor - 1.0) for iso, v in share_retreat.items()}
    growth_by_iso_const = {iso: growth_factor for iso in ctx["iso_lookup"].values()}

    def exposure_fn(c: ChunkData, ff: np.ndarray) -> np.ndarray:
        apf, existing_pf = _avoid_apf_pair(c, variant, slr_int)
        return _avoid_exposure_fn(c, ff, apf, growth_by_iso_const, share_avoid, existing_pf)

    avoid_eai = _stream_eai(
        ctx["chunk_ids"], ctx["chunks_dir"], ctx["flood_frac_dir"],
        ctx["return_periods"], ctx["waterlevel_names"], ctx["iso_lookup"], exposure_fn,
    )
    dense = interpolate_eai_linear(avoid_eai, ctx["slr_mm_sorted"], ctx["slr_interp_mm"])
    dense.columns = [f"EAI_{c}_g{g_pct}" for c in dense.columns]
    return variant, slr_int, g_pct, dense


# Chunk-partitioned HPC pipeline (scripts/run_exposure_pass1.py,
# reduce_exposure_shares.py, run_exposure_pass2.py, reduce_exposure_write.py).
# Chunks, not tasks, are partitioned across nodes and every task is evaluated
# per chunk read. Two passes are needed: retreat/avoid shares must be summed
# over ALL chunks before any chunk's EAI can be computed.

@dataclass
class AnalysisContext:
    """Hold config-derived setup shared by the HPC pass1/pass2/write steps.

    slr_mm_sorted: SLR in mm, parallel to slr_scenarios. growth_rates: fraction
    (e.g. -0.5 to 1.5). slr_traj: SSP median SLR trajectories. growth_df/slr_traj
    None -> File 3 skipped. rp_applied: geogunit id -> FLOPROS-snapped RP.
    """
    return_periods: list[int]
    slr_scenarios: list[str]
    slr_mm_sorted: list[float]
    slr_baseline: str
    slr_intensities: list[str]
    ssps: list[str]
    output_years: list[int]
    growth_rates: np.ndarray
    slr_interp_mm: np.ndarray
    growth_df: "pd.DataFrame | None"
    slr_traj: "pd.DataFrame | None"
    rp_applied: dict[int, int]
    iso_lookup: dict[int, str]
    chunks_dir: Path
    flood_frac_dir: Path


def load_analysis_context(cfg: dict) -> AnalysisContext:
    """Build the AnalysisContext from config (shared by main() and the HPC steps)."""
    bc = cfg["boundary_conditions"]
    adapt_cfg = cfg["adaptation"]
    pg_cfg = cfg["population_growth"]
    prot_cfg = cfg["protection"]
    viz = cfg["visualization"]

    return_periods = bc["return_periods"]
    slr_scenarios = merged_waterlevel_names(bc, adapt_cfg)
    slr_mm_sorted = [float(_slr_mm(s)) for s in slr_scenarios]
    slr_baseline = prot_cfg["baseline_waterlevel_name"]
    slr_intensities = adapt_cfg["slr_intensities"]
    default_rp = float(prot_cfg["default_rp"])
    sorted_rps = sorted(return_periods)

    ssps = pg_cfg["ssps"]
    output_years = [int(y) for y in pg_cfg["output_years"]]

    # Uniform (not per-country) growth axis, shared with the burning-ember y-axis.
    gr_cfg = viz["growth_rates"]
    growth_rates = np.arange(
        float(gr_cfg["min"]),
        float(gr_cfg["max"]) + float(gr_cfg["step"]) / 2,
        float(gr_cfg["step"]),
    )

    # EAI is assumed linear in SLR between modelled steps.
    si_cfg = viz["slr_interp"]
    slr_interp_mm = np.linspace(
        float(si_cfg["min_mm"]),
        float(si_cfg["max_mm"]),
        int(si_cfg["n_points"]),
    )

    traj_path = viz["slr_trajectories_csv"]
    slr_traj = None
    if traj_path and Path(traj_path).exists():
        traj_full = load_slr_trajectories(traj_path)
        p50_cols = {c: c.replace("_p50", "") for c in traj_full.columns if c.endswith("_p50")}
        slr_traj = traj_full[list(p50_cols)].rename(columns=p50_cols)

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    flopros = catalog.get_dataframe("flopros_protection_standards")
    coastal_rp = flopros["Coastal"].fillna(flopros["Riverine"]).fillna(default_rp)

    def _snap_rp(flopros_rp: float) -> int:
        candidates = [r for r in sorted_rps if r >= flopros_rp]
        return int(min(candidates)) if candidates else int(max(sorted_rps))

    rp_applied = {int(gid): _snap_rp(float(rp)) for gid, rp in coastal_rp.items()}
    iso_lookup = load_iso_lookup(catalog, "flopros_protection_standards")

    xlsx_path = Path(catalog.get_source("ssp_population_growth_factors").path)
    growth_df = load_ssp_growth_factors(xlsx_path) if xlsx_path.exists() else None

    merged_dir = Path(cfg["postprocessing"]["merged_outputs"])
    return AnalysisContext(
        return_periods=return_periods, slr_scenarios=slr_scenarios, slr_mm_sorted=slr_mm_sorted,
        slr_baseline=slr_baseline, slr_intensities=slr_intensities, ssps=ssps, output_years=output_years,
        growth_rates=growth_rates, slr_interp_mm=slr_interp_mm, growth_df=growth_df, slr_traj=slr_traj,
        rp_applied=rp_applied, iso_lookup=iso_lookup,
        chunks_dir=merged_dir / "chunks", flood_frac_dir=merged_dir / "chunks" / "flood_fraction",
    )


@dataclass
class ExposureTask:
    """Describe one exposure-EAI task; fields not relevant to `kind` stay None.

    existing_intensity: slr_baseline for "avoid", None (= design_intensity) for
    "protect_avoid". g_pct: growth rate in percent.
    """
    key: str
    kind: str  # "baseline" | "protect" | "retreat" | "avoid_ssp" | "avoid_growth"
    design_intensity: str  # slr_baseline for "baseline"
    slr_int: str | None = None
    variant: str | None = None
    existing_intensity: str | None = None
    share_retreat: dict | None = None
    growth_by_iso: dict | None = None
    redirected_share: dict | None = None
    ssp: str | None = None
    year: int | None = None
    g_pct: int | None = None


def build_exposure_tasks(
    ctx: AnalysisContext, shares_by_intensity: dict[str, dict[str, float]],
) -> list[ExposureTask]:
    """Enumerate every baseline/protect/retreat/avoid_ssp/avoid_growth task.

    `shares_by_intensity` must be the FINAL, globally-reduced share; a partial
    batch sum silently gives wrong retreat/avoid EAI.
    """
    tasks: list[ExposureTask] = [
        ExposureTask(key="baseline", kind="baseline", design_intensity=ctx.slr_baseline),
    ]

    for slr_int in ctx.slr_intensities:
        tasks.append(ExposureTask(
            key=f"protect_{slr_int}", kind="protect", design_intensity=slr_int, slr_int=slr_int,
        ))

    for slr_int in ctx.slr_intensities:
        share_retreat = shares_by_intensity.get(slr_int, {})
        tasks.append(ExposureTask(
            key=f"retreat_{slr_int}", kind="retreat", design_intensity=slr_int, slr_int=slr_int,
            share_retreat=share_retreat,
        ))

    available_ssps = [s for s in ctx.ssps if ctx.slr_traj is not None and s in ctx.slr_traj.columns]
    growth_pct_labels = [int(round(g * 100)) for g in ctx.growth_rates]

    for variant, keeps_baseline in AVOID_VARIANTS.items():
        existing_intensity = ctx.slr_baseline if keeps_baseline else None
        for slr_int in ctx.slr_intensities:
            share_retreat = shares_by_intensity.get(slr_int, {})

            if ctx.growth_df is not None and ctx.slr_traj is not None:
                # Thousands of geogunits but ~200 ISOs; first-occurrence order kept.
                unique_isos = list(dict.fromkeys(ctx.iso_lookup.values()))
                for ssp in available_ssps:
                    for yr in ctx.output_years:
                        growth_by_iso = {
                            iso: interpolate_growth_factor(ctx.growth_df, ssp, iso, yr, default=1.0)
                            for iso in unique_isos
                        }
                        redirected_share = {
                            iso: share_retreat[iso] * max(0.0, growth_by_iso.get(iso, 1.0) - 1.0)
                            for iso in share_retreat
                        }
                        tasks.append(ExposureTask(
                            key=f"{variant}_{slr_int}_ssp_{ssp}_{yr}", kind="avoid_ssp",
                            design_intensity=slr_int, slr_int=slr_int,
                            variant=variant, existing_intensity=existing_intensity,
                            growth_by_iso=growth_by_iso, redirected_share=redirected_share,
                            ssp=ssp, year=yr,
                        ))

            for g_pct in growth_pct_labels:
                growth_factor = 1.0 + g_pct / 100.0
                growth_by_iso = {iso: growth_factor for iso in ctx.iso_lookup.values()}
                redirected_share = {iso: v * max(0.0, growth_factor - 1.0) for iso, v in share_retreat.items()}
                tasks.append(ExposureTask(
                    key=f"{variant}_{slr_int}_growth_{g_pct}", kind="avoid_growth",
                    design_intensity=slr_int, slr_int=slr_int,
                    variant=variant, existing_intensity=existing_intensity,
                    growth_by_iso=growth_by_iso, redirected_share=redirected_share,
                    g_pct=g_pct,
                ))
    return tasks


def pass1_shares_all_intensities(
    chunk_ids: list[str],
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    slr_intensities: list[str],
    rp_applied: dict[int, int],
    iso_lookup: dict[int, str],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Accumulate per-country (amount, capacity) raw sums for every slr_intensity.

    Pass 1 of the HPC pipeline. Sums are not divided: reduce_exposure_shares.py
    adds them across batches before the final divide.
    """
    amt_total: dict[str, dict[str, float]] = {si: {} for si in slr_intensities}
    cap_total: dict[str, dict[str, float]] = {si: {} for si in slr_intensities}
    # build_adapt_protection_fraction only reads ff[(rp, slr_int)] at design intensities.
    needed_slr = [slr for slr in slr_scenarios if slr in set(slr_intensities)]
    for cid in chunk_ids:
        if not _chunk_is_populated(chunks_dir, cid):
            continue
        c = _load_chunk(cid, chunks_dir, flood_frac_dir, return_periods, needed_slr, iso_lookup)
        for slr_int in slr_intensities:
            apf = build_adapt_protection_fraction(c.ff, return_periods, slr_int, rp_applied, c.geo)
            retreating = apf * c.pop
            cap = np.maximum(0.0, 1.0 - apf)
            amt, cap_d = country_sums(retreating, cap, c.geo, iso_lookup, c.iso_index)
            for iso, v in amt.items():
                amt_total[slr_int][iso] = amt_total[slr_int].get(iso, 0.0) + v
            for iso, v in cap_d.items():
                cap_total[slr_int][iso] = cap_total[slr_int].get(iso, 0.0) + v
    return {
        slr_int: {iso: (amt_total[slr_int].get(iso, 0.0), cap_total[slr_int][iso]) for iso in cap_total[slr_int]}
        for slr_int in slr_intensities
    }


def pass2_all_tasks(
    chunk_ids: list[str],
    chunks_dir: Path,
    flood_frac_dir: Path,
    return_periods: list[int],
    slr_scenarios: list[str],
    rp_applied: dict[int, int],
    iso_lookup: dict[int, str],
    tasks: list[ExposureTask],
) -> dict[str, pd.DataFrame]:
    """Compute every task's partial per-country EAI for this batch of chunks.

    Pass 2 of the HPC pipeline; partials are summed across batches. Only cells
    with a country are evaluated, and per (design, existing) intensity pair,
    cells where no scenario's ff exceeds either protection are dropped: they
    contribute exactly 0 to every task, and np.bincount keeps cell order, so
    results are bit-identical to the full-grid computation.
    """
    totals: dict[str, pd.DataFrame] = {}
    for cid in chunk_ids:
        if not _chunk_is_populated(chunks_dir, cid):
            continue
        c = _load_chunk(cid, chunks_dir, flood_frac_dir, return_periods, slr_scenarios, iso_lookup)
        iso_list, iso_idx, cell_idx = c.iso_index
        pop_cells = c.pop.ravel()[cell_idx]
        ff_cells = {key: ff.ravel()[cell_idx] for key, ff in c.ff.items()}
        apf_cache: dict[str, np.ndarray] = {}

        def _apf_cells(intensity: str) -> np.ndarray:
            if intensity not in apf_cache:
                apf_cache[intensity] = build_adapt_protection_fraction(
                    c.ff, return_periods, intensity, rp_applied, c.geo,
                ).ravel()[cell_idx]
            return apf_cache[intensity]

        sub_cache: dict[tuple, tuple] = {}

        for task in tasks:
            sub_key = (task.design_intensity, task.existing_intensity)
            if sub_key not in sub_cache:
                apf_cells = _apf_cells(task.design_intensity)
                epf_cells = _apf_cells(task.existing_intensity) if task.existing_intensity else None
                can_exceed = np.zeros(len(cell_idx), dtype=bool)
                for ff in ff_cells.values():
                    can_exceed |= _safe(ff) > _safe(apf_cells)
                    if epf_cells is not None:
                        can_exceed |= _safe(ff) > _safe(epf_cells)
                keep = np.nonzero(can_exceed)[0]
                sub_cache[sub_key] = (
                    iso_idx[keep], pop_cells[keep], apf_cells[keep],
                    epf_cells[keep] if epf_cells is not None else None,
                    {key: ff[keep] for key, ff in ff_cells.items()},
                )
            iso_k, pop_k, apf_k, epf_k, ff_k = sub_cache[sub_key]

            if task.kind in ("baseline", "protect"):
                weights = {key: protect_exposure_grid(ff, apf_k, pop_k) for key, ff in ff_k.items()}
            elif task.kind == "retreat":
                cap = np.maximum(0.0, 1.0 - apf_k)
                share = np.array([task.share_retreat.get(iso, 0.0) for iso in iso_list], dtype="float64")
                redistributed = cap * share[iso_k]
                eff_pop = pop_k * (1.0 - apf_k) + redistributed
                weights = {key: compute_retreat(ff, apf_k, eff_pop) for key, ff in ff_k.items()}
            else:  # avoid_ssp / avoid_growth
                g_values = np.array([task.growth_by_iso.get(iso, 1.0) for iso in iso_list], dtype="float64")
                g = g_values[iso_k]
                cap = np.maximum(0.0, 1.0 - apf_k)
                share = np.array([task.redirected_share.get(iso, 0.0) for iso in iso_list], dtype="float64")
                redirected = cap * share[iso_k]
                weights = {key: compute_avoid(ff, apf_k, pop_k, redirected, g, existing_prot_frac=epf_k)
                           for key, ff in ff_k.items()}

            chunk_eai = compute_country_eai_from_cells(weights, return_periods, slr_scenarios, iso_list, iso_k)
            totals[task.key] = chunk_eai if task.key not in totals else totals[task.key].add(chunk_eai, fill_value=0.0)
    return totals


def write_exposure_outputs(
    ctx: AnalysisContext,
    tasks: list[ExposureTask],
    results: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write every File 1/2/3 CSV from fully chunk-summed per-task EAI results."""
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)
    expect_ssp = ctx.growth_df is not None and ctx.slr_traj is not None

    for task in tasks:
        if task.kind not in ("baseline", "protect", "retreat"):
            continue
        if task.key not in results:
            print(f"  WARNING: no result for {task.key}; skipping.")
            continue
        _write_base_scenario_files(
            task.key, results[task.key], out_dir, ctx.slr_mm_sorted, ctx.slr_interp_mm, ctx.growth_rates,
            ctx.growth_df, ctx.ssps, ctx.output_years, ctx.slr_traj,
        )

    for variant in AVOID_VARIANTS:
        for slr_int in ctx.slr_intensities:
            ssp_series: list[pd.Series] = []
            for task in tasks:
                if task.kind != "avoid_ssp" or task.variant != variant or task.slr_int != slr_int:
                    continue
                avoid_eai = results.get(task.key)
                if avoid_eai is None or ctx.slr_traj is None or task.ssp not in ctx.slr_traj.columns:
                    continue
                traj_years = ctx.slr_traj.index.to_numpy(dtype=float)
                traj_mm = ctx.slr_traj[task.ssp].to_numpy(dtype=float)
                slr_at_yr = float(np.interp(float(task.year), traj_years, traj_mm))
                x = np.asarray(ctx.slr_mm_sorted, dtype=float)
                resolved = {
                    iso: float(np.clip(np.interp(slr_at_yr, x, avoid_eai.loc[iso].to_numpy(dtype=float)), 0.0, None))
                    for iso in avoid_eai.index
                }
                ssp_series.append(pd.Series(resolved, name=f"EAI_{task.ssp}_{task.year}"))

            growth_dfs: list[pd.DataFrame] = []
            for task in tasks:
                if task.kind != "avoid_growth" or task.variant != variant or task.slr_int != slr_int:
                    continue
                avoid_eai = results.get(task.key)
                if avoid_eai is None:
                    continue
                dense = interpolate_eai_linear(avoid_eai, ctx.slr_mm_sorted, ctx.slr_interp_mm)
                dense.columns = [f"EAI_{c}_g{task.g_pct}" for c in dense.columns]
                growth_dfs.append(dense)

            if ssp_series:
                ssp_out = pd.concat(ssp_series, axis=1).fillna(0.0)
                ssp_out.index.name = "ISO"
                ssp_out.to_csv(out_dir / f"exposure_{variant}_{slr_int}_ssp.csv")
                print(f"  Written: exposure_{variant}_{slr_int}_ssp.csv")
            elif expect_ssp:
                print(f"  Skipping exposure_{variant}_{slr_int}_ssp.csv: no matching results.")

            if growth_dfs:
                growth_out = pd.concat(growth_dfs, axis=1).fillna(0.0)
                growth_out.index.name = "ISO"
                growth_out.to_csv(out_dir / f"exposure_{variant}_{slr_int}_growth_matrix.csv")
                print(f"  Written: exposure_{variant}_{slr_int}_growth_matrix.csv")


def main() -> None:
    """Run the single-machine exposure pipeline, writing all CSVs to --out-dir."""
    _default_cfg = str(Path(__file__).resolve().parents[1] / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a scenario (baseline/protect_*/retreat_*) or an avoid / protect + avoid "
             "slr_intensity entirely if its output CSVs already exist in "
             "--out-dir, instead of recomputing and overwriting them. Useful "
             "for resuming after a partial failure without redoing already-"
             "correct, expensive work.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loading FLOPROS protection standards and SSP growth factors…")
    ctx = load_analysis_context(cfg)
    return_periods, slr_scenarios = ctx.return_periods, ctx.slr_scenarios
    slr_mm_sorted, slr_baseline, slr_intensities = ctx.slr_mm_sorted, ctx.slr_baseline, ctx.slr_intensities
    ssps, output_years = ctx.ssps, ctx.output_years
    growth_rates, slr_interp_mm = ctx.growth_rates, ctx.slr_interp_mm
    growth_df, slr_traj = ctx.growth_df, ctx.slr_traj
    rp_applied, iso_lookup = ctx.rp_applied, ctx.iso_lookup
    chunks_dir, flood_frac_dir = ctx.chunks_dir, ctx.flood_frac_dir
    if slr_traj is None:
        traj_path = cfg["visualization"]["slr_trajectories_csv"]
        print(f"  WARNING: SLR trajectories not found at {traj_path}; File 3 (*_ssp.csv) will be skipped.")
    if growth_df is None:
        print("  WARNING: SSP growth factors not found; File 3 (*_ssp.csv) will be skipped.")

    out_dir = Path(args.out_dir)
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    # Ocean/unpopulated chunks have zero-byte placeholder population files.
    chunk_ids, all_chunk_ids = discover_chunk_ids(flood_frac_dir, chunks_dir)
    if not all_chunk_ids:
        print(f"ERROR: no flood fraction files in {flood_frac_dir}")
        sys.exit(1)
    if not chunk_ids:
        print("ERROR: no population chunk files found — run the Snakemake postprocess target first.")
        sys.exit(1)
    skipped = len(all_chunk_ids) - len(chunk_ids)
    if skipped:
        print(f"  Skipping {skipped} unpopulated/ocean chunks (zero-byte population files).")
    print(f"Using {len(chunk_ids)} populated chunks, {len(return_periods)} RPs, {len(slr_scenarios)} SLR scenarios.")
    print("Processing chunk-by-chunk (no global mosaic) - peak memory stays flat regardless of study-area size.")

    expect_ssp = growth_df is not None and slr_traj is not None

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

    # share_retreat depends only on the design grid; reused by every avoid task.
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

    avoid_units = [(v, s) for v in AVOID_VARIANTS for s in slr_intensities]
    if args.skip_existing:
        done_units = [u for u in avoid_units if _avoid_done(*u, out_dir, expect_ssp)]
        for v, s in done_units:
            print(f"  Skipping {v.upper()} exposure ({s}) (--skip-existing, files already present).")
        avoid_units = [u for u in avoid_units if u not in done_units]

    # An SSP without a trajectory column would never reach expected_ssp, so no
    # avoid CSV would be written for that slr_intensity.
    available_ssps = [s for s in ssps if slr_traj is not None and s in slr_traj.columns]
    missing_ssps = [s for s in ssps if s not in available_ssps]
    if missing_ssps:
        print(f"  WARNING: no SLR trajectory for {missing_ssps} - excluded from AVOID File 3 (*_ssp.csv).")
    growth_pct_labels = [int(round(g * 100)) for g in growth_rates]
    ssp_tasks = [
        (v, slr_int, ssp, yr)
        for v, slr_int in avoid_units
        for ssp in available_ssps
        for yr in output_years
    ] if (growth_df is not None and slr_traj is not None) else []
    growth_tasks = [
        (v, slr_int, g_pct)
        for v, slr_int in avoid_units
        for g_pct in growth_pct_labels
    ]
    n_tasks = len(ssp_tasks) + len(growth_tasks)
    n_workers = max(1, min(os.cpu_count() or 1, max(n_tasks, 1)))
    print(
        f"\nComputing AVOID / PROTECT + AVOID exposure: {len(avoid_units)} (variant, slr_intensity) units x "
        f"{len(ssps)} SSPs x {len(output_years)} years = {len(ssp_tasks)} SSP tasks, plus "
        f"{len(avoid_units)} units x {len(growth_pct_labels)} growth rates = "
        f"{len(growth_tasks)} scenario-neutral tasks ({n_tasks} total). "
        f"-> {n_workers} worker process(es)…"
    )

    ssp_results: dict[tuple, list[pd.Series]] = {u: [] for u in avoid_units}
    growth_results: dict[tuple, list[pd.DataFrame]] = {u: [] for u in avoid_units}
    expected_ssp = {u: sum(1 for v, s, _, _ in ssp_tasks if (v, s) == u) for u in avoid_units}
    expected_growth = {u: sum(1 for v, s, _ in growth_tasks if (v, s) == u) for u in avoid_units}
    written = {u: False for u in avoid_units}

    def _maybe_write_avoid(unit: tuple) -> None:
        variant, slr_int = unit
        if written[unit]:
            return
        if len(ssp_results[unit]) < expected_ssp[unit]:
            return
        if len(growth_results[unit]) < expected_growth[unit]:
            return
        if ssp_results[unit]:
            ssp_df = pd.concat(ssp_results[unit], axis=1).fillna(0.0)
            ssp_df.index.name = "ISO"
            ssp_df.to_csv(out_dir / f"exposure_{variant}_{slr_int}_ssp.csv")
            print(f"  Written: exposure_{variant}_{slr_int}_ssp.csv")
        else:
            print(f"  Skipping exposure_{variant}_{slr_int}_ssp.csv: no SSP growth factors or SLR trajectory available.")
        growth_df_out = pd.concat(growth_results[unit], axis=1).fillna(0.0)
        growth_df_out.index.name = "ISO"
        growth_df_out.to_csv(out_dir / f"exposure_{variant}_{slr_int}_growth_matrix.csv")
        print(f"  Written: exposure_{variant}_{slr_int}_growth_matrix.csv")
        written[unit] = True
        ssp_results[unit] = []
        growth_results[unit] = []

    init_args = (chunk_ids, chunks_dir, flood_frac_dir, iso_lookup, rp_applied,
                 return_periods, slr_scenarios, growth_df,
                 slr_mm_sorted, slr_interp_mm, slr_traj, slr_baseline)
    if n_workers == 1:
        _init_avoid_worker(*init_args)
        for v, slr_int, ssp, yr in ssp_tasks:
            _, _, _, _, series = _avoid_ssp_worker_task(v, slr_int, shares_by_intensity[slr_int], ssp, yr)
            if series is not None:
                ssp_results[(v, slr_int)].append(series)
            print(f"  [{v} {slr_int}] {ssp} {yr} done.")
            _maybe_write_avoid((v, slr_int))
        for v, slr_int, g_pct in growth_tasks:
            _, _, _, df = _avoid_growth_matrix_worker_task(v, slr_int, shares_by_intensity[slr_int], g_pct)
            growth_results[(v, slr_int)].append(df)
            print(f"  [{v} {slr_int}] growth g{g_pct} done.")
            _maybe_write_avoid((v, slr_int))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_avoid_worker, initargs=init_args) as pool:
            futures = {
                pool.submit(_avoid_ssp_worker_task, v, slr_int, shares_by_intensity[slr_int], ssp, yr): "ssp"
                for v, slr_int, ssp, yr in ssp_tasks
            }
            futures.update({
                pool.submit(_avoid_growth_matrix_worker_task, v, slr_int, shares_by_intensity[slr_int], g_pct): "growth"
                for v, slr_int, g_pct in growth_tasks
            })
            for future in as_completed(futures):
                if futures[future] == "ssp":
                    v, slr_int, ssp, yr, series = future.result()
                    if series is not None:
                        ssp_results[(v, slr_int)].append(series)
                    print(f"  [{v} {slr_int}] {ssp} {yr} done.")
                else:
                    v, slr_int, g_pct, df = future.result()
                    growth_results[(v, slr_int)].append(df)
                    print(f"  [{v} {slr_int}] growth g{g_pct} done.")
                _maybe_write_avoid((v, slr_int))

    for unit in avoid_units:
        _maybe_write_avoid(unit)

    print(f"\nAll exposure CSVs written to: {out_dir}")


if __name__ == "__main__":
    main()
