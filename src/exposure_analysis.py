"""Compute coarse flood exposure for baseline, protect, retreat, avoid, protect + avoid.

All grids are on the population raster's ~1 km grid (H, W). Flood fractions are 0-1,
-1.0 = nodata, keyed by (return_period, slr_name); geo_ids are WRI geogunit-107 IDs
(-1 = none); population is the 2020 baseline. FLOPROS protection is per geogunit and
each 1 km cell lies in one geogunit, so protection needs no hazard-curve interpolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_NODATA = -1.0


def _valid(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr != _NODATA)


def _safe(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    # Returns `arr` itself when already clean float64 (callers never mutate it):
    # repeated large copies fragmented the Windows heap enough to fail allocations.
    if arr.dtype == np.float64 and np.all(_valid(arr)):
        return arr
    out = arr.astype("float64", copy=True)
    out[~_valid(out)] = fill
    return out


def _trapezoid_eai(exposures: np.ndarray, return_periods: list[int]) -> np.ndarray:
    # Trapezoid over exceedance probability 1/RP, plus a rectangle tail beyond the max RP.
    rps = sorted(return_periods)
    p = 1.0 / np.array(rps, dtype=float)
    eai = np.zeros(exposures.shape[0], dtype=float)
    for i in range(len(rps) - 1):
        eai += 0.5 * (exposures[:, i] + exposures[:, i + 1]) * (p[i] - p[i + 1])
    eai += exposures[:, -1] * p[-1]
    return eai


def _build_iso_index(
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    # Returns (iso_list, iso_idx, cell_idx) for O(H*W) per-country np.bincount sums.
    iso_list = sorted(set(iso_lookup.values()))
    if not iso_list:
        empty_i = np.array([], dtype="int32")
        empty_c = np.array([], dtype="int64")
        return iso_list, empty_i, empty_c

    iso_to_idx = {iso: i for i, iso in enumerate(iso_list)}

    max_gid = max(iso_lookup.keys()) + 1
    gid_to_iso_idx = np.full(max_gid, -1, dtype="int32")
    for gid, iso in iso_lookup.items():
        if 0 <= gid < max_gid:
            idx = iso_to_idx.get(iso, -1)
            if idx >= 0:
                gid_to_iso_idx[gid] = idx

    flat_geo = geo_ids.ravel()
    in_range = (flat_geo >= 0) & (flat_geo < max_gid)
    candidate_idx = np.where(in_range)[0]
    iso_idx_mapped = gid_to_iso_idx[flat_geo[candidate_idx]]

    has_iso = iso_idx_mapped >= 0
    return iso_list, iso_idx_mapped[has_iso].astype("int32"), candidate_idx[has_iso]


def scatter_country_values(
    values_by_iso: dict[str, float],
    iso_list: list[str],
    iso_idx: np.ndarray,
    cell_idx: np.ndarray,
    shape: tuple[int, int],
    default: float = 1.0,
) -> np.ndarray:
    """Scatter one value per ISO country onto an (H, W) grid.

    Cells without a geogunit, or whose country is missing from `values_by_iso`, get
    `default`. `iso_list`/`iso_idx`/`cell_idx` come from `_build_iso_index`.
    """
    values = np.array([values_by_iso.get(iso, default) for iso in iso_list], dtype="float64")
    grid = np.full(shape, default, dtype="float64")
    grid_flat = grid.ravel()
    if len(iso_list) > 0:
        grid_flat[cell_idx] = values[iso_idx]
    return grid_flat.reshape(shape)


def build_protection_fraction(
    flood_fractions: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_baseline: str,
    rp_applied: dict[int, float],
    geo_ids: np.ndarray,
) -> np.ndarray:
    """Compute the per-cell protection-level flood fraction.

    ``prot_frac = flood_fraction[(RP_applied(geogunit), slr_baseline)]``, where
    RP_applied is the smallest simulated RP >= the geogunit's FLOPROS standard
    (`rp_applied`, years), else the highest simulated RP. 0 where no geogunit/RP matches.
    """
    sorted_rps = sorted(return_periods)
    prot_frac = np.zeros(geo_ids.shape, dtype="float64")

    def _snap_rp(flopros_rp: float) -> int:
        candidates = [rp for rp in sorted_rps if rp >= flopros_rp]
        return min(candidates) if candidates else max(sorted_rps)

    rp_groups: dict[int, list[int]] = {}
    for geo_id, flopros_rp in rp_applied.items():
        snapped = _snap_rp(flopros_rp)
        rp_groups.setdefault(snapped, []).append(geo_id)

    for rp, geo_ids_in_group in rp_groups.items():
        key = (rp, slr_baseline)
        if key not in flood_fractions:
            continue
        ff = _safe(flood_fractions[key])
        mask = np.isin(geo_ids, geo_ids_in_group) & (geo_ids >= 0)
        prot_frac[mask] = ff[mask]

    return prot_frac


def build_adapt_protection_fraction(
    flood_fractions: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_intensity: str,
    rp_applied: dict[int, float],
    geo_ids: np.ndarray,
) -> np.ndarray:
    """Compute `build_protection_fraction` at an adaptation SLR intensity."""
    return build_protection_fraction(
        flood_fractions, return_periods, slr_intensity, rp_applied, geo_ids
    )


# Redistribution is split into country_sums + apply_country_shares so chunk-streaming
# callers can accumulate per-ISO sums across chunks (addition is associative).

def country_sums(
    per_cell_amount: np.ndarray,
    capacity: np.ndarray,
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
    iso_index: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-ISO sums ``(amount_by_iso, capacity_by_iso)`` for ISOs in `geo_ids`.

    `iso_index` is an optional precomputed `_build_iso_index` result.
    """
    iso_list, iso_idx, cell_idx = (
        iso_index if iso_index is not None else _build_iso_index(geo_ids, iso_lookup)
    )
    if not iso_list:
        return {}, {}

    amt = per_cell_amount.ravel()[cell_idx]
    cap = capacity.ravel()[cell_idx]
    total_amt = np.bincount(iso_idx, weights=amt, minlength=len(iso_list))
    total_cap = np.bincount(iso_idx, weights=cap, minlength=len(iso_list))
    return (
        {iso: float(total_amt[i]) for i, iso in enumerate(iso_list)},
        {iso: float(total_cap[i]) for i, iso in enumerate(iso_list)},
    )


def apply_country_shares(
    capacity: np.ndarray,
    share_by_iso: dict[str, float],
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
    iso_index: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Return ``capacity[cell] * share_by_iso[country_of(cell)]``, 0 for missing ISOs."""
    iso_list, iso_idx, cell_idx = (
        iso_index if iso_index is not None else _build_iso_index(geo_ids, iso_lookup)
    )
    out = np.zeros_like(capacity)
    if not iso_list:
        return out

    share = np.array([share_by_iso.get(iso, 0.0) for iso in iso_list], dtype="float64")
    cap = capacity.ravel()[cell_idx]
    out_flat = out.ravel()
    out_flat[cell_idx] = cap * share[iso_idx]
    return out_flat.reshape(capacity.shape)


def _redistribute_by_country(
    per_cell_amount: np.ndarray,
    capacity: np.ndarray,
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> np.ndarray:
    # redistributed = country_total_amount * capacity / country_total_capacity;
    # 0 for countries with zero total amount or capacity.
    iso_index = _build_iso_index(geo_ids, iso_lookup)
    total_amt, total_cap = country_sums(per_cell_amount, capacity, geo_ids, iso_lookup, iso_index)
    share = {iso: total_amt[iso] / total_cap[iso] for iso in total_cap if total_cap[iso] > 0.0}
    return apply_country_shares(capacity, share, geo_ids, iso_lookup, iso_index)


def protect_exposure_grid(
    ff: np.ndarray,
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """Compute per-cell protect (and baseline) exposure for one (RP, SLR) scenario.

    Binary: 0 where ``ff <= adapt_prot_frac``, else ``ff * population`` (a defence
    is fully breached beyond its design flood fraction). Baseline is this with the
    design threshold built at SLR_0, so protect can never exceed baseline.
    """
    ff_s = _safe(ff)
    exposed_mask = ff_s > _safe(adapt_prot_frac)
    return np.where(exposed_mask, ff_s * population, 0.0)


def compute_retreat_capacity(
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> np.ndarray:
    """Compute effective population after retreat for one design grid.

    ``eff_pop = (1 - apf) * population + redistributed(apf * population)``, with
    retreaters redistributed within the country proportional to ``1 - apf``.
    Independent of the evaluated (RP, SLR): compute once per slr_intensity.
    """
    apf = _safe(adapt_prot_frac)

    retreating = apf * population
    safe_cap_area = np.maximum(0.0, 1.0 - apf)

    redistributed = _redistribute_by_country(retreating, safe_cap_area, geo_ids, iso_lookup)
    return population * (1.0 - apf) + redistributed


def compute_retreat(
    ff: np.ndarray,
    adapt_prot_frac: np.ndarray,
    eff_pop: np.ndarray,
) -> np.ndarray:
    """Compute per-cell retreat exposure for one (RP, SLR) scenario.

    ``exposure = clip((ff - apf) / (1 - apf), 0, 1) * eff_pop``: after retreat the
    population sits only in the safe share ``1 - apf``, so exceedance is measured
    relative to that share.
    """
    ff_s = _safe(ff)
    apf = _safe(adapt_prot_frac)
    safe_frac = np.maximum(1e-10, 1.0 - apf)
    adapted_ff = np.minimum(1.0, np.maximum(0.0, ff_s - apf) / safe_frac)
    return adapted_ff * eff_pop


def compute_avoid_redirected(
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
    growth_factor: "float | np.ndarray",
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> np.ndarray:
    """Compute the redirected-growth grid for one (design grid, growth factor) pair.

    ``apf * population * max(0, g - 1)`` redistributed within the country
    proportional to ``1 - apf``. Independent of the evaluated (RP, SLR): compute
    once per (slr_intensity, ssp, year). `growth_factor` 1 = no growth.
    """
    apf = _safe(adapt_prot_frac)
    g = np.broadcast_to(np.asarray(growth_factor, dtype="float64"), population.shape)

    avoiding = apf * population * np.maximum(0.0, g - 1.0)
    safe_cap_area = np.maximum(0.0, 1.0 - apf)

    return _redistribute_by_country(avoiding, safe_cap_area, geo_ids, iso_lookup)


def compute_avoid(
    ff: np.ndarray,
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
    redirected: np.ndarray,
    growth_factor: "float | np.ndarray",
    existing_prot_frac: "np.ndarray | None" = None,
) -> np.ndarray:
    """Compute per-cell avoid (or protect + avoid) exposure for one (RP, SLR) scenario.

    `population` is the unscaled 2020 E0. `existing_prot_frac` = baseline's grid
    gives avoid (equals baseline at g <= 1); None uses `adapt_prot_frac`, giving
    protect + avoid (equals protect at g <= 1). Sum of: E0 exposed under its own
    protection scaled by min(1, g); in-place growth outside the design floodplain;
    redirected growth. Growth terms use retreat's (ff - apf) / (1 - apf) exceedance
    on the increment max(0, g - 1).
    """
    ff_s = _safe(ff)
    apf = _safe(adapt_prot_frac)
    epf = apf if existing_prot_frac is None else _safe(existing_prot_frac)
    g = np.broadcast_to(np.asarray(growth_factor, dtype="float64"), population.shape)

    avoid_exp = np.zeros_like(population)

    exposed = ff_s > epf
    term1_scale = np.minimum(g, 1.0)
    avoid_exp[exposed] = ff_s[exposed] * population[exposed] * term1_scale[exposed]

    exceeds = ff_s > apf
    safe_frac = np.maximum(1e-10, 1.0 - apf)
    marginal_frac = np.minimum(1.0, (ff_s - apf) / safe_frac)

    organic_grown = population * np.maximum(0.0, 1.0 - apf) * np.maximum(0.0, g - 1.0)
    avoid_exp[exceeds] += marginal_frac[exceeds] * organic_grown[exceeds]

    avoid_exp[exceeds] += marginal_frac[exceeds] * redirected[exceeds]

    return avoid_exp


def compute_country_eai(
    exposure_per_rp_slr: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_scenarios: list[str],
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
    iso_index: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Aggregate exposure grids to per-country EAI (index ISO, columns SLR scenario).

    A missing (RP, SLR) key counts as zero exposure.
    """
    sorted_rps = sorted(return_periods)

    iso_list, iso_idx, cell_idx = (
        iso_index if iso_index is not None else _build_iso_index(geo_ids, iso_lookup)
    )
    n_iso = len(iso_list)

    rows: dict[str, dict[str, float]] = {}
    for slr in slr_scenarios:
        country_rp = np.zeros((n_iso, len(sorted_rps)), dtype="float64")
        for rp_i, rp in enumerate(sorted_rps):
            grid = exposure_per_rp_slr.get((rp, slr), np.zeros(geo_ids.shape))
            weights = grid.ravel()[cell_idx]
            country_rp[:, rp_i] = np.bincount(iso_idx, weights=weights, minlength=n_iso)

        eai_arr = _trapezoid_eai(country_rp, sorted_rps)
        for i, iso in enumerate(iso_list):
            rows.setdefault(iso, {})[slr] = float(eai_arr[i])

    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def compute_country_eai_from_cells(
    weights_per_rp_slr: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_scenarios: list[str],
    iso_list: list[str],
    iso_idx: np.ndarray,
) -> pd.DataFrame:
    """Compute `compute_country_eai` from exposure already gathered per cell.

    Each array is parallel to `iso_idx` (exact zeros may be dropped). Bit-identical
    to `compute_country_eai`: np.bincount sums in cell order, and zeros never change
    a sum. A missing key counts as zero exposure.
    """
    sorted_rps = sorted(return_periods)
    n_iso = len(iso_list)
    zeros = np.zeros(len(iso_idx), dtype="float64")

    rows: dict[str, dict[str, float]] = {}
    for slr in slr_scenarios:
        country_rp = np.zeros((n_iso, len(sorted_rps)), dtype="float64")
        for rp_i, rp in enumerate(sorted_rps):
            weights = weights_per_rp_slr.get((rp, slr), zeros)
            country_rp[:, rp_i] = np.bincount(iso_idx, weights=weights, minlength=n_iso)

        eai_arr = _trapezoid_eai(country_rp, sorted_rps)
        for i, iso in enumerate(iso_list):
            rows.setdefault(iso, {})[slr] = float(eai_arr[i])

    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def interpolate_eai_linear(
    eai_df: pd.DataFrame,
    slr_mm_values: list[float],
    target_slr_mm: np.ndarray,
) -> pd.DataFrame:
    """Linearly interpolate per-ISO EAI from simulated SLR levels to `target_slr_mm`.

    Assumes EAI varies linearly between modelled SLR steps. `eai_df` columns must
    match the strictly increasing `slr_mm_values` (mm) 1:1. Output columns
    ``SLR_{mm}`` (rounded), clipped to >= 0; ValueError on mismatched inputs.
    """
    x = np.asarray(slr_mm_values, dtype=float)
    if x.size < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("slr_mm_values must have >= 2 strictly increasing values.")
    if eai_df.shape[1] != x.size:
        raise ValueError(
            f"eai_df has {eai_df.shape[1]} columns {list(eai_df.columns)} but "
            f"slr_mm_values has {x.size} entries {slr_mm_values} — they must "
            "correspond 1:1 in the same order."
        )

    cols = [f"SLR_{int(round(mm))}" for mm in target_slr_mm]
    rows: dict = {}
    for iso, row in eai_df.iterrows():
        y = row.to_numpy(dtype=float)
        rows[iso] = np.clip(np.interp(target_slr_mm, x, y), 0.0, None)
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def resolve_ssp_scenario_eai(
    eai_df: pd.DataFrame,
    slr_mm_values: list[float],
    growth_df: "pd.DataFrame",
    ssps: list[str],
    years: list[int],
    slr_traj: "pd.DataFrame",
) -> pd.DataFrame:
    """Resolve per-country EAI at each (SSP, year) as ``EAI_{SSP}_{year}`` columns.

    Interpolates `slr_traj` (global-mean SLR in mm, indexed by year) to the year,
    interpolates `eai_df` linearly to that SLR, then scales by the country's
    growth factor. SSPs missing from `slr_traj` are skipped; ValueError as in
    `interpolate_eai_linear`.
    """
    from population_growth import interpolate_growth_factor

    x = np.asarray(slr_mm_values, dtype=float)
    if x.size < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("slr_mm_values must have >= 2 strictly increasing values.")
    if eai_df.shape[1] != x.size:
        raise ValueError(
            f"eai_df has {eai_df.shape[1]} columns but slr_mm_values has "
            f"{x.size} entries — they must correspond 1:1 in the same order."
        )

    traj_years = slr_traj.index.to_numpy(dtype=float)
    rows: list[dict] = []
    for iso in eai_df.index:
        y = eai_df.loc[iso].to_numpy(dtype=float)
        row: dict = {"ISO": iso}
        for ssp in ssps:
            if ssp not in slr_traj.columns:
                continue
            traj_mm = slr_traj[ssp].to_numpy(dtype=float)
            for yr in years:
                slr_at_yr = float(np.interp(float(yr), traj_years, traj_mm))
                eai_at_yr = float(np.clip(np.interp(slr_at_yr, x, y), 0.0, None))
                g = interpolate_growth_factor(growth_df, ssp, iso, yr, default=1.0)
                row[f"EAI_{ssp}_{yr}"] = eai_at_yr * g
        rows.append(row)

    return pd.DataFrame(rows).set_index("ISO")


def apply_growth_rates_to_eai(
    eai_df: pd.DataFrame,
    growth_rates: np.ndarray,
) -> pd.DataFrame:
    """Scale per-country EAI by scenario-neutral growth rates (``EAI_{SLR}_g{pct}``).

    `growth_rates` are fractions (-0.5 = -50%); pct = round(g * 100). Valid only for
    baseline/protect/retreat, which are linear in uniform population scaling;
    avoid is not and must be recomputed.
    """
    rows: list[dict] = []
    pct_labels = [int(round(g * 100)) for g in growth_rates]
    for iso in eai_df.index:
        row: dict = {"ISO": iso}
        for slr in eai_df.columns:
            base_eai = float(eai_df.loc[iso, slr])
            for pct, g in zip(pct_labels, growth_rates):
                row[f"EAI_{slr}_g{pct}"] = base_eai * (1.0 + float(g))
        rows.append(row)

    return pd.DataFrame(rows).set_index("ISO")
