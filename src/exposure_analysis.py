"""Coarse-resolution flood exposure analysis — baseline, protect, retreat, avoid, protect + avoid.

All functions operate on flood_fraction grids at the population raster's native
~1 km resolution (one float per population cell, values 0–1).  No fine-resolution
raster I/O is performed here; that is handled once in compute_flood_fraction_chunk.

Key data structures
-------------------
flood_fractions : dict[(return_period_int, slr_name) -> np.ndarray (H, W)]
    Flood fraction grids, one per (RP, SLR) combination.  Shape matches the
    population raster (H, W).  Values 0–1; -1.0 = nodata.

population : np.ndarray (H, W)  float64
    2020 baseline population per coarse cell.

geo_ids : np.ndarray (H, W)  int32
    WRI geogunit-107 ID per coarse cell; -1 = no geogunit.

iso_lookup : dict[int, str]
    geogunit_id → ISO-3 country code (from FLOPROS table).

rp_applied : dict[int, float]
    geogunit_id → applied FLOPROS protection return period (years).

Protection fraction per cell
-----------------------------
For each coarse cell, the protection-level flood fraction is simply:

    prot_frac[cell] = flood_fraction[(RP_applied(geogunit), SLR_0)][cell]

Because FLOPROS protection standards are per-geogunit (not per-pixel), and
every 1 km cell falls entirely within one geogunit, RP_applied is constant
per cell.  No fine-resolution hazard-curve interpolation is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


_NODATA = -1.0


# ── Utility helpers ───────────────────────────────────────────────────────────

def _valid(arr: np.ndarray) -> np.ndarray:
    """Boolean mask of valid (non-nodata, non-NaN) cells."""
    return np.isfinite(arr) & (arr != _NODATA)


def _safe(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace nodata / NaN with `fill`.

    Zero-copy fast path: if `arr` is already float64 and already fully
    valid (no nodata/NaN), it is returned UNCHANGED instead of allocating a
    fresh copy. Safe because every caller of _safe() in this module only
    reads the result (comparisons, multiplication) — none mutates it in
    place. This matters because _safe() runs on the same large flood-
    fraction/design-threshold grids thousands of times over a full run, and
    that many allocate/free cycles can fragment the Windows heap badly
    enough that a later, much smaller allocation fails despite plenty of
    total free memory. Pre-cleaning flood_fractions / design grids once
    right after they're built (see compute_exposure_analysis.py) means
    almost every subsequent _safe() call on them hits this fast path.
    """
    if arr.dtype == np.float64 and np.all(_valid(arr)):
        return arr
    out = arr.astype("float64", copy=True)
    out[~_valid(out)] = fill
    return out


def _trapezoid_eai(exposures: np.ndarray, return_periods: list[int]) -> np.ndarray:
    """Trapezoidal EAI integration (n_entities × n_rp) → (n_entities,)."""
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
    """Pre-compute flat index arrays for O(H×W) country aggregation via np.bincount.

    Returns
    -------
    iso_list : sorted list of ISO codes present in the study area.
    iso_idx  : 1-D int32 array — iso_list index for each valid cell (parallel to cell_idx).
    cell_idx : 1-D int64 array — flat index into geo_ids.ravel() for each valid cell.

    Build once per analysis run and pass the result to functions that need
    per-country sums, instead of rescanning the grid with one boolean mask per
    country (O(n_countries × H×W)).
    """
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
    """Scatter one value per ISO country onto a (H, W) grid.

    Cells with no resolvable geogunit (not covered by `cell_idx`), or whose
    country has no entry in `values_by_iso`, get `default`. O(H×W) via the
    precomputed index arrays, instead of an O(n_countries) Python loop
    assigning into one boolean mask per country.

    Parameters
    ----------
    values_by_iso : dict[str, float]
        Value to assign per ISO-3 country code.
    iso_list : list[str]
        Sorted ISO codes present in the study area, from `_build_iso_index`.
    iso_idx : np.ndarray
        Per-valid-cell index into `iso_list`, from `_build_iso_index`.
    cell_idx : np.ndarray
        Per-valid-cell flat index into the output grid, from `_build_iso_index`.
    shape : tuple[int, int]
        Output grid shape (H, W).
    default : float, optional
        Value for cells with no resolvable geogunit or unlisted country. Default 1.0.

    Returns
    -------
    np.ndarray
        (H, W) float64 grid of scattered values.
    """
    values = np.array([values_by_iso.get(iso, default) for iso in iso_list], dtype="float64")
    grid = np.full(shape, default, dtype="float64")
    grid_flat = grid.ravel()
    if len(iso_list) > 0:
        grid_flat[cell_idx] = values[iso_idx]
    return grid_flat.reshape(shape)


# ── Protection fraction ───────────────────────────────────────────────────────

def build_protection_fraction(
    flood_fractions: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_baseline: str,
    rp_applied: dict[int, float],
    geo_ids: np.ndarray,
) -> np.ndarray:
    """Compute the per-cell protection-level flood fraction.

    For each coarse cell: prot_frac = flood_fraction[(RP_applied(geogunit), slr_baseline)].
    RP_applied is the nearest simulated RP >= the FLOPROS standard for that geogunit
    (the highest simulated RP if none qualifies).

    Parameters
    ----------
    flood_fractions : dict[tuple[int, str], np.ndarray]
        Flood fraction grids keyed by (return_period, slr_name).
    return_periods : list[int]
        Simulated return periods (years).
    slr_baseline : str
        SLR scenario name to read protection fractions from (e.g. "SLR_0").
    rp_applied : dict[int, float]
        Geogunit ID -> FLOPROS protection standard return period (years).
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.

    Returns
    -------
    np.ndarray
        (H, W) float64 protection-level flood fraction grid; 0 where no
        matching geogunit/RP group is found.
    """
    sorted_rps = sorted(return_periods)
    prot_frac = np.zeros(geo_ids.shape, dtype="float64")

    # Build geogunit → applied simulated RP
    def _snap_rp(flopros_rp: float) -> int:
        candidates = [rp for rp in sorted_rps if rp >= flopros_rp]
        return min(candidates) if candidates else max(sorted_rps)

    # Group cells by their snapped RP
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
    """Compute `build_protection_fraction` for an adaptation SLR intensity.

    Parameters
    ----------
    flood_fractions : dict[tuple[int, str], np.ndarray]
        Flood fraction grids keyed by (return_period, slr_name).
    return_periods : list[int]
        Simulated return periods (years).
    slr_intensity : str
        Adaptation SLR scenario name to read protection fractions from.
    rp_applied : dict[int, float]
        Geogunit ID -> FLOPROS protection standard return period (years).
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.

    Returns
    -------
    np.ndarray
        (H, W) float64 protection-level flood fraction grid at `slr_intensity`.
    """
    return build_protection_fraction(
        flood_fractions, return_periods, slr_intensity, rp_applied, geo_ids
    )


# ── Country aggregation helper ────────────────────────────────────────────────

# ── Redistribution helpers (retreat + avoid) ──────────────────────────────────
#
# Split into two composable halves so a chunk-streaming caller can run them as
# separate passes over the data (see compute_exposure_analysis.py's pass1_shares/
# _stream_eai): country_sums only needs each cell once and produces small
# per-ISO dicts that are cheap to accumulate across chunks by plain addition
# (sums are associative - the accumulated dict after seeing every chunk is
# identical to summing over one global array); apply_country_shares then only
# needs the finished per-ISO share table (not the summed amounts) to produce
# each cell's redistributed value. _redistribute_by_country composes both in
# one call, for callers that already have a full array in memory.

def country_sums(
    per_cell_amount: np.ndarray,
    capacity: np.ndarray,
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
    iso_index: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Sum `per_cell_amount` and `capacity` per ISO country.

    Only includes ISOs actually present in this call's `geo_ids` - a chunk-local
    call naturally returns a subset of all countries; callers accumulating
    across chunks should sum these dicts together (plain float addition,
    associative).

    Parameters
    ----------
    per_cell_amount : np.ndarray
        (H, W) per-cell amount to sum per country.
    capacity : np.ndarray
        (H, W) per-cell capacity to sum per country.
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.
    iso_lookup : dict[int, str]
        Geogunit ID -> ISO-3 country code.
    iso_index : tuple[list[str], np.ndarray, np.ndarray] or None, optional
        Precomputed `_build_iso_index(geo_ids, iso_lookup)` result, to avoid
        rebuilding it when the caller already has one (e.g. once per chunk,
        reused across every (RP, SLR) or scenario at that chunk). Default None.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        `(total_amount_by_iso, total_capacity_by_iso)`.
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
    """Scatter each country's `share_by_iso[iso]` onto its cells, scaled by `capacity`.

        redistributed[cell] = capacity[cell] × share_by_iso[country_of(cell)]

    A country with no entry in `share_by_iso` (e.g. zero total capacity, or
    simply never seen) gets 0 everywhere - same "no redistribution" outcome
    as `_redistribute_by_country`'s zero-capacity handling.

    Parameters
    ----------
    capacity : np.ndarray
        (H, W) per-cell capacity weights.
    share_by_iso : dict[str, float]
        Per-ISO share to scatter.
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.
    iso_lookup : dict[int, str]
        Geogunit ID -> ISO-3 country code.
    iso_index : tuple[list[str], np.ndarray, np.ndarray] or None, optional
        Precomputed `_build_iso_index(geo_ids, iso_lookup)` result. Default None.

    Returns
    -------
    np.ndarray
        (H, W) grid of `capacity * share_by_iso[country]`, 0 where no entry.
    """
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
    """Distribute `per_cell_amount` (summed per country) proportionally to `capacity`.

    `capacity[cell]` drives how much of the country total flows to that cell:

        redistributed[cell] = country_total_amount × capacity[cell] / country_total_capacity

    Countries with zero total amount or zero total capacity get 0 everywhere.

    Uses `_build_iso_index` + `np.bincount` for O(H×W) aggregation, instead of
    an O(n_countries × H×W) Python loop over one boolean mask per country.
    Composes `country_sums` + `apply_country_shares` (see those for the
    chunk-streaming variants of this same computation).
    """
    iso_index = _build_iso_index(geo_ids, iso_lookup)
    total_amt, total_cap = country_sums(per_cell_amount, capacity, geo_ids, iso_lookup, iso_index)
    share = {iso: total_amt[iso] / total_cap[iso] for iso in total_cap if total_cap[iso] > 0.0}
    return apply_country_shares(capacity, share, geo_ids, iso_lookup, iso_index)


# ── Protect exposure (also used for baseline — see note below) ────────────────

def protect_exposure_grid(
    ff: np.ndarray,
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
) -> np.ndarray:
    """Compute per-cell protect exposure for one (RP, SLR) scenario.

    Binary: cells where flood_fraction <= adapt_prot_frac are fully
    protected -> 0. Cells where flood_fraction > adapt_prot_frac: exposure =
    ff x population (the full flooded population, not just the marginal
    exceedance - a structure holds exactly up to its design flood fraction
    and is treated as fully breached beyond it).

    Baseline reuses this SAME function: it is exactly "protect" with its
    design threshold calibrated at SLR_0 instead of an adaptation
    slr_intensities entry (see compute_exposure_analysis.py, which builds
    baseline's adapt_prot_frac via build_protection_fraction(..., "SLR_0",
    ...)). Using the same ff-vs-design-ff mechanism for baseline as for
    every adaptation measure makes baseline's threshold SLR-aware and
    guarantees protect_SLR_x can never show *more* exposure than baseline,
    since its design ff (calibrated at a higher SLR) is always >= baseline's
    (calibrated at SLR_0).

    Parameters
    ----------
    ff : np.ndarray
        (H, W) flood fraction grid for this (RP, SLR) scenario (0-1).
    adapt_prot_frac : np.ndarray
        (H, W) design-flood-fraction threshold grid (protection level).
    population : np.ndarray
        (H, W) population per cell.

    Returns
    -------
    np.ndarray
        (H, W) exposed population grid.
    """
    ff_s = _safe(ff)
    exposed_mask = ff_s > _safe(adapt_prot_frac)
    return np.where(exposed_mask, ff_s * population, 0.0)


# ── Retreat exposure ──────────────────────────────────────────────────────────

def compute_retreat_capacity(
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> np.ndarray:
    """Compute the effective population per cell after retreat, for one adaptation design grid.

    Depends only on `adapt_prot_frac` (i.e. the slr_intensity design grid),
    `population` and `geo_ids` — NOT on the (return_period, waterlevel_name)
    scenario being evaluated. Callers should compute this once per
    slr_intensity and reuse it across every simulated scenario via
    compute_retreat(), instead of recomputing the country redistribution for
    every (return_period, waterlevel_name) pair.

    People in the design floodplain (adapt_prot_frac × population per cell)
    retreat and are redistributed proportionally to the non-flooded capacity
    (1 − adapt_prot_frac) across all cells in the same country:

        eff_pop = (1 − adapt_prot_frac) × population + redistributed_retreaters

    Parameters
    ----------
    adapt_prot_frac : np.ndarray
        (H, W) design-flood-fraction threshold grid for this slr_intensity.
    population : np.ndarray
        (H, W) population per cell.
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.
    iso_lookup : dict[int, str]
        Geogunit ID -> ISO-3 country code.

    Returns
    -------
    np.ndarray
        (H, W) effective population grid after retreat.
    """
    apf = _safe(adapt_prot_frac)

    # People leaving each cell (in the design floodplain)
    retreating = apf * population
    # Receiving capacity per cell (non-flooded area under the design event)
    safe_cap_area = np.maximum(0.0, 1.0 - apf)

    redistributed = _redistribute_by_country(retreating, safe_cap_area, geo_ids, iso_lookup)
    return population * (1.0 - apf) + redistributed


def compute_retreat(
    ff: np.ndarray,
    adapt_prot_frac: np.ndarray,
    eff_pop: np.ndarray,
) -> np.ndarray:
    """Compute per-cell retreat exposure for one (RP, SLR) scenario.

    `eff_pop` is the precomputed effective population from
    compute_retreat_capacity() for this scenario's adaptation design grid.

    retreat_exposure = [(ff − adapt_prot_frac) / (1 − adapt_prot_frac)] × eff_pop,
    for ff > adapt_prot_frac (0 otherwise).

    The /(1 - adapt_prot_frac) normalization matters because retreat fully
    vacates the design floodplain (adapt_prot_frac × population leaves), so
    eff_pop is concentrated entirely in the remaining (1 - adapt_prot_frac)
    "safe" share of the cell, not spread over the whole cell. The exposed
    share of that concentrated population when a flood of extent ff occurs
    is therefore the marginal exceedance measured AS A FRACTION OF THE SAFE
    AREA, not of the whole cell.

    Parameters
    ----------
    ff : np.ndarray
        (H, W) flood fraction grid for this (RP, SLR) scenario (0-1).
    adapt_prot_frac : np.ndarray
        (H, W) design-flood-fraction threshold grid.
    eff_pop : np.ndarray
        (H, W) effective population grid from `compute_retreat_capacity`.

    Returns
    -------
    np.ndarray
        (H, W) exposed population grid.
    """
    ff_s = _safe(ff)
    apf = _safe(adapt_prot_frac)
    safe_frac = np.maximum(1e-10, 1.0 - apf)
    adapted_ff = np.minimum(1.0, np.maximum(0.0, ff_s - apf) / safe_frac)
    return adapted_ff * eff_pop


# ── Avoid exposure ────────────────────────────────────────────────────────────

def compute_avoid_redirected(
    adapt_prot_frac: np.ndarray,
    population: np.ndarray,
    growth_factor: "float | np.ndarray",
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
) -> np.ndarray:
    """Compute the per-cell redirected-growth grid for one (adapt_prot_frac, growth_factor) pair.

    Depends only on `adapt_prot_frac` (i.e. the slr_intensity design grid) and
    `growth_factor` (i.e. the SSP × year scenario) — NOT on the
    (return_period, waterlevel_name) scenario being evaluated. Callers should
    compute this once per (slr_intensity, ssp, year) and reuse it across every
    simulated scenario via compute_avoid(), instead of recomputing the country
    redistribution for every (return_period, waterlevel_name) pair.

    avoiding[cell] = adapt_prot_frac × population × max(0, growth_factor − 1)
    redistributed proportionally to (1 − adapt_prot_frac) across all cells in
    the same country.

    Parameters
    ----------
    adapt_prot_frac : np.ndarray
        (H, W) design-flood-fraction threshold grid for this slr_intensity.
    population : np.ndarray
        (H, W) 2020 baseline population per cell.
    growth_factor : float or np.ndarray
        Per-country or scenario-neutral population growth multiplier (1 = no growth).
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.
    iso_lookup : dict[int, str]
        Geogunit ID -> ISO-3 country code.

    Returns
    -------
    np.ndarray
        (H, W) redistributed redirected-growth grid.
    """
    apf = _safe(adapt_prot_frac)
    g = np.broadcast_to(np.asarray(growth_factor, dtype="float64"), population.shape)

    # Avoiding people: growth increment that would have gone into the floodplain
    avoiding = apf * population * np.maximum(0.0, g - 1.0)
    # Receiving capacity (area-weighted)
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

    `population` is E0, the unscaled 2020 population; `redirected` is the
    precomputed grid from `compute_avoid_redirected` for this scenario's
    (adapt_prot_frac, growth_factor). `existing_prot_frac` selects the
    strategy: pass baseline's protection grid for avoid (growth steered away,
    no new defence raised; equals baseline exactly at growth_factor <= 1), or
    leave it None (default) to use `adapt_prot_frac` for protect + avoid
    (protection raised AND growth steered away; equals protect exactly at
    growth_factor <= 1).

    Exposure sums three terms: the existing E0 population exposed under its
    own protection (scaled by min(1,g) so it shrinks under population decline
    like protect/baseline), growth that stays in place outside the design
    floodplain, and growth redirected elsewhere in the country. The latter two
    use retreat's /(1 - adapt_prot_frac)-normalized marginal exceedance,
    applied to the growth increment max(0, g-1) — omitting the in-place-growth
    term would wrongly treat it as permanently safe beyond the design standard.

    Parameters
    ----------
    ff : np.ndarray
        (H, W) flood fraction grid for this (RP, SLR) scenario (0-1).
    adapt_prot_frac : np.ndarray
        (H, W) design-flood-fraction threshold that growth is steered out of.
    population : np.ndarray
        (H, W) E0, the unscaled 2020 population per cell.
    redirected : np.ndarray
        (H, W) redistributed-growth grid from `compute_avoid_redirected`.
    growth_factor : float or np.ndarray
        Per-country or scenario-neutral population growth multiplier (1 = no growth).
    existing_prot_frac : np.ndarray or None, optional
        Protection grid for the existing population: baseline's grid for avoid,
        or None (default) to use `adapt_prot_frac` (protect + avoid).

    Returns
    -------
    np.ndarray
        (H, W) exposed population grid.
    """
    ff_s = _safe(ff)
    apf = _safe(adapt_prot_frac)
    epf = apf if existing_prot_frac is None else _safe(existing_prot_frac)
    g = np.broadcast_to(np.asarray(growth_factor, dtype="float64"), population.shape)

    avoid_exp = np.zeros_like(population)

    # Original (2020) population term: same binary form as protect_exposure_grid,
    # at the existing population's protection, scaled by min(1,g) so it shrinks
    # under decline like protect/baseline do (see docstring).
    exposed = ff_s > epf
    term1_scale = np.minimum(g, 1.0)
    avoid_exp[exposed] = ff_s[exposed] * population[exposed] * term1_scale[exposed]

    # Growth terms: only where the scenario exceeds the design level the
    # growth was steered out of. Marginal exceedance fraction, same
    # /(1-apf) normalization as compute_retreat.
    exceeds = ff_s > apf
    safe_frac = np.maximum(1e-10, 1.0 - apf)
    marginal_frac = np.minimum(1.0, (ff_s - apf) / safe_frac)

    # Organic-growth term: the GROWTH INCREMENT (max(0, g-1), matching
    # compute_avoid_redirected's own convention so this term also vanishes
    # at growth_factor <= 1) of population already outside the design
    # floodplain, exposed to the marginal risk.
    organic_grown = population * np.maximum(0.0, 1.0 - apf) * np.maximum(0.0, g - 1.0)
    avoid_exp[exceeds] += marginal_frac[exceeds] * organic_grown[exceeds]

    # Redirected-growth term (growth that would have occurred in the
    # floodplain, resettled elsewhere in the country).
    avoid_exp[exceeds] += marginal_frac[exceeds] * redirected[exceeds]

    return avoid_exp


# ── EAI and country-level aggregation ────────────────────────────────────────

def compute_country_eai(
    exposure_per_rp_slr: dict[tuple[int, str], np.ndarray],
    return_periods: list[int],
    slr_scenarios: list[str],
    geo_ids: np.ndarray,
    iso_lookup: dict[int, str],
    iso_index: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Aggregate per-cell exposure grids to per-country EAI for each SLR scenario.

    Uses np.bincount for O(H×W) aggregation per (RP, SLR) instead of scanning
    the grid once per country — avoids O(n_countries × H×W × n_rp × n_slr) cost.

    Parameters
    ----------
    exposure_per_rp_slr : dict[tuple[int, str], np.ndarray]
        (H, W) exposure grids keyed by (return_period, slr_name). A missing
        key is treated as all-zero exposure.
    return_periods : list[int]
        Simulated return periods (years), used for the trapezoidal EAI integration.
    slr_scenarios : list[str]
        SLR scenario names to produce output columns for.
    geo_ids : np.ndarray
        (H, W) int32 geogunit ID per cell; -1 = no geogunit.
    iso_lookup : dict[int, str]
        Geogunit ID -> ISO-3 country code.
    iso_index : tuple[list[str], np.ndarray, np.ndarray] or None, optional
        Precomputed `_build_iso_index(geo_ids, iso_lookup)` result. Pass this
        when calling many times for the same (geo_ids, iso_lookup) (e.g. once
        per baseline/protect/retreat/avoid scenario). Default None.

    Returns
    -------
    pd.DataFrame
        Index = ISO code, columns = SLR scenario name, values = EAI.
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
    """Compute `compute_country_eai` for exposure values already gathered per cell.

    `weights_per_rp_slr[key]` holds the exposure of exactly the cells whose
    country index is `iso_idx` (same length, same order) - i.e. what
    compute_country_eai reads as `grid.ravel()[cell_idx]`, optionally with
    cells of exactly zero exposure left out. Produces the identical DataFrame
    (same values to the last bit, same index/column order): np.bincount adds
    weights per bin sequentially in cell order starting from +0.0, so the
    same weights in the same order give the same sums, and dropping exact
    zeros never changes a sum. A missing key counts as all-zero exposure,
    exactly like compute_country_eai's zeros-grid fallback.

    Parameters
    ----------
    weights_per_rp_slr : dict[tuple[int, str], np.ndarray]
        1-D exposure-weight arrays (parallel to `iso_idx`) keyed by
        (return_period, slr_name). A missing key counts as all-zero exposure.
    return_periods : list[int]
        Simulated return periods (years), used for the trapezoidal EAI integration.
    slr_scenarios : list[str]
        SLR scenario names to produce output columns for.
    iso_list : list[str]
        Sorted ISO codes present in the study area.
    iso_idx : np.ndarray
        Per-cell index into `iso_list`, parallel to each `weights_per_rp_slr` array.

    Returns
    -------
    pd.DataFrame
        Index = ISO code, columns = SLR scenario name, values = EAI.
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
    """Linearly interpolate per-ISO EAI from discrete simulated SLR levels to a dense grid.

    Aqueduct is only ever run at the discrete SLR scenarios
    (boundary_conditions.slr_scenarios + adaptation.slr_intensities); this
    linearly interpolates each country's EAI-vs-SLR curve (already computed
    at those discrete levels, e.g. by compute_country_eai) to fill in the
    gaps at `target_slr_mm` — sea level (and EAI along with it) is assumed
    to change linearly between two modelled SLR steps. Used to densify the
    base EAI-vs-SLR curve onto visualization.slr_interp's grid for the
    burning-ember growth-matrix file.

    Parameters
    ----------
    eai_df : pd.DataFrame
        Per-ISO EAI at the discrete simulated SLR levels. Columns must be SLR
        scenario names (e.g. "SLR_0", "SLR_200", ...) in the same ascending
        order as `slr_mm_values` (see `config_utils.merged_slr_scenarios`).
    slr_mm_values : list[float]
        Strictly increasing SLR levels (mm) matching `eai_df`'s columns, 1:1.
    target_slr_mm : np.ndarray
        SLR levels (mm) to interpolate EAI onto.

    Returns
    -------
    pd.DataFrame
        Indexed like `eai_df`; one column "SLR_{mm}" per `target_slr_mm` entry
        (mm rounded to the nearest integer). Values clipped to >= 0.

    Raises
    ------
    ValueError
        If `slr_mm_values` has fewer than 2 entries or is not strictly
        increasing, or its length does not match `eai_df`'s column count.
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
    """Resolve per-country EAI at each (SSP, year) to a single ready-to-plot value.

    For each (SSP, year): linearly interpolate `slr_traj` to that year's
    global-mean SLR (SLR is assumed to rise linearly between the two nearest
    modelled trajectory years), linearly interpolate `eai_df`'s discrete
    SLR-EAI curve to that SLR, then scale by that country's SSP/year
    population growth factor. Both interpolation steps happen here, at
    file-creation time, rather than being left to plotting code.

    Parameters
    ----------
    eai_df : pd.DataFrame
        Per-ISO EAI at the discrete simulated SLR levels, columns matching
        `slr_mm_values` 1:1 in the same order.
    slr_mm_values : list[float]
        Strictly increasing SLR levels (mm) matching `eai_df`'s columns.
    growth_df : pd.DataFrame
        Per-country SSP/year population growth factors (see `population_growth`).
    ssps : list[str]
        SSP scenario names to resolve; entries missing from `slr_traj.columns`
        are skipped.
    years : list[int]
        Target years to resolve.
    slr_traj : pd.DataFrame
        Global-mean SLR trajectory in mm, indexed by year, one column per SSP
        (see `visualization.load_slr_trajectories`).

    Returns
    -------
    pd.DataFrame
        Indexed by ISO; one column "EAI_{SSP}_{year}" per resolved (SSP, year)
        pair (one value per country per column, not a per-SLR grid).

    Raises
    ------
    ValueError
        If `slr_mm_values` has fewer than 2 entries or is not strictly
        increasing, or its length does not match `eai_df`'s column count.
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
    """Scale per-country base EAI by a generic (scenario-neutral) growth-rate axis.

    Unlike a per-country SSP projection, the growth factor here is the same
    single scalar for every ISO/SLR cell. Valid for baseline/protect/retreat
    because their exposure is linear in a uniform population scaling (see
    compute_exposure_analysis.py's growth-matrix section for the proof); avoid
    is NOT linear in growth and must be recomputed directly instead of scaled
    this way.

    Parameters
    ----------
    eai_df : pd.DataFrame
        Per-ISO base EAI, columns = SLR scenario names.
    growth_rates : np.ndarray
        Fractional population growth rates to apply (e.g. -0.5 = -50%, 1.5 = +150%).

    Returns
    -------
    pd.DataFrame
        Indexed by ISO; one column ``EAI_{SLR}_g{pct}`` per (SLR, growth_rates
        entry), e.g. ``EAI_SLR_0_g-50``, ``EAI_SLR_1400_g150``
        (pct = round(g * 100)).
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
