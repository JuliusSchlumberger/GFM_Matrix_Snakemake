"""Visualization functions for GFM coastal-flood exposure statistics.

Chart functions operate on compute_exposure_analysis.py's per-scenario CSVs,
read directly with plain pd.read_csv (each has a simple, single column
convention — no shared loader needed):

  File 1 — exposure_{label}_base.csv           cols = plain SLR_{mm} names
  File 2 — exposure_{label}_growth_matrix.csv   cols = EAI_SLR_{mm}_g{pct}
           (parse with load_growth_matrix_csv)
  File 3 — exposure_{label}_ssp.csv             cols = EAI_{SSP}_{year}

The functions here are organized in three layers:
  1. Data preparation — parse growth-matrix CSVs, aggregate, build lookups.
  2. Single-chart functions — each returns a matplotlib Figure.
  3. Batch helpers — iterate over countries / continents / scenarios.

Usage (standalone, after the Snakemake pipeline has completed):

    from visualization import (
        load_growth_matrix_csv,
        build_growth_matrix_from_grid,
        plot_burning_ember,
        plot_world_map,
    )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import pchip_interpolate

from config_utils import retry_transient_io


# ── Constants / defaults ──────────────────────────────────────────────────────

_DEFAULT_SLR_COLORS = {
    "SSP1": "#1f77b4",
    "SSP2": "#2ca02c",
    "SSP3": "#ff7f0e",
    "SSP5": "#d62728",
}

_DEFAULT_MEASURE_COLORS = {
    "baseline": "#7f7f7f",
    "protect": "#1f77b4",
    "retreat": "#9467bd",
    "avoid": "#2ca02c",
    "protect_avoid": "#17becf",
}


# ── Continents ────────────────────────────────────────────────────────────────
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

# ISO-3 codes the exposure outputs use but Natural Earth's lowres country table
# lacks (small islands/territories, mostly) - Natural Earth's own continent
# convention: Caribbean and Central America under North America, Pacific
# islands under Oceania. Open-ocean/Antarctic territories with no population
# (BVT, HMD, IOT) are left out of every continent.
_ISO_CONTINENT_OVERRIDES = {
    **dict.fromkeys(["ABW", "AIA", "ATG", "BES", "BLM", "BMU", "BRB", "CUW", "CYM", "DMA", "GLP", "GRD", "KNA",
                     "LCA", "MAF", "MSR", "MTQ", "SPM", "SXM", "TCA", "VCT", "VGB", "VIR", "XCL"], "North America"),
    **dict.fromkeys(["GUF", "SGS"], "South America"),
    **dict.fromkeys(["ALA", "AND", "FRO", "GGY", "GIB", "IMN", "JEY", "LIE", "MCO", "MLT", "SJM", "SMR", "VAT",
                     "XKO"], "Europe"),
    **dict.fromkeys(["COM", "CPV", "MUS", "MYT", "REU", "SHN", "STP", "SYC"], "Africa"),
    **dict.fromkeys(["BHR", "HKG", "MAC", "MDV", "SGP", "XAD", "XNC", "PIS", "SP-", "XCA"], "Asia"),
    **dict.fromkeys(["ASM", "CCK", "COK", "CXR", "FSM", "GUM", "KIR", "MHL", "MNP", "NFK", "NIU", "NRU", "PCN",
                     "PLW", "PYF", "TKL", "TON", "TUV", "UMI", "WLF", "WSM"], "Oceania"),
}


def iso_to_continent(isos) -> dict[str, str]:
    """{ISO-3: continent} for every ISO in `isos` that belongs to one of
    CONTINENTS - Natural Earth lowres's own `continent` field, plus
    _ISO_CONTINENT_OVERRIDES for the codes that table lacks. ISOs with no
    continent (open ocean, Antarctica, unknown codes) are left out."""
    import warnings

    import geopandas as gpd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    lookup = {str(r.iso_a3).strip(): r.continent for r in world.itertuples() if str(r.iso_a3).strip() not in ("-99", "")}
    lookup.update(_ISO_CONTINENT_OVERRIDES)
    return {iso: lookup[iso] for iso in isos if lookup.get(iso) in CONTINENTS}


def draw_ember_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    slr_mm: np.ndarray,
    growth_rates: np.ndarray,
    cmap: str = "YlOrRd",
    vmax: float | None = None,
    n_contours: int = 6,
    contour_labels: bool = True,
    fontsize: float = 10,
    log_vmin: float | None = None,
    label_fontsize: float | None = None,
):
    """The burning-ember heatmap + contours on an existing axis - the
    scenario-neutral impact matrix only (no SSP trajectories, no SLR
    uncertainty subplot; see plot_burning_ember for those). Returns the image
    for a colorbar.

    `log_vmin`: logarithmic colour scale from log_vmin to vmax (values below
    are drawn at log_vmin) with contours at 1-2-5 steps per decade, instead of
    the default linear scale from 0 with `n_contours` evenly spaced contours."""
    slr_x = np.asarray(slr_mm, dtype=float) / 1000.0
    gr_y = np.asarray(growth_rates, dtype=float) * 100.0
    vmax_c = vmax if vmax is not None else (float(np.nanmax(matrix)) or 1.0)
    extent = [slr_x[0], slr_x[-1], gr_y[0], gr_y[-1]]
    if log_vmin is not None:
        shown = np.clip(matrix, log_vmin, None)
        im = ax.imshow(shown, origin="lower", aspect="auto", extent=extent, cmap=cmap,
                       norm=mcolors.LogNorm(vmin=log_vmin, vmax=vmax_c), interpolation="bilinear")
        decades = range(int(np.floor(np.log10(log_vmin))), int(np.ceil(np.log10(vmax_c))) + 1)
        levels = [m * 10.0 ** d for d in decades for m in (1, 2, 5) if log_vmin < m * 10.0 ** d < vmax_c]
        if levels and np.nanmax(matrix) > levels[0]:
            ax.contour(slr_x, gr_y, shown, levels=levels, colors="k", linewidths=0.7, alpha=0.6)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.tick_params(axis="both", labelsize=max(6, fontsize - 1))
        return im
    im = ax.imshow(matrix, origin="lower", aspect="auto", extent=extent,
                   cmap=cmap, vmin=0, vmax=vmax_c, interpolation="bilinear")
    levels = np.linspace(0, vmax_c, n_contours + 1)[1:]
    if np.nanmax(matrix) > levels[0]:
        cs = ax.contour(slr_x, gr_y, matrix, levels=levels, colors="k", linewidths=0.8, alpha=0.7)
        if contour_labels:
            for lbl in ax.clabel(cs, inline=True, fontsize=label_fontsize or max(6, fontsize - 3),
                                 fmt=lambda v: f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} k"):
                lbl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=max(6, fontsize - 1))
    return im


def ssp_rcp_label(ssp: str, ssp_rcp_codes: dict[str, int] | None) -> str:
    """'SSP2' -> 'SSP2-RCP4.5' using the CMIP6 code's last two digits as the
    RCP value (naming convention: ssp245 = SSP2 + RCP4.5, ssp585 = SSP5 +
    RCP8.5). Falls back to the bare SSP label if no code is configured.
    Used by plot_timeseries.py's panel titles."""
    code = (ssp_rcp_codes or {}).get(ssp)
    if code is None:
        return ssp
    rcp = (code % 100) / 10.0
    return f"{ssp}-RCP{rcp:.1f}"


# ── 1. Data preparation ───────────────────────────────────────────────────────

def load_growth_matrix_csv(
    csv_path: str | Path, window: dict | None = None,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Load a scenario-neutral growth-matrix CSV (exposure_*_growth_matrix.csv).

    Columns are named ``EAI_{SLR}_g{pct}`` (e.g. ``EAI_SLR_250_g-50``,
    ``EAI_SLR_1400_g150``), written by compute_exposure_analysis.py's
    apply_growth_rates_to_eai / avoid growth-matrix worker. Unlike the SSP-
    growth columns in the main exposure_*.csv files, the growth axis here is
    a single scenario-neutral scalar applied uniformly to every ISO.

    Returns:
        df:            index=ISO, columns=MultiIndex (slr_mm, growth_frac),
                        sorted ascending on both levels.
        slr_mm_values: sorted unique SLR levels (mm) found in the columns.
        growth_fracs:  sorted unique growth fractions (e.g. -0.5 … 1.5).

    `window` (e.g. visualization.ember_display: slr_max_mm / growth_min /
    growth_max, each optional) keeps only that part of the matrix - for
    figures that show less than the stored matrix covers.
    """
    import re

    raw = pd.read_csv(csv_path, index_col=0)
    pattern = re.compile(r"^EAI_(SLR_\d+)_g(-?\d+)$")
    parsed: dict[tuple[float, float], str] = {}
    for col in raw.columns:
        m = pattern.match(col)
        if not m:
            continue
        slr_mm = float(m.group(1).split("_")[1])
        g_frac = int(m.group(2)) / 100.0
        parsed[(slr_mm, g_frac)] = col

    slr_mm_values = sorted({k[0] for k in parsed})
    growth_fracs = sorted({k[1] for k in parsed})
    if window:
        eps = 1e-9
        slr_mm_values = [s for s in slr_mm_values if s <= window.get("slr_max_mm", np.inf) + eps]
        growth_fracs = [g for g in growth_fracs
                        if window.get("growth_min", -np.inf) - eps <= g <= window.get("growth_max", np.inf) + eps]
    df = pd.DataFrame(
        {(slr, g): raw[parsed[(slr, g)]] for slr in slr_mm_values for g in growth_fracs},
        index=raw.index,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["slr_mm", "growth_frac"])
    df.index.name = raw.index.name or "ISO"
    return df, slr_mm_values, growth_fracs


def build_growth_matrix_from_grid(
    eai_grid: pd.DataFrame,
    slr_mm_values: list[float],
    growth_fracs: list[float],
) -> np.ndarray:
    """Build a (len(growth_fracs), len(slr_mm_values)) EAI matrix from a
    scenario-neutral growth-matrix table (see load_growth_matrix_csv), summed
    over whichever ISO rows are already present in `eai_grid` (subset it to
    one country / a region's ISOs / everything before calling, as needed).

    Growth is NOT assumed to scale EAI linearly — each growth_fracs column
    was independently computed (this matters for avoid, whose exposure is
    genuinely non-linear in growth; harmless for baseline/protect/retreat,
    which are linear anyway). No SLR interpolation happens here:
    compute_exposure_analysis.py already linearly interpolates each
    scenario's base EAI-vs-SLR curve onto the dense visualization.slr_interp
    grid at file-creation time, so `slr_mm_values` (from load_growth_matrix_csv)
    is already the dense axis — this is a plain aggregate + reshape.
    """
    totals = eai_grid.sum(axis=0)  # MultiIndex (slr_mm, growth_frac) -> summed EAI
    # unstack (vectorized, C-level pivot) instead of one Python-level scalar
    # MultiIndex lookup per (growth_frac, slr_mm) cell - confirmed bit-identical
    # output, ~10x faster on a real grid (5719 cells: 58.6ms -> 6.0ms) - this
    # function is called once per (country, strategy/design) in every script
    # that builds per-country/per-continent pathways (plot_tipping_points.py,
    # compute_tipping_points_summary.py, compute_adaptation_pathways.py, ...),
    # so the per-call saving compounds into real wall-clock minutes at
    # continent scale (hundreds of countries x several strategy/design grids).
    pivot = totals.unstack(level="slr_mm")
    matrix = pivot.reindex(index=growth_fracs, columns=slr_mm_values).to_numpy(dtype=float)
    return np.clip(matrix, 0.0, None)


def load_slr_trajectories(csv_path: str | Path) -> pd.DataFrame:
    """Load the global-mean SLR trajectory CSV (output of extract_slr_trajectories.py).

    Expected columns: year (int) + one column per SSP (e.g. SSP1, SSP2, …) with
    global-median SLR in mm relative to the 2020 baseline.
    """
    df = pd.read_csv(csv_path, index_col="year")
    return df


# ── 2. Chart functions ────────────────────────────────────────────────────────

def draw_slr_uncertainty_band(
    ax: plt.Axes,
    slr_uncertainty: dict[str, pd.DataFrame],
    highlight_years: list[int],
    colors: dict[str, str],
    fontsize: float = 10,
) -> None:
    """The burning-ember "uncertainty in SLR projections" strip, factored out
    of plot_burning_ember so other figures (e.g. analysis/plot_tipping_points.py)
    can place the identical sub-panel under their own SLR-axis plot.

    One row per highlight year, every SSP's P17-P83 SLR error bar plotted
    side by side within that row (colour identifies the SSP, matching
    whatever legend the caller's own main panel uses - no per-row label).
    Only the left spine remains; the x-axis has no tick marks/labels of its
    own - it reads against whatever SLR axis sits directly above `ax`
    (shared via `sharex`), not as a separately boxed panel.

    Args:
        ax:               Target axes - must share its x-axis (SLR, metres)
                           with the panel above it.
        slr_uncertainty:  {SSP: pd.DataFrame(index=year, columns=[p17,p50,p83] in mm)}.
        highlight_years:  Years to draw a row for (e.g. [2050, 2100]).
        colors:           Per-SSP colour map.
        fontsize:         Matches the panel above it.
    """
    ssp_list = [ssp for ssp in colors if ssp in slr_uncertainty]
    n_unc = len(ssp_list)
    n_years = len(highlight_years)
    # Evenly spaced offsets within each year's unit-height row so each SSP's
    # error bar sits at its own y position instead of overlapping; a lone
    # SSP sits exactly on the tick (offset 0).
    offsets = np.linspace(-0.3, 0.3, n_unc) if n_unc > 1 else np.array([0.0])
    for yi, yr in enumerate(highlight_years):
        for ssp, offset in zip(ssp_list, offsets):
            unc = slr_uncertainty[ssp]
            if yr not in unc.index:
                continue
            color = colors[ssp]
            p17 = float(unc.loc[yr, "p17"]) / 1000.0
            p50 = float(unc.loc[yr, "p50"]) / 1000.0
            p83 = float(unc.loc[yr, "p83"]) / 1000.0
            ax.errorbar(
                p50, yi + offset, xerr=[[p50 - p17], [p83 - p50]],
                fmt="o", color=color, markersize=5, capsize=3, linewidth=1.5,
            )
    ax.set_yticks(range(n_years))
    ax.set_yticklabels([str(yr) for yr in highlight_years], fontsize=max(6, fontsize - 3))
    ax.set_ylim(-0.5, max(n_years, 1) - 0.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.set_ylabel("Uncertainty in SLR projections", fontsize=fontsize)


def plot_burning_ember(
    matrix: np.ndarray,
    slr_interp_mm: np.ndarray,
    growth_rates: np.ndarray,
    title: str = "",
    slr_trajectories: pd.DataFrame | None = None,
    ssp_growth: dict[str, pd.Series] | None = None,
    slr_uncertainty: dict[str, pd.DataFrame] | None = None,
    highlight_years: list[int] | None = None,
    cmap: str = "YlOrRd",
    vmax: float | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (8, 6),
    unit_label: str = "EAI (people)",
    ssp_colors: dict[str, str] | None = None,
    n_contours: int = 6,
    fontsize: float = 10,
    diverging_center: float | None = None,
    diverging_vmin: float | None = None,
    contour_vmax: float | None = None,
) -> plt.Figure:
    """Burning-ember impact matrix: EAI as a function of SLR × population growth.

    Two stacked subplots sharing the SLR (x) axis:
      1. The impact-matrix heatmap + contours + SSP trajectory overlays.
      2. A single uncertainty subplot, one y-tick per highlight year (not
         per SSP): each year's row shows every SSP's P17-P83 SLR error bar
         side by side, offset slightly around that year's tick position so
         they don't overlap. Colour (matching subplot 1's legend), not a
         per-row label, identifies which SSP each error bar belongs to.
         Only the left spine remains (top/bottom/right hidden) and the
         x-axis has no tick marks or labels of its own (just the vertical
         gridlines, read against the heatmap's x-axis above) - it reads as
         one continuous panel, not a separately boxed one. The
         "Uncertainty in SLR projections" annotation (rotated to read
         upward) is this subplot's y-axis label directly.

    Both subplots live in the same GridSpec column so their plot areas are
    IDENTICAL widths regardless of the heatmap's colorbar (which occupies
    its own dedicated GridSpec column, row 0 only) — attaching a colorbar
    via the simpler `fig.colorbar(im, ax=ax)` shrinks only ax's width,
    breaking x-axis alignment with any other sharex'd subplot.

    Args:
        matrix:             2-D array (n_growth_rates, n_slr_interp).
        slr_interp_mm:      Dense SLR axis values in mm.
        growth_rates:       Growth-rate axis values (fractions, e.g. -0.5 → 1.5).
        title:              Figure title (country name / continent / scenario).
        slr_trajectories:   Per-SSP global-mean (median) SLR in mm (index=year, cols=SSP*).
        ssp_growth:         {SSP: pd.Series(index=year, values=fraction growth rate)}.
        slr_uncertainty:    {SSP: pd.DataFrame(index=year, columns=[p17,p50,p83] in mm)} —
                             the uncertainty subplot is created only if this is non-empty,
                             with one error bar per key present here at each highlight year.
        highlight_years:    Years to mark on SSP trajectories (e.g. [2050, 2100]).
        cmap:               Matplotlib colormap name.
        vmax:               Colour scale maximum (auto if None).
        dpi:                Figure resolution.
        figsize:            Size (inches) of the heatmap row; total figure height
                             grows with the number of highlight_years.
        unit_label:         Colourbar label.
        ssp_colors:         Per-SSP colour overrides.
        n_contours:         Number of labelled EAI contour lines drawn on the heatmap.
        fontsize:           Base font size (points); axis/tick labels use this directly,
                             the title and colourbar label scale up from it, contour
                             labels and legend scale down from it.
        diverging_center:   If given, colour the heatmap on a diverging scale (RdBu_r,
                             blue=below/red=above) centred on this EAI value instead of
                             the default sequential scale (cmap, vmin=0). Useful for a
                             growth-only panel where population decline can pull EAI
                             below its growth=0% value - pass that value here to make
                             the reduction read as a distinct colour, not just a lighter
                             shade of the same ramp.
        diverging_vmin:     Lower bound for the diverging norm (only used when
                             diverging_center is set). Defaults to this panel's own
                             matrix minimum (auto). Pass a fixed value shared across
                             several plot_burning_ember() calls to give them all an
                             identical colour-to-value mapping instead of each panel
                             auto-scaling its own low end.
        contour_vmax:       Upper bound for contour LEVEL placement, independent of the
                             colour scale's vmax. Pass this when several panels share one
                             fixed vmax for cross-panel colour comparability but have very
                             different value ranges themselves - without it, a low-range
                             panel's contour levels are spaced across the shared (much
                             larger) vmax and mostly fall outside its own data, leaving
                             few or no contour lines to show its internal shape. Defaults
                             to the colour scale's own vmax (current behaviour).
    """
    colors_ = ssp_colors or _DEFAULT_SLR_COLORS
    if highlight_years is None:
        highlight_years = [2050, 2100]
    slr_interp_mm = np.asarray(slr_interp_mm, dtype=float)
    growth_rates = np.asarray(growth_rates, dtype=float)

    ssp_list = [ssp for ssp in colors_ if slr_uncertainty and ssp in slr_uncertainty]
    n_unc = len(ssp_list)
    n_years = len(highlight_years)

    fig = plt.figure(figsize=(figsize[0], figsize[1] + 0.7 * n_years), dpi=dpi)
    outer_rows = 2 if n_unc > 0 else 1
    outer_height_ratios = [3, max(0.5, 0.6 * n_years)] if n_unc > 0 else [3]

    gs = fig.add_gridspec(
        outer_rows, 2,
        height_ratios=outer_height_ratios,
        width_ratios=[30, 1],
        hspace=0.2, wspace=0.05,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    unc_ax = fig.add_subplot(gs[1, 0], sharex=ax) if n_unc > 0 else None

    slr_x = slr_interp_mm / 1000.0  # mm → m for axis label
    gr_y = growth_rates * 100.0      # fraction → percent

    extent = [slr_x[0], slr_x[-1], gr_y[0], gr_y[-1]]
    if vmax is None:
        vmax_c = float(np.nanmax(matrix)) if np.any(matrix > 0) else 1.0
    else:
        vmax_c = vmax

    if diverging_center is not None:
        vmin_c = diverging_vmin if diverging_vmin is not None else float(np.nanmin(matrix))
        vmin_c = min(vmin_c, diverging_center - 1e-9)
        vmax_c = max(vmax_c, diverging_center + 1e-9)
        norm = mcolors.TwoSlopeNorm(vmin=vmin_c, vcenter=diverging_center, vmax=vmax_c)
        im = ax.imshow(
            matrix, origin="lower", aspect="auto",
            extent=extent, cmap="RdBu_r", norm=norm,
            interpolation="bilinear",
        )
    else:
        im = ax.imshow(
            matrix, origin="lower", aspect="auto",
            extent=extent,
            cmap=cmap, vmin=0, vmax=vmax,
            interpolation="bilinear",
        )
    cb = fig.colorbar(im, cax=cax, label=unit_label)
    cb.ax.tick_params(labelsize=max(6, fontsize - 2))
    cb.set_label(unit_label, fontsize=fontsize)

    # Contour lines, labelled with their EAI value. Level placement can be
    # decoupled from the colour scale's vmax via contour_vmax (see docstring)
    # so a shared cross-panel colour scale doesn't starve a lower-range
    # panel of its own meaningful contour lines.
    contour_lo = diverging_center if diverging_center is not None else 0.0
    contour_hi = contour_vmax if contour_vmax is not None else vmax_c
    levels = np.linspace(contour_lo, contour_hi, n_contours + 1)[1:]
    cs = ax.contour(slr_x, gr_y, matrix, levels=levels, colors="k", linewidths=1.1, alpha=0.75)
    clabels = ax.clabel(cs, inline=True, fontsize=max(7, fontsize - 1), fmt=lambda v: f"{v:,.0f}")
    for lbl in clabels:
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # SSP trajectory overlays — line stops at the last highlight year (e.g. 2100),
    # even if the underlying trajectory/growth data extends further.
    if slr_trajectories is not None and ssp_growth is not None:
        line_cutoff = max(highlight_years)
        for ssp, color in colors_.items():
            if ssp not in slr_trajectories.columns:
                continue
            slr_traj = slr_trajectories[ssp]
            gr_traj = ssp_growth.get(ssp, pd.Series(dtype=float))
            years_common = slr_traj.index.intersection(gr_traj.index)
            years_common = years_common[years_common <= line_cutoff]
            if years_common.empty:
                continue
            xs = slr_traj[years_common].to_numpy() / 1000.0
            ys = (gr_traj[years_common].to_numpy() - 1.0) * 100.0
            ax.plot(xs, ys, "-", color=color, linewidth=2, label=ssp, zorder=5)
            for yr in highlight_years:
                if yr in years_common:
                    xi = float(slr_traj[yr]) / 1000.0
                    yi = (float(gr_traj[yr]) - 1.0) * 100.0
                    ax.plot(xi, yi, "o", color=color, markersize=7, zorder=6)
                    ax.annotate(str(yr), (xi, yi), textcoords="offset points",
                                xytext=(4, 4), fontsize=max(6, fontsize - 3), color=color)
        ax.legend(loc="upper left", fontsize=max(6, fontsize - 2), title="SSP")

    ax.set_ylabel("Population growth relative to 2020 (%)", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 2)
    ax.tick_params(axis="both", labelsize=max(6, fontsize - 1))
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)

    if unc_ax is not None:
        draw_slr_uncertainty_band(unc_ax, slr_uncertainty, highlight_years, colors_, fontsize=fontsize)

    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel("Global mean SLR (m)", fontsize=fontsize)
    return fig


def plot_timeseries(
    eai_dict: dict[str, pd.DataFrame],
    slr_trajectories: pd.DataFrame,
    ssp_growth: dict[str, pd.Series],
    years: list[int],
    slr_mm_values: list[float],
    entity_label: str = "",
    measure_colors: dict[str, str] | None = None,
    ssp_colors: dict[str, str] | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (10, 6),
    unit_label: str = "EAI (people)",
) -> plt.Figure:
    """Time-series plot of EAI along SSP SLR trajectories for each adaptation measure.

    For each (SSP, year), the SLR level is read from `slr_trajectories`, the EAI
    for that SLR is interpolated from the discrete scenario grid, and optionally
    scaled by the corresponding population growth factor.

    Args:
        eai_dict:          {measure_label: EAI DataFrame (index=geogunit, cols=SLR scenarios)}.
        slr_trajectories:  Global-median SLR in mm (index=year, cols=SSP names).
        ssp_growth:        {SSP: pd.Series(index=year, values=growth factor)}.
        years:             Years to evaluate (e.g. list(range(2020, 2151, 10))).
        slr_mm_values:     Discrete SLR levels (mm) corresponding to eai_dict columns.
        entity_label:      Title suffix.
        measure_colors:    Colour overrides per measure.
        ssp_colors:        Colour overrides per SSP.
        dpi:               Figure resolution.
        figsize:           Figure size.
        unit_label:        Y-axis label.
    """
    m_colors = measure_colors or _DEFAULT_MEASURE_COLORS
    s_colors = ssp_colors or _DEFAULT_SLR_COLORS

    x_known = np.array(slr_mm_values, dtype=float)

    ssps = [s for s in slr_trajectories.columns if s in (ssp_growth or {})]
    measures = list(eai_dict.keys())

    fig, axes = plt.subplots(1, len(ssps), figsize=figsize, dpi=dpi, sharey=True)
    if len(ssps) == 1:
        axes = [axes]

    for ax, ssp in zip(axes, ssps):
        color_ssp = s_colors.get(ssp, "#444444")
        for measure in measures:
            df = eai_dict[measure]
            # Sum geogunits → series indexed by SLR scenario name
            eai_totals = np.array([df[slr].sum() if slr in df.columns else 0.0
                                   for slr in [f"SLR_{int(mm)}" for mm in slr_mm_values]])

            time_vals = []
            for yr in years:
                if yr not in slr_trajectories.index:
                    continue
                slr_at_yr = float(slr_trajectories.loc[yr, ssp])
                # Interpolate EAI at this SLR
                y_known = eai_totals.copy()
                x_ext = x_known.copy()
                if x_ext[0] > 0:
                    x_ext = np.concatenate([[0.0], x_ext])
                    y_known = np.concatenate([[0.0], y_known])
                eai_interp = float(np.clip(
                    pchip_interpolate(x_ext, y_known, np.array([slr_at_yr])), 0, None
                )[0])
                # Apply population growth
                g = 1.0
                if ssp in ssp_growth and yr in ssp_growth[ssp].index:
                    g = float(ssp_growth[ssp][yr])
                time_vals.append((yr, eai_interp * g))

            if not time_vals:
                continue
            yrs_plot, eai_plot = zip(*time_vals)
            lw = 2.5 if measure == "baseline" else 1.8
            ls = "-" if measure == "baseline" else ("--" if "protect" in measure else
                  (":" if "retreat" in measure else "-."))
            col = m_colors.get(measure.split("_")[0], "#999999")
            ax.plot(yrs_plot, eai_plot, linestyle=ls, linewidth=lw, color=col,
                    label=measure, alpha=0.9)

        ax.set_title(ssp, color=color_ssp, fontsize=10)
        ax.set_xlabel("Year", fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(unit_label, fontsize=10)
    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", fontsize=8, bbox_to_anchor=(0.01, 0.97))
    fig.suptitle(f"EAI time-series — {entity_label}", fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


def build_geo109_to_iso_lookup(
    geo109_nc_path: str | Path,
    geo107_nc_path: str | Path,
    flopros_xlsx_path: str | Path,
    subsample: int = 10,
) -> dict[int, str]:
    """Build a geogunit_109 ID → ISO-3 country code lookup table.

    Overlaps the geogunit_109 country raster with the geogunit_107 sub-national
    raster (both at ~30 arcsec), then maps each 109-unit to the ISO code of the
    most-common 107-unit within it (via the FLOPROS table).

    Args:
        geo109_nc_path:    Path to geogunit_109_all.nc.
        geo107_nc_path:    Path to geogunit_107_all.nc.
        flopros_xlsx_path: Path to FLOPROS_NL_geogunit_107.xlsx (has ISO column).
        subsample:         Read every Nth pixel to reduce memory; default 10 (100×
                           fewer pixels at the cost of some spatial precision).

    Returns:
        Dict mapping each geogunit_109 integer ID → ISO-3 string.
    """
    flopros = pd.read_excel(flopros_xlsx_path, index_col=0, header=0)
    if "ISO" not in flopros.columns:
        raise ValueError("FLOPROS table must have an 'ISO' column.")

    geo107_to_iso: dict[int, str] = {}
    for gid, row in flopros.iterrows():
        iso = row.get("ISO")
        if pd.notna(iso):
            geo107_to_iso[int(gid)] = str(iso)

    with retry_transient_io(rasterio.open, f"netcdf:{geo109_nc_path}:Geogunits") as src109, \
         retry_transient_io(rasterio.open, f"netcdf:{geo107_nc_path}:Geogunits") as src107:

        height = src109.height
        width = src109.width

        from collections import Counter
        geo109_iso_votes: dict[int, Counter] = {}

        step = int(subsample)
        for row_start in range(0, height, max(1, height // 200)):
            row_end = min(row_start + max(1, height // 200), height)
            window = rasterio.windows.Window(0, row_start, width, row_end - row_start)
            arr109 = src109.read(1, window=window)[::step, ::step].ravel()
            arr107 = src107.read(1, window=window)[::step, ::step].ravel()

            mask = (arr109 > 0) & (arr107 > 0)
            for g109, g107 in zip(arr109[mask], arr107[mask]):
                g109_i, g107_i = int(g109), int(g107)
                iso = geo107_to_iso.get(g107_i)
                if iso:
                    if g109_i not in geo109_iso_votes:
                        geo109_iso_votes[g109_i] = Counter()
                    geo109_iso_votes[g109_i][iso] += 1

    return {gid: ctr.most_common(1)[0][0]
            for gid, ctr in geo109_iso_votes.items() if ctr}


def plot_world_map(
    eai_by_iso: pd.Series,
    geo109_nc_path: str | Path,
    geo109_to_iso: dict[int, str],
    title: str = "",
    cmap: str = "YlOrRd",
    vmax: float | None = None,
    dpi: int = 200,
    figsize: tuple[float, float] = (14, 7),
    unit_label: str = "EAI (people)",
    lat_range: tuple[float, float] = (-60, 85),
    downsample: int = 4,
) -> plt.Figure:
    """Raster choropleth world map coloured by per-country EAI.

    Each pixel of the geogunit_109 raster is coloured by the EAI of the country
    it belongs to.  No external vector shapefile required.

    Args:
        eai_by_iso:       pd.Series indexed by ISO-3 country code.
        geo109_nc_path:   Path to geogunit_109_all.nc.
        geo109_to_iso:    Dict {geogunit_109_id → ISO-3} (from build_geo109_to_iso_lookup).
        title:            Figure title.
        cmap:             Matplotlib colormap name.
        vmax:             Colour scale maximum (auto if None).
        dpi:              Figure resolution.
        figsize:          Figure size in inches.
        unit_label:       Colorbar label.
        lat_range:        (min_lat, max_lat) to clip the display.
        downsample:       Read every Nth pixel row/col to reduce memory and draw time.
    """
    # Build ISO→EAI lookup
    iso_to_eai = eai_by_iso.to_dict()

    with retry_transient_io(rasterio.open, f"netcdf:{geo109_nc_path}:Geogunits") as src:
        transform = src.transform
        height, width = src.height, src.width

        # Determine lat-clipping row indices
        lat_max_row = max(0, int((90.0 - lat_range[1]) / abs(transform.e)))
        lat_min_row = min(height, int((90.0 - lat_range[0]) / abs(transform.e)))

        window = rasterio.windows.Window(
            0, lat_max_row, width, lat_min_row - lat_max_row
        )
        geo109 = src.read(1, window=window, out_dtype="int32")

    # Downsample for display
    ds = int(downsample)
    geo109_ds = geo109[::ds, ::ds]

    # Build coloured array: invalid → NaN
    eai_grid = np.full(geo109_ds.shape, np.nan, dtype=float)
    for g109_id, iso in geo109_to_iso.items():
        eai_val = iso_to_eai.get(iso, np.nan)
        if not np.isnan(eai_val):
            eai_grid[geo109_ds == g109_id] = eai_val

    # Extent for imshow
    left = transform.c
    right = transform.c + width * transform.a
    top = transform.f + lat_max_row * transform.e
    bottom = transform.f + lat_min_row * transform.e
    extent = [left, right, bottom, top]

    norm = mcolors.Normalize(vmin=0,
                              vmax=vmax if vmax else float(np.nanmax(eai_grid)) or 1.0)
    cmap_obj = plt.get_cmap(cmap)
    cmap_obj.set_bad("#e0e0e0")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(eai_grid, extent=extent, origin="upper", cmap=cmap_obj, norm=norm,
                   aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label=unit_label, shrink=0.6,
                 orientation="horizontal", pad=0.02)

    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig
