"""Plot GFM coastal-flood exposure statistics from compute_exposure_analysis.py's CSVs.

Inputs: exposure_{label}_base.csv (cols SLR_{mm}), exposure_{label}_growth_matrix.csv
(cols EAI_SLR_{mm}_g{pct}, via `load_growth_matrix_csv`) and exposure_{label}_ssp.csv
(cols EAI_{SSP}_{year}).
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


CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]

# ISO-3 codes missing from Natural Earth lowres, using its continent convention
# (Caribbean/Central America -> North America, Pacific islands -> Oceania).
# Unpopulated BVT, HMD, IOT belong to no continent.
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
    """Map each ISO in `isos` to one of CONTINENTS (Natural Earth lowres + overrides).

    ISOs with no continent (open ocean, Antarctica, unknown codes) are left out.
    """
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
    """Draw the scenario-neutral burning-ember heatmap + contours on `ax`; return the image.

    `log_vmin` switches to a log colour scale from log_vmin to vmax (values below are
    drawn at log_vmin) with 1-2-5 contours per decade.
    """
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
    """Label 'SSP2' as 'SSP2-RCP4.5' from its CMIP6 code (ssp245 -> RCP4.5).

    Falls back to the bare SSP label if no code is configured.
    """
    code = (ssp_rcp_codes or {}).get(ssp)
    if code is None:
        return ssp
    rcp = (code % 100) / 10.0
    return f"{ssp}-RCP{rcp:.1f}"


def load_growth_matrix_csv(
    csv_path: str | Path, window: dict | None = None,
) -> tuple[pd.DataFrame, list[float], list[float]]:
    """Load a growth-matrix CSV (``EAI_{SLR}_g{pct}`` columns) into a sorted MultiIndex frame.

    Returns ``(df, slr_mm_values, growth_fracs)``; df columns are (slr_mm,
    growth_frac), growth as a fraction (-0.5 = -50%), applied uniformly to every ISO.
    `window` (slr_max_mm / growth_min / growth_max, each optional) crops the matrix.
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
    """Sum `eai_grid` over its ISO rows into a (growth_fracs, slr_mm_values) matrix.

    Each growth column was computed independently (avoid is non-linear in growth),
    and the SLR axis is already dense, so this is a plain aggregate + reshape.
    """
    totals = eai_grid.sum(axis=0)  # MultiIndex (slr_mm, growth_frac) -> summed EAI
    pivot = totals.unstack(level="slr_mm")
    matrix = pivot.reindex(index=growth_fracs, columns=slr_mm_values).to_numpy(dtype=float)
    return np.clip(matrix, 0.0, None)


def load_slr_trajectories(csv_path: str | Path) -> pd.DataFrame:
    """Load the global-mean SLR trajectory CSV (extract_slr_trajectories.py).

    Index year, one column per SSP: global-median SLR in mm relative to 2020.
    """
    df = pd.read_csv(csv_path, index_col="year")
    return df


def draw_slr_uncertainty_band(
    ax: plt.Axes,
    slr_uncertainty: dict[str, pd.DataFrame],
    highlight_years: list[int],
    colors: dict[str, str],
    fontsize: float = 10,
) -> None:
    """Draw the burning-ember "uncertainty in SLR projections" strip on `ax`.

    One row per highlight year with each SSP's P17-P83 error bar (`slr_uncertainty`:
    {SSP: DataFrame(index=year, columns p17/p50/p83 in mm)}); `ax` must share its
    x-axis (SLR, metres) with the panel above and has no x ticks of its own.
    """
    ssp_list = [ssp for ssp in colors if ssp in slr_uncertainty]
    n_unc = len(ssp_list)
    n_years = len(highlight_years)
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
    """Plot the burning-ember impact matrix: EAI over SLR x population growth.

    `matrix` is (n_growth_rates, n_slr_interp); `slr_interp_mm` in mm, `growth_rates`
    as fractions. Optional SSP trajectory overlays (`slr_trajectories` in mm,
    `ssp_growth` growth factors) stop at the last highlight year; an SLR-uncertainty
    strip is added below if `slr_uncertainty` is non-empty. The colorbar has its own
    GridSpec column so both panels keep identical widths for `sharex`.
    `diverging_center` switches to RdBu_r centred on that EAI (`diverging_vmin` fixes
    the low end across panels); `contour_vmax` places contour levels independently
    of the shared colour vmax.
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

    contour_lo = diverging_center if diverging_center is not None else 0.0
    contour_hi = contour_vmax if contour_vmax is not None else vmax_c
    levels = np.linspace(contour_lo, contour_hi, n_contours + 1)[1:]
    cs = ax.contour(slr_x, gr_y, matrix, levels=levels, colors="k", linewidths=1.1, alpha=0.75)
    clabels = ax.clabel(cs, inline=True, fontsize=max(7, fontsize - 1), fmt=lambda v: f"{v:,.0f}")
    for lbl in clabels:
        lbl.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

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
    """Plot EAI along each SSP's SLR trajectory per adaptation measure.

    EAI at the trajectory's SLR (mm) is PCHIP-interpolated over `slr_mm_values`
    (anchored at 0 EAI at 0 mm) and scaled by the SSP growth factor when available.
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
            eai_totals = np.array([df[slr].sum() if slr in df.columns else 0.0
                                   for slr in [f"SLR_{int(mm)}" for mm in slr_mm_values]])

            time_vals = []
            for yr in years:
                if yr not in slr_trajectories.index:
                    continue
                slr_at_yr = float(slr_trajectories.loc[yr, ssp])
                y_known = eai_totals.copy()
                x_ext = x_known.copy()
                if x_ext[0] > 0:
                    x_ext = np.concatenate([[0.0], x_ext])
                    y_known = np.concatenate([[0.0], y_known])
                eai_interp = float(np.clip(
                    pchip_interpolate(x_ext, y_known, np.array([slr_at_yr])), 0, None
                )[0])
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
    """Map each geogunit_109 ID to the ISO-3 of its most common geogunit_107 unit.

    Both rasters are ~30 arcsec; ISO comes from the FLOPROS table. `subsample` reads
    every Nth pixel per axis (10 = 100x fewer pixels).
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
    """Plot a raster choropleth world map of per-country EAI on the geogunit_109 grid.

    Assumes the raster's top edge is 90N (latitude rows from `lat_range`).
    """
    iso_to_eai = eai_by_iso.to_dict()

    with retry_transient_io(rasterio.open, f"netcdf:{geo109_nc_path}:Geogunits") as src:
        transform = src.transform
        height, width = src.height, src.width

        lat_max_row = max(0, int((90.0 - lat_range[1]) / abs(transform.e)))
        lat_min_row = min(height, int((90.0 - lat_range[0]) / abs(transform.e)))

        window = rasterio.windows.Window(
            0, lat_max_row, width, lat_min_row - lat_max_row
        )
        geo109 = src.read(1, window=window, out_dtype="int32")

    ds = int(downsample)
    geo109_ds = geo109[::ds, ::ds]

    eai_grid = np.full(geo109_ds.shape, np.nan, dtype=float)
    for g109_id, iso in geo109_to_iso.items():
        eai_val = iso_to_eai.get(iso, np.nan)
        if not np.isnan(eai_val):
            eai_grid[geo109_ds == g109_id] = eai_val

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
