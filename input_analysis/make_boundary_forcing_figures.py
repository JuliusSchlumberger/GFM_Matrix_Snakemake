"""Draw the figures of the boundary-forcing input analysis, each with a .txt description.

Reads the tables written by the `boundary_test/` scripts (INPUT_ANALYSIS.md §10) from
<out-dir>/ and <out-dir>/analysis/ (fig12-17: the level product in --product-dir),
and writes figNN_*.png + figNN_*.txt into --dest. Figures carry no titles;
everything a reader needs to decode them is in the description file.

Usage:
    python input_analysis/make_boundary_forcing_figures.py [--only fig01 fig07]
        [--out-dir .cleanup_baseline/level_sources] [--product-dir .cleanup_baseline/level_product]
        [--dest input_analysis/boundary_forcing/figures]
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from map_style import LAND_COLOR, draw_caption_box, draw_panel_letter  # noqa: E402
from tide_prediction import VERSION_DIFF_M, VERSION_RATIO, version_flag  # noqa: E402
from regions import ar6_regions  # noqa: E402
from site_sets import GRID, INK, MUTED, _save, _style  # noqa: E402

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
RP_BINS = (0, 2, 5, 10, 20, 50)
RP_SHADES = ("#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08306b")
PROFILE_RANGES = ((0, 1), (2, 4))
MIN_REGION_SITES = 5
SURGE_SEARCH_HR = 12
MIN_SURGE_M = 0.1
WINDOW_PLOT_HR = 48
MAP_PROJ = ccrs.EqualEarth()
PC = ccrs.PlateCarree()


def describe(path: Path, text: str) -> None:
    paras = [textwrap.fill(" ".join(p.split()), 100) for p in textwrap.dedent(text).strip().split("\n\n")]
    path.with_suffix(".txt").write_text("\n\n".join(paras) + "\n", encoding="utf-8")


def _rp_label(iv) -> str:
    return f"RP {iv.left:g}-{iv.right:g} y"


def _land(ax) -> None:
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=0)


def _sites(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)


def fig01_tide_scatter(out: Path, dest: Path) -> None:
    s = _sites(out)
    q = s[s.quality].copy()
    q["fes2014__mhws_m"] = pd.read_csv(out / "analysis" / "atlas_gauges.csv", index_col=0).fes2014__mhws_m
    panels = (("gtsm_harm__mhws_m", "GTSM"), ("fes_gauge__mhws_m", "FES2022"), ("fes2014__mhws_m", "FES2014"))
    hi = float(q[[c for c, _ in panels] + ["gesla_harm__mhws_m"]].max().max()) * 1.05
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    stats = {}
    for ax, letter, (col, label) in zip(axes, "abc", panels):
        ok = q[[col, "gesla_harm__mhws_m"]].dropna()
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1)
        ax.scatter(ok.gesla_harm__mhws_m, ok[col], s=9, color=BLUE, alpha=0.6, linewidths=0)
        ratio = (ok[col] / ok.gesla_harm__mhws_m).median()
        mae = (ok[col] - ok.gesla_harm__mhws_m).abs().mean()
        stats[label] = (ratio, mae, len(ok), int((q[col].isna()).sum()))
        draw_caption_box(ax, [f"median ratio {ratio:.2f}", f"MAE {mae:.3f} m", f"n = {len(ok)}"], loc="lower right")
        ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="GESLA MHWS (m)", ylabel=f"{label} MHWS (m)", aspect="equal")
        _style(ax)
        draw_panel_letter(ax, letter)
    fig.tight_layout()
    _save(fig, dest / "fig01_tide_mhws_scatter.png")
    missing14 = ", ".join(q.site_name[q.fes2014__mhws_m.isna()])
    missing22 = ", ".join(q.site_name[q.fes_gauge__mhws_m.isna()]) or "none"
    describe(dest / "fig01_tide_mhws_scatter.png", f"""
        Mean high water springs (MHWS, height above local mean sea level) of three tide sources
        against the GESLA tide gauges, at the {len(q)} quality sites of the tide set (harmonic fit
        over >= 3 years explaining >= 50% of the variance, GTSM station within 10 km). Each dot is
        one gauge; the grey line is 1:1. (a) GTSM v3 tide-only run at the paired GTSM station;
        (b) FES2022 atlas at the gauge position; (c) FES2014 atlas at the gauge position. The box
        gives the median of the ratio source / gauge, the mean absolute error (MAE) and the number
        of sites n.

        All sources go through the same chain, so method differences cannot explain what
        remains: GESLA and GTSM series are fitted by least squares with the 34 FES constituents
        (nodal corrections, inferred minor constituents), FES uses its atlas constants, all are
        predicted over the same 18.61-year nodal cycle at 10 min without the seasonal Sa / Ssa
        terms, and MHWS is computed by the same code (src/tide_statistics.py).

        Why n differs between the panels: every quality site has a GTSM station, so (a) has all
        {stats['GTSM'][2]}. FES values are read at the gauge itself and need a valid atlas cell
        within 20 km. FES2022 has no value at {missing22} (sites lost to a since-fixed extraction
        error, filled by rerunning levels_tides.py), FES2014 none at {missing14} (inner harbours
        outside its 1/16 deg ocean mask).

        Reading: GTSM's tide is {(stats['GTSM'][0] - 1) * 100:.0f}% too large in the median, mostly
        from S2 and M2 (fig03). FES2022 matches the gauges best; FES2014 is close but has more
        scatter, which is why FES2022 is the primary tide source and FES2014 only the fallback at
        basin heads (fig04-fig06).
        """)


def fig02_tide_ratio_maps(out: Path, dest: Path) -> None:
    s = _sites(out)
    q = s[s.quality]
    norm = LogNorm(1 / 1.5, 1.5)
    fig = plt.figure(figsize=(10, 10.2))
    sc = None
    for k, (col, letter) in enumerate((("gtsm_harm__mhws_m", "a"), ("fes_gauge__mhws_m", "b"))):
        ax = fig.add_axes([0.02, 0.54 - k * 0.465, 0.96, 0.42], projection=MAP_PROJ)
        ax.set_global()
        _land(ax)
        r = (q[col] / q.gesla_harm__mhws_m).dropna()
        r = r.iloc[np.argsort(np.abs(np.log(r.to_numpy())))]
        sc = ax.scatter(q.loc[r.index, "lon"], q.loc[r.index, "lat"], c=r, s=12, cmap="RdBu_r", norm=norm,
                        linewidths=0.2, edgecolors="white", transform=PC, zorder=2)
        ax.spines["geo"].set_edgecolor(MUTED)
        draw_panel_letter(ax, letter)
    cax = fig.add_axes([0.25, 0.055, 0.5, 0.018])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_ticks([1 / 1.5, 0.8, 1.0, 1.25, 1.5], labels=["0.67", "0.8", "1", "1.25", "1.5"])
    cb.set_label("MHWS ratio, source / GESLA")
    _save(fig, dest / "fig02_tide_mhws_ratio_maps.png")
    describe(dest / "fig02_tide_mhws_ratio_maps.png", """
        The MHWS ratios of fig01 as maps (Equal Earth projection; land in grey). Each dot is one
        quality tide gauge; its colour is the ratio of the source's MHWS to the gauge's MHWS on a
        logarithmic scale centred on 1: red = source too large, blue = too small, white = equal.
        Ratios beyond 0.67 or 1.5 take the end colours. Dots closest to 1 are drawn first so that
        departures stay visible where gauges cluster. (a) GTSM v3 tide; (b) FES2022.

        Reading: GTSM is too large almost everywhere (Africa and the Mediterranean ~20%, NW
        Europe ~10%) and too small on the North American Atlantic and Gulf coast. FES2022 is
        close to 1 nearly everywhere; its remaining departures sit in estuaries, fjords and narrow
        entrances (upper St Lawrence, Long Island Sound, the Salish Sea, Port Phillip entrance,
        Juneau). FES2022 read at the GTSM station instead of the gauge gives the same values (MAE
        0.01 m), so the GTSM pattern is a model bias, not a location effect.
        """)


def fig03_constituents(out: Path, dest: Path) -> None:
    c = pd.read_csv(out / "analysis" / "levels_constituent_ratios.csv")
    order = ["m2", "s2", "n2", "k1", "o1", "m4", "ms4"]
    series = (("gtsm_harm vs gesla_harm", "GTSM", BLUE, -0.12), ("fes_gauge vs gesla_harm", "FES2022", ORANGE, 0.12))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(order))
    for comp, label, color, dx in series:
        t = c[c.comparison == comp].set_index("constituent").reindex(order)
        ax.vlines(x + dx, t.p25, t.p75, color=color, linewidth=2.2, alpha=0.6)
        ax.scatter(x + dx, t.median_ratio, color=color, s=34, zorder=3, label=label)
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_xticks(x, [k.upper() for k in order])
    ax.set(ylabel="amplitude ratio, source / GESLA")
    ax.legend(frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    _save(fig, dest / "fig03_tide_constituent_ratios.png")
    n = c[c.comparison == "gtsm_harm vs gesla_harm"].set_index("constituent").n
    describe(dest / "fig03_tide_constituent_ratios.png", f"""
        Amplitude of the main tidal constituents relative to the gauge, at the quality tide
        gauges. For each constituent the dot is the median over sites of source amplitude / gauge
        amplitude and the bar the interquartile range; blue = GTSM v3 tide (harmonic fit of the
        tide-only run at the paired station), orange = FES2022 at the gauge. Only sites where the
        gauge amplitude of that constituent exceeds 2 cm enter (n per constituent from
        {n.min()} to {n.max()}), because ratios of near-zero amplitudes are noise.

        M2, S2, N2: semidiurnal; K1, O1: diurnal; M4, MS4: shallow-water overtides.

        Reading: GTSM's S2 is about 20% and M2 about 6% too large, against 0.99-1.00 for FES2022.
        The spring range is roughly M2 + S2, so this is the source of GTSM's MHWS excess in fig01.
        The likely cause is GTSM v3's uncalibrated bathymetry and bottom friction (Wang et al.,
        2022).
        """)


def _profiles(out: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    ev = pd.read_parquet(out / "events.parquet").reset_index(drop=True)
    prof = np.load(out / "event_profiles.npz")
    sites = _sites(out)
    ev["spring_range_m"] = ev.file_name.map(sites.gesla_harm__spring_range_m)
    ev["rp_class"] = pd.cut(ev.rp_empirical_yr, RP_BINS)
    keep = ev.file_name.map(sites.quality).fillna(False).astype(bool).to_numpy()
    return ev[keep].reset_index(drop=True), prof["profiles"][keep], prof["surge_m"][keep], prof["offset_hr"]


def fig08_profiles_by_rp(out: Path, dest: Path) -> None:
    ev, prof, _, offset = _profiles(out)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
    counts = {}
    for i, src in enumerate(("gesla", "gtsm")):
        for j, (lo, hi) in enumerate(PROFILE_RANGES):
            ax = axes[i, j]
            in_range = ev.spring_range_m.between(lo, hi, inclusive="right")
            for rp, color in zip(ev.rp_class.cat.categories, RP_SHADES):
                sel = ((ev.source == src) & in_range & (ev.rp_class == rp)).to_numpy()
                ax.plot(offset, np.nanmedian(prof[sel], axis=0), color=color, linewidth=1.6, label=_rp_label(rp))
                counts[(src, lo, rp.right)] = int(sel.sum())
            hg = ((ev.source == "coast_hg_spring") & in_range).to_numpy()
            ax.plot(offset, np.nanmedian(prof[hg], axis=0), color=GREEN, linewidth=1.4, linestyle="--",
                    label="COAST-HG spring, RP100")
            counts[("coast_hg", lo)] = int(hg.sum())
            ax.axhline(0, color=MUTED, linewidth=0.8)
            ax.set(xlim=(-WINDOW_PLOT_HR, WINDOW_PLOT_HR), ylim=(-1.0, 1.05))
            if i == 1:
                ax.set_xlabel("hours from peak")
            if j == 0:
                ax.set_ylabel("water level / peak")
            _style(ax)
            draw_panel_letter(ax, "abcd"[2 * i + j])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, dest / "fig08_profiles_by_rp.png")
    n_gesla = "; ".join(f"{_rp_label(rp)}: {counts[('gesla', 0, rp.right)]} / {counts[('gesla', 2, rp.right)]}"
                        for rp in ev.rp_class.cat.categories)
    describe(dest / "fig08_profiles_by_rp.png", f"""
        Median storm-tide profile of the annual maxima, by empirical return period (RP), at the
        quality tide gauges. Every event is cut out of the record with +-72 h around its peak
        (+-48 h shown), expressed relative to local mean sea level (horizontal line at 0) and
        divided by its peak, so each event starts at 1 at hour 0. Each line is the median over
        all events of a class at every hour. Rows: (a, b) GESLA gauge records, (c, d) GTSM
        (surge + harmonic tide) at the paired station, 1979-2024. Columns: gauges with a spring
        tidal range of 0-1 m (a, c) and 2-4 m (b, d). Blue shades: RP classes from light (RP 0-2
        y) to dark (RP 20-50 y). Dashed green: the COAST-HG spring-tide RP100 hydrograph at the
        paired stations ({counts[('coast_hg', 0)]} / {counts[('coast_hg', 2)]} sites), the shape
        the pipeline uses for every RP.

        Empirical RP: each event's rank among its site's annual maxima, RP = (n + 1) / rank with n
        the number of complete years. With 30-46 years of record the largest RP is about 45, so
        the top class holds RP 20-~45 events. GESLA events per class (0-1 m / 2-4 m range):
        {n_gesla}.

        Why the curves look so smooth and aligned: the peak sits at hour 0 and equals 1 by
        construction, the median over hundreds to thousands of events removes the individual
        noise, and the semidiurnal tide (period 12.42 h) puts the neighbouring high waters of
        nearly every event at the same hours. The event-to-event spread around these medians is
        considerable but not drawn.

        Reading: rarer events are more surge-dominated, so their neighbouring high waters are
        lower relative to the peak, most clearly in the microtidal class (a, c), where the
        next-day high water falls from about 0.8 to 0.55 of the peak. On microtidal coasts
        COAST-HG's next-day high water lies within the observed range, but between the high
        waters it drains to about -0.3 of the peak where observed events stay near mean sea
        level. On meso- and macrotidal coasts (b, d) COAST-HG's neighbouring high waters are
        lower than every observed class. Both mean a shorter time near the peak, i.e. a shorter T
        for the eikonal duration correction. Test 7 (fig09) traces the meso- and macrotidal
        deficit to COAST-HG's tide window, not to its surge.
        """)


def _recentre(s: np.ndarray, k: int) -> np.ndarray:
    """Shift s so that index k lands in the centre; vacated ends are NaN."""
    out = np.full(len(s), np.nan)
    shift = len(s) // 2 - k
    if shift >= 0:
        out[shift:] = s[:len(s) - shift]
    else:
        out[:shift] = s[-shift:]
    return out


def _dur_half(shifted: np.ndarray) -> int:
    above = np.where(np.isfinite(shifted), shifted, -np.inf) >= 0.5
    a = b = len(shifted) // 2
    while a > 0 and above[a - 1]:
        a -= 1
    while b < len(above) - 1 and above[b + 1]:
        b += 1
    return b - a + 1


def fig09_surge_by_rp(out: Path, dest: Path) -> None:
    ev, _, surge, offset = _profiles(out)
    parts_file = out / "analysis" / "coast_hg_parts.npz"
    centre = len(offset) // 2
    rows, shapes = [], []
    for i in np.flatnonzero(ev.source.isin(["gesla", "gtsm"]).to_numpy()):
        s = surge[i].astype(float)
        seg = s[centre - SURGE_SEARCH_HR:centre + SURGE_SEARCH_HR + 1]
        if not np.isfinite(seg).any():
            continue
        k = centre - SURGE_SEARCH_HR + int(np.nanargmax(seg))
        smax = s[k]
        if smax < MIN_SURGE_M:
            continue
        shifted = _recentre(s / smax, k)
        rows.append({"source": ev.source[i], "rp_class": ev.rp_class[i], "dur50": _dur_half(shifted)})
        shapes.append(shifted)
    t = pd.DataFrame(rows)
    shapes = np.vstack(shapes)
    hg = None
    if parts_file.exists():
        p = np.load(parts_file)
        q = set(_sites(out).index[_sites(out).quality])
        sel = np.array([f in q for f in p["file_name"]])
        hs = p["spring_surge"][sel].astype(float)
        hg_shapes, hg_dur = [], []
        for s in hs:
            k = int(np.argmax(s))
            shifted = _recentre(s / s[k], k)
            hg_shapes.append(shifted)
            hg_dur.append(_dur_half(shifted))
        hg = (np.nanmedian(np.vstack(hg_shapes), axis=0), float(np.median(hg_dur)), len(hs))
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.9))
    cats = ev.rp_class.cat.categories
    for j, src in enumerate(("gesla", "gtsm")):
        ax = fig.add_subplot(gs[0, j])
        for rp, color in zip(cats, RP_SHADES):
            sel = ((t.source == src) & (t.rp_class == rp)).to_numpy()
            ax.plot(offset, np.nanmedian(shapes[sel], axis=0), color=color, linewidth=1.6, label=_rp_label(rp))
        if hg is not None:
            ax.plot(offset, hg[0], color=GREEN, linestyle="--", linewidth=1.4, label="COAST-HG spring, RP100")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axhline(0.5, color=GRID, linewidth=0.8, linestyle=":")
        ax.set(xlim=(-WINDOW_PLOT_HR, WINDOW_PLOT_HR), ylim=(-0.4, 1.05), xlabel="hours from surge maximum")
        if j == 0:
            ax.set_ylabel("surge / surge maximum")
        _style(ax)
        draw_panel_letter(ax, "ab"[j])
    ax = fig.add_subplot(gs[0, 2])
    pos, data, colors = [], [], []
    for k, rp in enumerate(cats):
        for d, (src, color) in enumerate((("gesla", INK), ("gtsm", BLUE))):
            pos.append(k + (d - 0.5) * 0.32)
            data.append(t.dur50[(t.source == src) & (t.rp_class == rp)])
            colors.append(color)
    bp = ax.boxplot(data, positions=pos, widths=0.26, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor="white", edgecolor=color)
    for key in ("whiskers", "caps", "medians"):
        for line, color in zip(bp[key], np.repeat(colors, 2) if key != "medians" else colors):
            line.set_color(color)
    if hg is not None:
        ax.axhline(hg[1], color=GREEN, linestyle="--", linewidth=1.4)
    ax.set_xticks(range(len(cats)), [f"{iv.left:g}-{iv.right:g}" for iv in cats])
    ax.set(xlabel="empirical RP (y)", ylabel="surge duration above half its maximum (h)")
    _style(ax)
    draw_panel_letter(ax, "c")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK), plt.Line2D([], [], color=BLUE)]
    labels += ["GESLA (c)", "GTSM (c)"]
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, dest / "fig09_surge_by_rp.png")
    med = t.groupby(["source", "rp_class"], observed=True).dur50.median().unstack()
    n = t.groupby(["source", "rp_class"], observed=True).size().unstack()
    hg_text = (f"Dashed green: the surge part of COAST-HG's spring-tide RP100 hydrograph at {hg[2]} "
               f"paired stations, separated from its tide in test 7 (boundary_test/coast_hg_decompose.py); "
               f"median duration {hg[1]:.0f} h." if hg is not None else "COAST-HG: not yet separated.")
    describe(dest / "fig09_surge_by_rp.png", f"""
        Surge-only hydrographs of the annual maxima by empirical return period, at the quality
        tide gauges. The surge is the water level minus the astronomical tide: for GESLA the
        record minus the site's harmonic tide prediction (the non-tidal residual), for GTSM the
        ERA5-forced surge series at the paired station. For each annual maximum the largest
        surge within +-{SURGE_SEARCH_HR} h of the water-level peak is taken (the storm that made
        the annual maximum), the series is shifted so that this surge maximum sits at hour 0 and
        divided by it. Events with a surge maximum below {MIN_SURGE_M} m are left out (pure
        spring-tide maxima). (a) GESLA and (b) GTSM: median normalised surge per RP class (blue
        shades as in fig08); the dotted line marks half the maximum. (c) The duration the surge
        stays above half its maximum, per event: boxes = interquartile range, line = median,
        whiskers = 1.5 x IQR; black GESLA, blue GTSM. {hg_text}

        Median duration above half the maximum (h), GESLA by RP class: {', '.join(f'{v:.0f}' for v in med.loc['gesla'])};
        GTSM: {', '.join(f'{v:.0f}' for v in med.loc['gtsm'])}. Events per class, GESLA:
        {', '.join(str(int(v)) for v in n.loc['gesla'])}; GTSM: {', '.join(str(int(v)) for v in n.loc['gtsm'])}.

        On macrotidal coasts the non-tidal residual carries tide-surge interaction (oscillations
        at tidal periods), which makes the GESLA surges look shorter and noisier than the GTSM
        surges; GTSM's surge comes from a tide + surge run minus a tide-only run and has the same
        effect in weaker form. The bumps at about +-12 and +-24 h in (a) and (b) are this residual
        tide, not separate surges.

        Reading: the surge itself hardly changes with return period. Its core (above half the
        maximum) lasts about the same in every class, and COAST-HG's surge core matches it. What
        differs is the tail: observed surges still hold 0.2-0.35 of their maximum 30-48 h from the
        peak (set-up that outlasts the storm, and successive storms), whereas COAST-HG's surge is
        back to zero by about +-35 h. That tail is small next to the tide on meso- and macrotidal
        coasts. Test 7 shows where COAST-HG's low neighbouring high waters (fig08) come from: on
        coasts with a spring range above 1 m the surge parts of the next-day high water agree
        (0.03-0.12 of the peak for COAST-HG and gauges), but the tide parts do not (0.59-0.70 for
        COAST-HG against 0.66-0.83 at the gauges), because observed extremes coincide with tides of
        1.18-1.28 x MHWS while COAST-HG's tide window reaches 1.03-1.07 x MHWS and the rest of the
        RP100 level is assigned to the surge. Data: coast_hg_vs_events.csv.
        """)


CONTINENT_ORDER = ("NORTH-AMERICA", "CENTRAL-AMERICA", "SOUTH-AMERICA", "EUROPE", "EUROPE-AFRICA", "AFRICA",
                   "ASIA", "OCEANIA", "POLAR", "ATLANTIC", "INDIAN", "PACIFIC", "ARCTIC", "SOUTHERN")
RP_COLORS = {10: BLUE, 100: ORANGE}
GROUP_LABELS = {"EUROPE-AFRICA": "Mediterranean", "OTHER": "", "INDIAN": "Indian Ocean",
                "PACIFIC": "Pacific Ocean", "ATLANTIC": "Atlantic Ocean", "ARCTIC": "Arctic Ocean",
                "SOUTHERN": "Southern Ocean"}


def fig07_coast_rp_regions(out: Path, dest: Path) -> None:
    e = pd.read_parquet(out / "extremes.parquet")
    e = e[e.source == "gesla"].set_index("file_name")
    sets = pd.read_csv(out / "analysis" / "site_sets.csv", index_col=0)
    keep = sets.index[sets.extremes].intersection(e.index)
    d = pd.DataFrame({rp: e.loc[keep, f"coast_rp_rp{rp}_m"] - e.loc[keep, f"rp{rp}_m"] for rp in (10, 100)}).dropna()
    reg = ar6_regions(sets.loc[d.index, "lon"].to_numpy(), sets.loc[d.index, "lat"].to_numpy())
    reg.index = d.index
    name = (reg.region_name.str.replace("Equatorial.", "Equatorial ", regex=False)
            .str.replace(".", ". ", regex=False).str.replace("-", " "))
    counts = name.value_counts()
    small = set(counts.index[counts < MIN_REGION_SITES])
    d["label"] = np.where(name.isin(small), f"other ({len(small)} regions)", name)
    d["continent"] = np.where(name.isin(small), "OTHER", reg.continent)
    pooled = ", ".join(f"{r} ({counts[r]})" for r in sorted(small))

    def stats(g: pd.DataFrame) -> dict:
        return {"n": len(g), **{f"{q}_{rp}": g[rp].quantile(v) for rp in (10, 100)
                                for q, v in (("p25", 0.25), ("med", 0.5), ("p75", 0.75))}}

    rows = [("all sites", "", stats(d))]
    for cont in (*CONTINENT_ORDER, "OTHER"):
        g = d[d.continent == cont]
        for label in sorted(g.label.unique()):
            rows.append((label, cont, stats(g[g.label == label])))
    st = pd.DataFrame([r[2] for r in rows], index=[r[0] for r in rows])
    st["continent"] = [r[1] for r in rows]
    y, ticks, labels, heads, ypos = [], [], [], [], 0.0
    for k, (label, cont) in enumerate(zip(st.index, st.continent)):
        if k > 0 and cont != st.continent.iloc[k - 1]:
            ypos += 0.3
            head = GROUP_LABELS.get(cont, cont.title().replace("-", " "))
            if head and list(st.index[st.continent == cont]) != [head]:
                ticks.append(ypos)
                labels.append(head)
                heads.append(ypos)
                ypos += 1
        y.append(ypos)
        ticks.append(ypos)
        labels.append(label)
        ypos += 1
    y = np.array(y)
    lo = float(st[["p25_10", "p25_100"]].min().min())
    hi = float(st[["p75_10", "p75_100"]].max().max())
    pad = 0.06 * (hi - lo)
    fig = plt.figure(figsize=(9.2, 0.25 * ypos + 1.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 0.3), wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    for rp, dy in ((10, -0.17), (100, 0.17)):
        ax.hlines(y + dy, st[f"p25_{rp}"], st[f"p75_{rp}"], color=RP_COLORS[rp], linewidth=2, alpha=0.5)
        ax.scatter(st[f"med_{rp}"], y + dy, color=RP_COLORS[rp], s=20, zorder=3, label=f"RP{rp}")
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(ticks, labels, fontsize=8)
    for tick, label in zip(ax.get_yticklabels(), labels):
        if ticks[labels.index(label)] in heads:
            tick.set_fontweight("bold")
    ax.set(xlim=(lo - pad, hi + pad), ylim=(ypos - 0.4, -0.6), xlabel="COAST-RP - gauge GEV (m)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    ax2.barh(y, st.n, color=MUTED, height=0.6)
    ax2.set(xlabel="sites", xscale="log", xlim=(1, st.n.max() * 1.3))
    plt.setp(ax2.get_yticklabels(), visible=False)
    _style(ax2)
    draw_panel_letter(ax2, "b")
    fig.subplots_adjust(left=0.2, right=0.98, bottom=0.65 / (0.25 * ypos + 1.4), top=1 - 0.45 / (0.25 * ypos + 1.4))
    _save(fig, dest / "fig07_coast_rp_vs_gauges.png")
    st.drop(columns="continent").round(3).to_csv(dest / "fig07_coast_rp_vs_gauges_data.csv")
    describe(dest / "fig07_coast_rp_vs_gauges.png", f"""
        COAST-RP storm-tide return levels against the tide gauges, by IPCC AR6 reference region
        (Iturbide et al., 2020). Both sides are total still-water levels (tide plus surge) relative
        to local mean sea level: COAST-RP at the paired GTSM station, and a GEV fit to the gauge's
        annual maxima. Sites: the record-based extremes set ({len(d)} GESLA sites with >= 20
        complete years since 1979 and a GTSM station within 10 km, whatever their tide).

        (a) COAST-RP minus gauge per region: dot = median over the region's sites, bar =
        interquartile range; blue = RP10, orange = RP100. Negative = COAST-RP too low. (b) Number of
        sites per region (log scale). The top row pools all sites.

        Order: regions are grouped by their AR6 continent in a fixed geographic order (North,
        Central and South America, Europe, the Mediterranean, Africa, Asia, Oceania, the polar land
        regions, then the ocean regions of remote islands: Atlantic, Indian, Pacific, Arctic,
        Southern), alphabetically within each group; the bold rows name the groups. Each gauge
        takes the nearest AR6 land region within 200 km, otherwise the AR6 ocean region containing
        it. The {len(small)} regions with fewer than {MIN_REGION_SITES} sites are pooled in the last
        row: {pooled}. The numbers are in fig07_coast_rp_vs_gauges_data.csv.

        Reading: overall COAST-RP is {st.loc['all sites', 'med_10']:+.2f} m (RP10) and
        {st.loc['all sites', 'med_100']:+.2f} m (RP100) relative to the gauges. That near-zero
        median hides two offsetting errors: GTSM's tide is too large (fig01) and its surge too low,
        so with the tide excess removed COAST-RP is about 0.11-0.12 m too low. COAST-RP is lowest in
        tropical-cyclone regions (eastern North America, East Asia, the Caribbean, the Pacific),
        where 30-40-year gauge records also under-sample cyclones, and too high in West and Central
        Europe, where GTSM's tide excess dominates.
        """)


def _nodes(out: Path) -> pd.DataFrame:
    n = pd.read_parquet(out / "analysis" / "atlas_node_mhws.parquet")
    n = n.dropna(subset=["fes2022_mhws_m", "fes2014_mhws_m"]).reset_index(drop=True)
    n["replaced"] = version_flag(n.fes2022_mhws_m, n.fes2014_mhws_m)
    return n


def fig04_fes_versions(out: Path, dest: Path) -> None:
    n = _nodes(out)
    r = n.fes2022_mhws_m / n.fes2014_mhws_m
    order = np.argsort(np.abs(np.log(r.clip(lower=1e-3).to_numpy())))
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0.01, 0.2, 0.6, 0.76], projection=MAP_PROJ)
    ax.set_global()
    _land(ax)
    sc = ax.scatter(n.node_lon[order], n.node_lat[order], c=r[order], s=3, cmap="RdBu_r", norm=LogNorm(1 / 3, 3),
                    linewidths=0, transform=PC, zorder=2)
    ax.spines["geo"].set_edgecolor(MUTED)
    draw_panel_letter(ax, "a")
    cax = fig.add_axes([0.13, 0.12, 0.36, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_ticks([1 / 3, 0.5, 1, 2, 3], labels=["0.33", "0.5", "1", "2", "3"])
    cb.set_label("MHWS ratio, FES2022 / FES2014")
    ax = fig.add_axes([0.68, 0.12, 0.3, 0.84])
    hi = float(n[["fes2022_mhws_m", "fes2014_mhws_m"]].max().max()) * 1.03
    kept = n[~n.replaced]
    ax.scatter(kept.fes2014_mhws_m, kept.fes2022_mhws_m, s=3, color=MUTED, alpha=0.5, linewidths=0, label="kept")
    rep = n[n.replaced]
    ax.scatter(rep.fes2014_mhws_m, rep.fes2022_mhws_m, s=14, marker="x", color="#e34948", linewidths=0.8,
               label="replaced by FES2014")
    x = np.linspace(0, hi, 300)
    ax.plot([0, hi], [0, hi], color=INK, linewidth=0.8)
    ax.plot(x, np.maximum(VERSION_RATIO * x, x + VERSION_DIFF_M), color=INK, linewidth=1, linestyle="--")
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="FES2014 MHWS (m)", ylabel="FES2022 MHWS (m)", aspect="equal")
    ax.legend(frameon=False, loc="upper left", markerscale=1.5)
    _style(ax)
    draw_panel_letter(ax, "b")
    _save(fig, dest / "fig04_fes2022_vs_fes2014.png")
    describe(dest / "fig04_fes2022_vs_fes2014.png", f"""
        FES2022 against FES2014 at the {len(n):,} FES boundary nodes of the test tiles (the atlas
        grid nodes that seed the tiles' boundary cells). At every node both atlases are predicted
        over the same 18.61-year nodal cycle at 10 min and reduced to MHWS (mean high water springs,
        height above local mean sea level) exactly as at the tide gauges (fig01).

        (a) Map (Equal Earth; land grey) of the ratio MHWS_FES2022 / MHWS_FES2014 per node on a
        logarithmic scale centred on 1: red = FES2022 higher, blue = FES2014 higher; beyond 1/3
        and 3 the end colours. Nodes closest to 1 are drawn first. (b) The same nodes as a scatter,
        FES2014 on x, FES2022 on y; thin line 1:1. Dashed line: the boundary of the version rule.
        A node above it, i.e. with MHWS_FES2022 > {VERSION_RATIO} x MHWS_FES2014 and
        MHWS_FES2022 - MHWS_FES2014 > {VERSION_DIFF_M} m, takes FES2014 instead of FES2022 (red
        crosses, {int(n.replaced.sum()):,} nodes = {n.replaced.mean():.1%}); grey dots keep FES2022.
        How R = {VERSION_RATIO} and D = {VERSION_DIFF_M} m were chosen is shown in fig06.

        Reading: at almost all nodes the two FES versions agree (ratio within 0.9-1.1). The
        outliers where FES2022 is far higher are a few basin and fjord heads (fig05). The ratio
        alone also flags microtidal coasts where both MHWS are a few centimetres; the minimum
        difference D keeps those on FES2022.
        """)


FIG05_BASINS = {"Minas Basin / Cobequid Bay": (-65.2, -63.2, 44.9, 45.6),
                "Strait of Magellan / Bahia Lomas": (-71.5, -68.0, -53.6, -52.0),
                "Upper Gulf of California": (-115.3, -113.0, 30.3, 32.2),
                "Glacier Bay / Lynn Canal": (-137.2, -134.2, 58.2, 59.6)}
BASIN_PAD = 0.08
CLUSTER_EPS_DEG = 0.1  # about three FES2022 grid cells
RING_PAD_DEG = 0.03


def _replaced_clusters(lon: np.ndarray, lat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.cluster import DBSCAN

    if not len(lon):
        return []
    labels = DBSCAN(eps=CLUSTER_EPS_DEG, min_samples=1).fit_predict(np.column_stack([lon, lat]))
    return [(lon[labels == k], lat[labels == k]) for k in np.unique(labels)]


PAGE_IN = (16.0, 10.0)
GAP_IN, PAIR_GAP_IN, ROW_GAP_IN, TOP_IN, CBAR_IN, SIDE_IN = 0.45, 0.32, 0.5, 0.35, 0.95, 0.25


def _row_height(row: list[tuple[int, bool]], aspects: list[float], width: float) -> float:
    """Box height that makes a row of basin pairs exactly `width` wide (side pairs: 2 maps across)."""
    side = [aspects[k] for k, s in row if s]
    stacked = [aspects[k] for k, s in row if not s]
    free = width - GAP_IN * (len(row) - 1) - PAIR_GAP_IN * len(side) + PAIR_GAP_IN * sum(stacked) / 2
    return free / (2 * sum(side) + sum(stacked) / 2)


def _basin_layout(aspects: list[float]) -> list[list[tuple[int, bool]]]:
    """Rows of (basin, side_by_side) that make the smallest map largest on PAGE_IN."""
    import itertools

    def score(rows):
        width = PAGE_IN[0] - 2 * SIDE_IN
        heights, maps = [], []
        for row in rows:
            h = _row_height(row, aspects, width)
            heights.append(h)
            maps += [(h if side else (h - PAIR_GAP_IN) / 2) for _, side in row]
        free = PAGE_IN[1] - TOP_IN - CBAR_IN - ROW_GAP_IN * (len(rows) - 1)
        scale = min(1.0, free / sum(heights))
        return min(m * scale * (m * scale * aspects[k]) for m, (k, _) in zip(maps, (b for r in rows for b in r)))

    best, best_rows = -1.0, None
    n = len(aspects)
    for order in itertools.permutations(range(n)):
        for sides in itertools.product((False, True), repeat=n):
            boxes = list(zip(order, sides))
            for split in range(1, n + 1):
                rows = [boxes[:split], boxes[split:]] if split < n else [boxes]
                value = score(rows)
                if value > best:
                    best, best_rows = value, rows
    return best_rows


def fig05_basin_heads(out: Path, dest: Path) -> None:
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd
    from matplotlib.patches import Ellipse
    from shapely.geometry import box

    n = _nodes(out)
    g = pd.read_csv(out / "analysis" / "atlas_gauges.csv", index_col=0)
    land = gpd.read_file(shpreader.natural_earth("50m", "physical", "land"))
    basins = []
    for name, (x0, x1, y0, y1) in FIG05_BASINS.items():
        b = n[n.node_lon.between(x0, x1) & n.node_lat.between(y0, y1)]
        gb = g[g.lon.between(x0, x1) & g.lat.between(y0, y1)]
        lons, lats = np.concatenate([b.node_lon, gb.lon]), np.concatenate([b.node_lat, gb.lat])
        px, py = max(BASIN_PAD * np.ptp(lons), 0.05), max(BASIN_PAD * np.ptp(lats), 0.05)
        ext = (lons.min() - px, lons.max() + px, lats.min() - py, lats.max() + py)
        stretch = 1 / np.cos(np.radians(np.mean(ext[2:])))
        basins.append({"name": name, "b": b, "gb": gb, "ext": ext, "stretch": stretch,
                       "aspect": (ext[1] - ext[0]) / ((ext[3] - ext[2]) * stretch)})
    rows = _basin_layout([d["aspect"] for d in basins])
    width = PAGE_IN[0] - 2 * SIDE_IN
    heights = [_row_height(row, [d["aspect"] for d in basins], width) for row in rows]
    free = PAGE_IN[1] - TOP_IN - CBAR_IN - ROW_GAP_IN * (len(rows) - 1)
    scale = min(1.0, free / sum(heights))
    heights = [h * scale for h in heights]
    fig_w, fig_h = PAGE_IN[0], TOP_IN + sum(heights) + ROW_GAP_IN * (len(rows) - 1) + CBAR_IN
    vmax = float(max(np.nanmax(d["b"][["fes2022_mhws_m", "fes2014_mhws_m"]].to_numpy()) for d in basins))
    norm = Normalize(0, vmax)
    fig = plt.figure(figsize=(fig_w, fig_h))
    y_top = fig_h - TOP_IN
    summary, sc, position = [], None, 0
    for row, h in zip(rows, heights):
        row_w = sum((basins[k]["aspect"] * 2 * h + PAIR_GAP_IN if side else basins[k]["aspect"] * (h - PAIR_GAP_IN) / 2)
                    for k, side in row) + GAP_IN * (len(row) - 1)
        x = (fig_w - row_w) / 2
        for k, side in row:
            d = basins[k]
            letter = "abcd"[position]
            position += 1
            map_h = h if side else (h - PAIR_GAP_IN) / 2
            map_w = d["aspect"] * map_h
            b, gb, ext = d["b"], d["gb"], d["ext"]
            clip = land.clip(box(ext[0], ext[2], ext[1], ext[3]))
            keep, rep = b[~b.replaced], b[b.replaced]
            clusters = _replaced_clusters(rep.node_lon.to_numpy(), rep.node_lat.to_numpy())
            for i, (col, rep_marker) in enumerate((("fes2022_mhws_m", "X"), ("fes2014_mhws_m", "D"))):
                ax_x = x + (i * (map_w + PAIR_GAP_IN) if side else 0)
                ax_y = y_top - map_h - (0 if side else i * (map_h + PAIR_GAP_IN))
                ax = fig.add_axes([ax_x / fig_w, ax_y / fig_h, map_w / fig_w, map_h / fig_h])
                clip.plot(ax=ax, color=LAND_COLOR, edgecolor="none", zorder=0)
                sc = ax.scatter(keep.node_lon, keep.node_lat, c=keep[col], s=14, cmap="viridis", norm=norm,
                                linewidths=0, zorder=2)
                ax.scatter(rep.node_lon, rep.node_lat, c=rep[col], s=26, marker=rep_marker, cmap="viridis",
                           norm=norm, linewidths=0.3, edgecolors=INK, zorder=3)
                ax.scatter(gb.lon, gb.lat, c=gb.gesla_harm__mhws_m, s=120, marker="*", cmap="viridis", norm=norm,
                           edgecolors=INK, linewidths=0.7, zorder=4)
                for clon, clat in clusters:
                    # sqrt(2) makes the axis-aligned ellipse pass through the box corners, so it encloses all nodes
                    ax.add_patch(Ellipse(((clon.min() + clon.max()) / 2, (clat.min() + clat.max()) / 2),
                                         np.sqrt(2) * np.ptp(clon) + 2 * RING_PAD_DEG * d["stretch"],
                                         np.sqrt(2) * np.ptp(clat) + 2 * RING_PAD_DEG, fill=False,
                                         edgecolor="#e34948", linewidth=1.1, zorder=5))
                ax.set(xlim=ext[:2], ylim=ext[2:], xticks=[], yticks=[])
                ax.set_aspect(d["stretch"], adjustable="datalim")
                for spine in ax.spines.values():
                    spine.set_edgecolor(MUTED)
                draw_panel_letter(ax, f"{letter}{i + 1}")
            summary.append(f"({letter}1, {letter}2) {d['name']}: {len(b)} nodes, {int(b.replaced.sum())} replaced in "
                           f"{len(clusters)} cluster(s); FES2022 MHWS max {b.fes2022_mhws_m.max():.1f} m, FES2014 max "
                           f"{b.fes2014_mhws_m.max():.1f} m"
                           + ("; gauges: " + ", ".join(f"{q.site_name} {q.gesla_harm__mhws_m:.2f} m"
                                                       for q in gb.itertuples()) if len(gb) else "; no gauge"))
            x += (2 * map_w + PAIR_GAP_IN if side else map_w) + GAP_IN
        y_top -= h + ROW_GAP_IN
    cax = fig.add_axes([0.3, 0.45 / fig_h, 0.4, 0.16 / fig_h])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("MHWS (m)")
    _save(fig, dest / "fig05_fes_basin_heads.png")
    names = "; ".join(s.split(":")[0] for s in summary)
    describe(dest / "fig05_fes_basin_heads.png", f"""
        MHWS (mean high water springs above local mean sea level) at the FES boundary nodes in the
        four basins where FES2022 has gross artefacts: {names}. For each basin the map marked 1
        shows FES2022 and the map marked 2 next to it (beside or below) FES2014, at the same nodes
        and over the same area. All maps share the colour scale at the bottom. The arrangement is
        the one that makes the smallest map largest on the page, so the basins are not in a fixed
        geographic order.

        Symbols: circles = nodes that keep FES2022 under the version rule (fig04, fig06). Crosses
        (maps 1) = FES2022 values the rule discards; diamonds (maps 2) = the FES2014 values that
        replace them. Red ovals enclose the replaced nodes: nodes closer than about three grid
        cells form one group, and each oval is the smallest axis-aligned ellipse around the box
        spanned by the group's westernmost, easternmost, northernmost and southernmost node.
        Stars = GESLA tide gauges, coloured by their observed MHWS. Each map is cropped to its nodes
        and gauges and drawn with the east-west scale corrected for latitude, so shapes are true.
        Land: Natural Earth 1:50m, grey.

        {chr(10).join(summary)}

        Observed references without a gauge in the map: Minas Basin mean spring range 14.5 m,
        record 16.3 m (MHWS about 7.3 m; Desplanque and Mossman, 2001); Strait of Magellan spring
        range about 9 m on the Atlantic side (Medeiros and Kjerfve, 1988); Colorado mouth spring
        range 7-10 m (Archer, 2013).

        Reading: FES2022 is plausible over most of each basin but gives MHWS as large as the whole
        observed spring range at a few head nodes; FES2014 shows none of these hot spots and stays
        within the observed ranges. The rule replaces exactly those head nodes.
        """)


def _heat(ax, grid: pd.DataFrame, col: str, text, cmap, norm, mark=None) -> object:
    ratios, diffs = sorted(grid.ratio.unique()), sorted(grid.diff_m.unique())
    m = grid.pivot(index="ratio", columns="diff_m", values=col).loc[ratios, diffs].to_numpy(dtype=float)
    im = ax.imshow(m, cmap=cmap, norm=norm, aspect="auto", origin="lower")
    for i, r in enumerate(ratios):
        for j, d in enumerate(diffs):
            row = grid[(grid.ratio == r) & (grid.diff_m == d)].iloc[0]
            rgb = im.cmap(im.norm(m[i, j]))[:3] if np.isfinite(m[i, j]) else (1.0, 1.0, 1.0)
            dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
            ax.text(j, i, text(row), ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
            if mark is not None and mark(row):
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False, edgecolor=INK,
                                           linewidth=0.9, linestyle=":"))
    i0, j0 = ratios.index(VERSION_RATIO), diffs.index(VERSION_DIFF_M)
    ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, edgecolor="#e34948", linewidth=2.5))
    ax.set_xticks(range(len(diffs)), [f"{d:g}" for d in diffs])
    ax.set_yticks(range(len(ratios)), [f"{r:g}" for r in ratios])
    ax.set(xlabel="D (m): FES2022 must exceed FES2014 by more than D", ylabel="R: FES2022 must exceed R x FES2014")
    return im


def fig06_rule_sensitivity(out: Path, dest: Path) -> None:
    grid = pd.read_csv(out / "analysis" / "atlas_rule_grid.csv")
    bands = pd.read_csv(out / "analysis" / "atlas_rule_bands.csv", index_col=0)
    best = int(grid.loc[grid.gauges_harmed == 0, "gauges_fixed"].max())
    known = int(grid.known_total.iloc[0])
    grid["known_share"] = grid.known_caught / known
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.8))
    g = pd.read_csv(out / "analysis" / "atlas_gauges.csv", index_col=0).dropna(
        subset=["gesla_harm__mhws_m", "fes_gauge__mhws_m", "fes2014__mhws_m"])
    rmse22 = float(np.sqrt(((g.fes_gauge__mhws_m - g.gesla_harm__mhws_m) ** 2).mean()))
    rmse14 = float(np.sqrt(((g.fes2014__mhws_m - g.gesla_harm__mhws_m) ** 2).mean()))
    im = _heat(axes[0], grid, "hybrid_rmse_m", lambda r: f"{int(r.gauges_fixed)}/{int(r.gauges_harmed)}",
               plt.get_cmap("Blues_r"), Normalize(float(grid.hybrid_rmse_m.min()),
                                                  float(grid.loc[grid.ratio <= 1.75, "hybrid_rmse_m"].max())), mark=lambda r: r.gauges_harmed == 0 and r.gauges_fixed == best)
    cb = fig.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.17, shrink=0.6, aspect=30, extend="max")
    cb.set_label("RMSE of MHWS against GESLA, 905 gauges (m)")
    _heat(axes[1], grid, "known_share", lambda r: f"{int(r.known_caught)}", plt.get_cmap("Blues"), Normalize(0, 1),
          mark=lambda r: r.known_caught == known)
    _heat(axes[2], grid, "nodes_replaced", lambda r: f"{int(r.nodes_replaced)}", plt.get_cmap("Oranges"),
          LogNorm(max(grid.nodes_replaced.min(), 1), grid.nodes_replaced.max()))
    for ax, letter in zip(axes, "abc"):
        draw_panel_letter(ax, letter)
    fig.tight_layout()
    _save(fig, dest / "fig06_version_rule_sensitivity.png")
    adopted = grid[(grid.ratio == VERSION_RATIO) & (grid.diff_m == VERSION_DIFF_M)].iloc[0]
    nodes_r1 = int(grid[(grid.ratio == 1.0) & (grid.diff_m == VERSION_DIFF_M)].nodes_replaced.iloc[0])
    band_lines = "; ".join(f"{idx} m: n={int(b.n)}, FES2014 closer at {b.fes2014_closer:.0%}, median gain "
                           f"{b.median_gain_m:+.2f} m" for idx, b in bands.iterrows())
    describe(dest / "fig06_version_rule_sensitivity.png", f"""
        How the two thresholds of the FES version rule were chosen. The rule replaces FES2022 by
        FES2014 at a point only if both conditions hold: (1) FES2022's MHWS is more than R times
        FES2014's, and (2) FES2022's MHWS exceeds FES2014's by more than D metres. R is a relative
        and D an absolute margin: R alone would switch microtidal coasts where both values are a
        few centimetres, D alone would switch large-tide coasts where FES2022 is only a little
        higher. Each cell of the two panels is one candidate pair (R vertical, D horizontal). The
        red frame is the adopted pair, R = {VERSION_RATIO} and D = {VERSION_DIFF_M} m.

        (a) What the candidate rule does at the 905 GESLA tide gauges that have MHWS from GESLA,
        FES2022 and FES2014. At every gauge where the rule would switch to FES2014, the switch is
        scored against the gauge's own MHWS: an improvement if FES2014 is closer to the gauge than
        FES2022 by more than 5 cm, a worsening if it is further by more than 5 cm. Cell text:
        improvements / worsenings. Colour: the root-mean-square error (RMSE) of MHWS against GESLA
        over all 905 gauges when the candidate rule is applied, i.e. FES2014 where it fires and
        FES2022 elsewhere; darker = lower error. The scale spans the pairs with R <= 1.75
        ({grid.hybrid_rmse_m.min():.3f}-{grid.loc[grid.ratio <= 1.75, "hybrid_rmse_m"].max():.3f} m);
        worse pairs (R >= 2, up to {grid.hybrid_rmse_m.max():.3f} m) are white. RMSE squares the
        errors, so it is dominated by the few gross ones: FES2022 alone is {rmse22:.3f} m mostly
        because of Skagway (error 4.7 m), and FES2014 alone would be {rmse14:.3f} m although its mean
        absolute error is larger than FES2022's (fig01). It therefore measures exactly what the
        rule is for: removing gross errors without adding new ones. The gauges with a real FES2022 error (Skagway, Juneau, Port
        Alberni) are the improvements to be had; Trondheim, where FES2014 is wrong, is the main
        worsening to avoid. Dotted squares: pairs with no worsening and the most improvements
        ({best}).

        (b) What the candidate rule does at {known} known artefacts: boundary nodes in the four
        basins of fig05 whose FES2022 MHWS exceeds 1.25 x the largest MHWS observed in that basin
        (Minas 8.15 m, Strait of Magellan 5.5 m, Colorado mouth 5.0 m, Lynn Canal 2.8 m). These
        nodes are wrong whatever FES2014 says, so they test the rule without circularity. Cell
        text: how many of the {known} the rule replaces; colour: that share (white 0%, dark blue
        100%). Dotted squares: pairs that replace all of them. In this range of D the result does
        not depend on D, because the artefacts exceed FES2014 by several metres: only R decides.

        (c) How many of the {len(_nodes(out)):,} FES boundary nodes of the test tiles the candidate
        rule switches to FES2014 (cell text; colour on a logarithmic scale, darker = more). At the
        adopted pair this is {int(adopted.nodes_replaced):,} nodes, the red crosses of fig04. Pairs
        that switch more nodes than needed change FES2022 values on coasts where no evidence says
        it is wrong.

        Reading off the adopted values: (a) leaves the dotted pairs in the column D = 1.25 m (lowest
        RMSE, {adopted.hybrid_rmse_m:.3f} m; no worsening, all {best} improvements; at D <= 1 m Trondheim is switched wrongly, at D = 1.5 m
        Port Alberni is no longer caught). (b) leaves R <= 1.5 (higher R misses artefacts). The
        row R = 1 switches the ratio condition off, so only D decides: from R = 1 to R = 1.5 panels
        (a) and (b) are identical, i.e. the ratio changes nothing at the gauges or at the known
        artefacts. It only changes how many other nodes are switched (c: {nodes_r1:,} at R = 1,
        {int(adopted.nodes_replaced):,} at R = {VERSION_RATIO}, for D = {VERSION_DIFF_M} m). R is
        therefore a precaution for coasts without evidence, not a value the data pin down; R = 1.5
        is the largest ratio that still replaces every known artefact. At the adopted pair: {int(adopted.gauges_fixed)} improvements,
        {int(adopted.gauges_harmed)} worsenings, {int(adopted.known_caught)} of {known} artefacts
        replaced.

        Behind the D bound: comparing the two atlases at all 905 gauges, binned by how much
        FES2022 exceeds FES2014 in MHWS (gain = |FES2022 - GESLA| - |FES2014 - GESLA|, positive
        when FES2014 is better): {band_lines}. Below about 1 m FES2014 is not the better atlas, so
        switching there does harm. The gauge evidence rests on few sites (5 with a difference above
        1 m), so the known artefacts carry the ratio bound.
        """)


GEN_VARIANTS = (("generator_smooth", "generator (smoothed surge)", BLUE, "-"),
                ("generator", "generator (raw surge)", BLUE, ":"),
                ("hgrapher", "HGRAPHER port", ORANGE, "-"))
GEN_RANGES = ((0, 1), (1, 2), (2, 4), (4, 20))


def fig10_generator_acceptance(out: Path, dest: Path) -> None:
    t = pd.read_csv(out / "analysis" / "generator_acceptance.csv")
    rps = sorted(t.rp.unique())
    x = np.arange(len(rps))
    fig, axes = plt.subplots(2, len(GEN_RANGES), figsize=(16, 7.4), sharex=True)
    for j, (lo, hi) in enumerate(GEN_RANGES):
        rc = f"({lo}, {hi}]"
        for i, (col, ylab) in enumerate((("t025_hr", "T(0.25) (h)"), ("nb24", "next-day high water / peak"))):
            ax = axes[i, j]
            g = t[(t.range_class == rc) & (t.variant == "generator_smooth")].set_index("rp").reindex(rps)
            ref = g[f"gauge_{col}"].to_numpy()
            ax.fill_between(x, 0.85 * ref, 1.15 * ref, color=GRID, alpha=0.8, linewidth=0, label="gauge +-15%")
            ax.plot(x, ref, color=INK, marker="o", linewidth=1.8, label="gauge (GESLA)")
            for variant, label, color, style in GEN_VARIANTS:
                v = t[(t.range_class == rc) & (t.variant == variant)].set_index("rp").reindex(rps)
                ax.plot(x, v[col], color=color, linestyle=style, marker="o", markersize=4, linewidth=1.5, label=label)
            hg = t[(t.range_class == rc) & (t.variant == "coast_hg")]
            if len(hg):
                ax.scatter([rps.index(100)], hg[col], color=GREEN, marker="D", s=40, zorder=4, label="COAST-HG (RP100)")
            ax.set_xticks(x, [str(r) for r in rps])
            if i == 1:
                ax.set_xlabel("return period (y)")
            if j == 0:
                ax.set_ylabel(ylab)
            _style(ax)
            draw_panel_letter(ax, "abcdefgh"[i * len(GEN_RANGES) + j])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, dest / "fig10_generator_acceptance.png")
    passes = t.groupby("variant").pass_shape.sum().astype(int)
    cells = t.groupby("variant").size()
    n = t[t.variant == "generator_smooth"].groupby("range_class").n_sites.max()
    describe(dest / "fig10_generator_acceptance.png", f"""
        Acceptance test of the per-RP storm-tide hydrograph generator against the tide gauges
        (boundary_test/generator_validation.py). For every quality site (tide set) hydrographs are
        built at the paired GTSM station for RP 2, 10, 25, 50 and 100 and compared with the
        gauge's own annual maxima of the matching empirical return period (within a factor 2 of
        the target RP; for RP 50 and 100, beyond the 30-46-year records, the gauge's events of RP
        20-45). Columns: gauges with a spring tidal range of (a, e) 0-1 m, (b, f) 1-2 m, (c, g)
        2-4 m, (d, h) more than 4 m ({', '.join(str(int(v)) for v in n)} sites). Top row: T(0.25),
        the hours the water level stays within a quarter of the peak (above mean sea level), the
        duration of the eikonal correction; bottom row: the next-day high water as a fraction of
        the peak. Each point is the median over the sites of a class.

        Black: the gauges, with the grey band of +-15% that the acceptance test allows. Blue solid:
        the generator as decided (FES2022 tide around a high water at the station's own tide
        level during extremes; surge shape = median normalised GTSM surge of the station's events
        of that RP window, smoothed by a 13-h running mean that removes the tidal-period ripple of
        the non-tidal residual; scaled to COAST-RP). Blue dotted: the same without smoothing.
        Orange: the HGRAPHER port (spring high water nearest MHWS, shape from events of matching
        surge size). Green diamond: COAST-HG at RP100. At tropical-cyclone RPs the surge shape
        comes from another quality gauge within 50 km, otherwise a Gaussian; a gauge is never
        scored with its own shape.

        Cells (tidal class x RP) within +-15% on all three measures (T(0.25), T(0.5), next-day
        high water): smoothed generator {passes['generator_smooth']} of {cells['generator_smooth']},
        raw generator {passes['generator']} of {cells['generator']}, HGRAPHER {passes['hgrapher']} of
        {cells['hgrapher']}, COAST-HG {passes['coast_hg']} of {cells['coast_hg']}. Data:
        generator_acceptance.csv and generator_scores.parquet.

        Reading: the generator fixes COAST-HG's main error. Its next-day high waters match the
        gauges in every class, and T(0.25) rises from COAST-HG's 5-6 h to within 10-20% of the
        gauges, falling with RP as observed. The remaining misses are T(0.25) at RP 2 on
        microtidal and macrotidal coasts (about 30% too short) and the RP 100 point, which is
        compared with RP 20-45 gauge events because nothing rarer is observed. Scaling to the gauge
        events' own peaks instead of COAST-RP leaves the RP 2 deficit unchanged, so it comes from
        the shape, not from COAST-RP.
        """)


def fig11_generator_profiles(out: Path, dest: Path) -> None:
    gp = np.load(out / "analysis" / "generator_profiles.npz")
    sites = _sites(out)
    rng = pd.Series(gp["file_name"]).map(sites.gesla_harm__spring_range_m).to_numpy()
    hours = np.arange(gp["profile"].shape[1]) - gp["profile"].shape[1] // 2
    shown = (((0, 1), 10), ((0, 1), 100), ((2, 4), 10), ((2, 4), 100))
    styles = (("gauge", "gauge (GESLA)", INK, "-"), ("generator_smooth", "generator (smoothed surge)", BLUE, "-"),
              ("hgrapher", "HGRAPHER port", ORANGE, "-"), ("coast_hg", "COAST-HG (RP100)", GREEN, "--"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), sharex=True, sharey=True)
    for ax, letter, ((lo, hi), rp) in zip(axes.ravel(), "abcd", shown):
        in_class = (rng > lo) & (rng <= hi) & (gp["rp"] == rp)
        for variant, label, color, style in styles:
            sel = in_class & (gp["variant"] == variant)
            if sel.any():
                ax.plot(hours, np.nanmedian(gp["profile"][sel], axis=0), color=color, linestyle=style, linewidth=1.6,
                        label=label)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set(xlim=(-WINDOW_PLOT_HR, WINDOW_PLOT_HR), ylim=(-1.0, 1.05))
        _style(ax)
        draw_panel_letter(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("hours from peak")
    for ax in axes[:, 0]:
        ax.set_ylabel("water level / peak")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, dest / "fig11_generator_profiles.png")
    describe(dest / "fig11_generator_profiles.png", """
        Median storm-tide hydrographs over the quality sites, normalised by their peak and
        centred on it (local mean sea level = 0): (a) spring range 0-1 m at RP 10, (b) 0-1 m at
        RP 100, (c) 2-4 m at RP 10, (d) 2-4 m at RP 100. Black: the gauges' own annual maxima of the
        matching empirical-RP window (median profile per site, then median over sites; at RP 100
        the gauge's RP 20-45 events). Blue: the generator with smoothed surge (as in fig10). Orange:
        the HGRAPHER port. Green dashed: COAST-HG (RP100 only). How each is built is described with
        fig10.

        Reading: on the 2-4 m coasts the generator's neighbouring high waters lie on the gauges'
        (c, d), where the HGRAPHER port and COAST-HG are 0.1-0.15 of the peak lower because their
        tide window is an MHWS-level spring tide (test 7). On microtidal coasts (a, b) the peaks and
        neighbouring high waters match, but all synthetic hydrographs drain lower between the high
        waters than the gauges, the generator to about -0.15 of the peak and COAST-HG to -0.3, where
        the observed events stay near mean sea level: the observed surge tail that holds the level up
        30-48 h from the peak (fig09) is not in their shapes.
        """)


PRODUCT_RPS = (2, 5, 10, 25, 50, 100, 250, 500, 1000)
PRODUCTION_MIN_LAT = -60.0  # prepare_boundary_conditions drops the Antarctic ice-shelf stations
EXAMPLE_SITES = (("Dunkerque (macrotidal)", 2.37, 51.05), ("Kiel (microtidal, surge)", 10.28, 54.50),
                 ("Galveston (TC)", -94.79, 29.31), ("Hiron Point, Bangladesh (TC)", 89.48, 21.78))
EXAMPLE_RPS = ((10, "#9ecae1"), (100, "#3182bd"), (1000, "#08306b"))


def _product(product: Path) -> pd.DataFrame:
    lev = pd.read_parquet(product / "levels.parquet")
    lev["has_tc"] = np.isfinite(lev.coast_rp_tc_rp100_m)
    return lev


def fig12_level_product_maps(product: Path, dest: Path) -> None:
    lev = _product(product)
    lev = lev[lev.lat > PRODUCTION_MIN_LAT]
    d = (lev.rp100_m - lev.coast_rp_rp100_m).dropna()
    d = d.iloc[np.argsort(np.abs(d.to_numpy()))]
    r = lev.r.iloc[np.argsort(np.abs(np.log(lev.r.to_numpy())))]
    panels = ((d, "RdBu_r", Normalize(-0.5, 0.5), "RP100 level, new - COAST-RP (m)", [-0.5, -0.25, 0, 0.25, 0.5],
               ["-0.5", "-0.25", "0", "0.25", "0.5"]),
              (r, "PuOr_r", LogNorm(1 / 1.5, 1.5), "surge ratio r (gauge / GTSM skew surge at RP10)",
               [1 / 1.5, 0.8, 1, 1.25, 1.5], ["0.67", "0.8", "1", "1.25", "1.5"]))
    fig = plt.figure(figsize=(10, 10.6))
    for k, (values, cmap, norm, label, ticks, ticklabels) in enumerate(panels):
        ax = fig.add_axes([0.02, 0.56 - k * 0.48, 0.96, 0.42], projection=MAP_PROJ)
        ax.set_global()
        _land(ax)
        sc = ax.scatter(lev.loc[values.index, "lon"], lev.loc[values.index, "lat"], c=values, s=3, cmap=cmap,
                        norm=norm, linewidths=0, transform=PC, zorder=2)
        ax.spines["geo"].set_edgecolor(MUTED)
        draw_panel_letter(ax, "ab"[k])
        cax = fig.add_axes([0.25, 0.54 - k * 0.48, 0.5, 0.014])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
        cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.set_ticks(ticks, labels=ticklabels)
        cb.set_label(label)
    _save(fig, dest / "fig12_level_product_maps.png")
    describe(dest / "fig12_level_product_maps.png", f"""
        The level product (ii) at the {d.size:,} COAST-RP stations north of 60 S, the ones production
        uses (Equal Earth projection; land in grey). (a) RP100 level of the new product minus COAST-RP's total RP100
        level: red = the new product is higher; differences beyond ±0.5 m take the end colours
        ({int((d.abs() > 0.5).sum()):,} stations). Stations closest to zero are drawn first. (b) The
        surge ratio r applied to the GTSM skew surges: the inverse-distance mean of the gauge ratios
        within 500 km (at least 3 gauges), otherwise the gauge median ({lev.r.median():.2f}), capped
        at 2. Orange = GTSM's surge is too small (r > 1); logarithmic scale, r above 1.5 in the end colour.

        Reading: most of the coast moves by less than 0.2 m. The large departures are regional:
        lower in the southern North Sea and Dover Strait, where GTSM's tide is too large and the
        FES tide replaces it; lower in Hudson Bay, the Canadian Arctic and parts of Indonesia; higher
        in the western Baltic, in Japan and East China, where r is above 1.25, in macrotidal basins
        where GTSM's tide is too small (Ungava Bay, Amazon mouth, King Sound), and on hurricane and
        typhoon coasts (US Gulf and south-east coasts, East China), where the ERA5 cyclones stay in
        the ETC fit and add to COAST-RP's TC part. At the TC-dominated gauges this centres the
        levels on the gauge (median -0.03 m against COAST-RP's -0.25 m) but widens the spread: one in
        ten is 0.5 m or more above it (fig14, fig21). Uniform r marks coasts without 3 gauges within
        500 km.
        """)


def fig13_level_product_by_rp(product: Path, dest: Path) -> None:
    lev = _product(product)
    comparisons = (("rp{}_m", "coast_rp_rp{}_m", "total: new - COAST-RP", INK, "-"),
                   ("etc_raw_rp{}_m", "coast_rp_etc_rp{}_m", "ETC with r = 1 (FES tide only) - COAST-RP ETC",
                    BLUE, "--"),
                   ("etc_rp{}_m", "coast_rp_etc_rp{}_m", "ETC with r - COAST-RP ETC", ORANGE, "-"))
    x = np.arange(len(PRODUCT_RPS))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, letter, (label, sel) in zip(axes, "ab", (("no TC part", ~lev.has_tc), ("TC part", lev.has_tc))):
        ax.axvspan(x[PRODUCT_RPS.index(250)] - 0.5, x[-1] + 0.3, color=GRID, alpha=0.5, linewidth=0)
        for col, ref, name, color, style in comparisons:
            q = np.stack([(lev[col.format(rp)] - lev[ref.format(rp)])[sel].dropna().quantile([0.1, 0.5, 0.9])
                          for rp in PRODUCT_RPS])
            if color == INK:
                ax.fill_between(x, q[:, 0], q[:, 2], color=INK, alpha=0.1, linewidth=0, label="total, 10-90%")
            ax.plot(x, q[:, 1], color=color, linestyle=style, linewidth=2, marker="o", markersize=4, label=name)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set(xticks=x, xticklabels=[str(rp) for rp in PRODUCT_RPS], xlabel="return period (y)",
               xlim=(x[0] - 0.3, x[-1] + 0.3))
        ax.text(0.03, 0.97, f"{label} ({int(sel.sum()):,} stations)", transform=ax.transAxes, va="top",
                fontsize=9, color=INK)
        _style(ax)
        draw_panel_letter(ax, letter)
    axes[0].set_ylabel("difference, median over stations (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, dest / "fig13_level_product_by_rp.png")
    describe(dest / "fig13_level_product_by_rp.png", """
        The new level product against COAST-RP per return period, (a) at stations without a TC
        component in COAST-RP, (b) at stations with one. Lines are medians over stations of three
        differences. Black: the total level (grey band: 10-90% of stations). Blue dashed: the ETC
        level with r = 1 minus COAST-RP's ETC level, i.e. only the FES tide replacing GTSM's.
        Orange: the ETC level with the surge ratio r minus COAST-RP's ETC level. Shaded: RPs above
        250 y, which extrapolate ~46 years of skew surges.

        Reading: without TC (a), the FES tide lowers the levels by 0.05-0.08 m (GTSM's tide is 7-8%
        too large) and r raises them by about 0.09 m, so the total rises by 0.01-0.07 m; orange and
        black coincide because there is no TC part. With TC (b), r is applied to the ETC part only
        while the TC part is COAST-RP's, so the total rises less than the ETC part at high RPs, where
        TC dominates the combined rate.
        """)


def fig14_level_product_gauges(product: Path, dest: Path) -> None:
    g = pd.read_parquet(product / "gauge_check.parquet")
    groups = (("ETC-dominated gauges", g.station_class != "tc_dominated", BLUE),
              ("TC-dominated gauges", g.station_class == "tc_dominated", ORANGE))
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for row, rp in enumerate((10, 100)):
        ref = g[f"gauge_rp{rp}_m"]
        hi = float(np.nanmax(g[[f"gauge_rp{rp}_m", f"coast_rp_rp{rp}_m", f"new_loo_rp{rp}_m"]].to_numpy())) * 1.03
        for ax, letter, (col, label) in zip(axes[row], "abcd"[2 * row:],
                                            (("coast_rp", "COAST-RP"), ("new_loo", "new product, r left out"))):
            ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1)
            lines = ["station - gauge: median / MAE"]
            for name, sel, color in groups:
                e = (g[f"{col}_rp{rp}_m"] - ref)[sel].dropna()
                ax.scatter(ref[sel], g.loc[sel, f"{col}_rp{rp}_m"], s=10, color=color, alpha=0.6, linewidths=0,
                           label=name)
                lines.append(f"{name.split()[0]}: {e.median():+.3f} / {e.abs().mean():.3f} m (n = {len(e)})")
            draw_caption_box(ax, lines, loc="lower right")
            ax.set(xlim=(0, hi), ylim=(0, hi), aspect="equal", xlabel=f"gauge RP{rp} (m)",
                   ylabel=f"{label} RP{rp} (m)")
            _style(ax)
            draw_panel_letter(ax, letter)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, dest / "fig14_level_product_gauges.png")
    describe(dest / "fig14_level_product_gauges.png", f"""
        RP levels at the test 9 gauges that have a COAST-RP station within 25 km ({len(g)} gauges)
        against the gauge's own skew-surge convolution (gauge surge on the gauge tide, the reference
        of test 9). Left: COAST-RP's total level at the nearest station. Right: the new product at
        that station with r rebuilt without the gauge (leave-one-out), so that no gauge calibrates
        itself. Top RP10, bottom RP100. Blue: gauges where ETC dominates; orange: TC-dominated gauges
        (COAST-RP's TC level above its ETC level at RP100). Boxes: median and mean absolute error of
        station - gauge per group. Levels in m above local mean sea level.
        """)


def fig15_level_product_hydrographs(product: Path, dest: Path) -> None:
    lev = _product(product)
    info = pd.read_parquet(product / "hydrograph_info.parquet").set_index(["station", "rp"])
    hyd = np.load(product / "hydrographs.npy", mmap_mode="r")
    hours = (np.arange(hyd.shape[2]) - hyd.shape[2] // 2) / 6.0
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True)
    for ax, letter, (name, lon, lat) in zip(axes.ravel(), "abcd", EXAMPLE_SITES):
        k = int(np.argmin(((lev.lon - lon) * np.cos(np.radians(lat))) ** 2 + (lev.lat - lat) ** 2))
        st = lev.index[k]
        for rp, color in EXAMPLE_RPS:
            source = info.loc[(st, rp), "shape_source"].replace("_", " ")
            ax.plot(hours, hyd[k, PRODUCT_RPS.index(rp)], color=color, linewidth=1.5, label=f"RP{rp} ({source})")
            ax.axhline(lev.loc[st, f"coast_rp_rp{rp}_m"], color=color, linestyle=":", linewidth=1.2,
                       label="COAST-RP at the same RP" if rp == EXAMPLE_RPS[-1][0] else None)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        y0, y1 = ax.get_ylim()
        ax.set(xlim=(-72, 72), ylim=(y0, y1 + 0.45 * (y1 - y0)))
        ax.text(0.02, 0.97, f"{name}\n{lev.loc[st, 'station_id']}, r = {lev.loc[st, 'r']:.2f}",
                transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        _style(ax)
        draw_panel_letter(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel("hours from peak")
    for ax in axes[:, 0]:
        ax.set_ylabel("water level above MSL (m)")
    fig.tight_layout()
    _save(fig, dest / "fig15_level_product_hydrographs.png")
    describe(dest / "fig15_level_product_hydrographs.png", """
        Storm-tide hydrographs of the level product at four example stations (the COAST-RP station
        nearest each named place) for RP10, RP100 and RP1000, 10-min steps, ±72 h around the peak.
        Each peak is the product's RP level; dotted lines mark COAST-RP's total level at the same RP.
        The legend names the surge shape: gtsm events (median smoothed surge of the station's GTSM
        annual maxima of matching empirical RP), tc gauge (median surge-dominated event of a gauge
        within 50 km), tc gaussian (Gaussian placeholder where there is no such gauge).

        Reading: at Dunkerque the levels are 0.6-0.85 m below COAST-RP, which carries GTSM's too-large
        tide; at Kiel they are 0.3-0.9 m above it, where the Baltic gauges side with the new product
        (fig14). At the TC-dominated Galveston and Hiron Point the RP100 and RP1000 levels are
        COAST-RP's (its TC rate dominates there); only the shape is new.
        """)


def fig16_station_interpolation(product: Path, dest: Path) -> None:
    table = pd.read_csv(product / "interpolation.csv")
    spacing = pd.read_csv(product / "interpolation_spacing.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    t100 = table[table.rp == 100]
    nearest = float(t100[t100.k == 1].mae_m.iloc[0])
    for power, color in ((1, GREEN), (2, BLUE), (3, ORANGE)):
        sel = t100[(t100.power == power) & (t100.k > 1)]
        ax.plot(sel.k, sel.mae_m * 100, color=color, marker="o", markersize=4, linewidth=1.8, label=f"IDW power {power}")
    ax.axhline(nearest * 100, color=INK, linestyle="--", linewidth=1, label="nearest station")
    former = t100[(t100.k == 15) & (t100.power == 2)].mae_m.iloc[0] * 100
    prod = t100[(t100.k == 5) & (t100.power == 3)].mae_m.iloc[0] * 100
    best = float(t100.mae_m.min())
    ax.annotate("former\n(k = 15, power 2)", (15, former), xytext=(17, former + 1.2), fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.annotate("production\n(k = 5, power 3)", (5, prod), xytext=(2.3, prod + 2.4), fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set(xscale="log", xticks=[2, 3, 5, 8, 15, 30], xticklabels=["2", "3", "5", "8", "15", "30"],
           xlabel="neighbours k", ylabel="RP100 leave-one-out MAE (cm)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = axes[1]
    x = np.arange(len(spacing))
    edges = [s.strip("(]").split(", ") for s in spacing.index]
    labels = [(f"{float(lo):g}-{float(hi):g}" if hi != "inf" else f"> {float(lo):g}") + f"\n(n = {n:,})"
              for (lo, hi), n in zip(edges, spacing.stations)]
    width = 0.27
    for j, (col, color) in enumerate((("nearest", INK), ("k = 15, p = 2 (former)", MUTED), ("k = 5, p = 3 (production)", BLUE))):
        ax.bar(x + (j - 1) * width, spacing[col] * 100, width * 0.92, color=color, label=col)
    ax.set(xticks=x, xticklabels=labels, xlabel="distance to the nearest other station (km)",
           ylabel="RP100 leave-one-out MAE (cm)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "b")
    fig.tight_layout()
    _save(fig, dest / "fig16_station_interpolation.png")
    describe(dest / "fig16_station_interpolation.png", f"""
        How well the level at a coastline cell can be interpolated from the stations around it,
        tested by leaving each station out and predicting its RP100 level (new product) from the
        others ({int(spacing.stations.sum()):,} stations north of 60 S). (a) Mean absolute error of
        inverse-distance weighting over the k nearest stations (haversine distance, floored at 1 km)
        for three weighting powers, and of the nearest station alone (dashed). (b) The error by
        the distance from the left-out station to its nearest neighbour, for the nearest station,
        the former production setting (k = 15, power 2) and the current one (k = 5, power 3).
        Production also restricts the neighbours to the tile's ocean component, which this global
        test does not.

        Reading: fewer neighbours or a higher power interpolate the level field better than the
        former k = 15, power 2, by about 1.5 cm of MAE ({former:.1f} cm there, {prod:.1f} cm at
        k = 5, power 3, {best * 100:.1f} cm at best, {nearest * 100:.1f} cm for the nearest station).
        The error is set by station spacing: 2 cm where stations are within 5 km, 8-12 cm at
        10-50 km and about 20 cm beyond 50 km.
        """)


def fig17_level_product_qa(product: Path, dest: Path) -> None:
    qa = np.load(product / "qa_examples.npz")
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.6), gridspec_kw={"width_ratios": (1.6, 1)})
    for row, (k, label) in enumerate(zip(qa["stations"], qa["labels"])):
        ax = axes[row, 0]
        tide, residual = qa[f"{k}_tide"], qa[f"{k}_residual"]
        hours = np.arange(len(tide)) - len(tide) // 2
        ax.plot(hours, tide, color=BLUE, linewidth=1.4, label="FES tide")
        ax.plot(hours, residual, color=ORANGE, linewidth=1.4, label="GTSM surge residual")
        ax.plot(hours, tide + residual, color=INK, linewidth=1, linestyle=":", label="sum")
        ax.scatter(qa[f"{k}_hw_old"], tide[qa[f"{k}_hw_old"] + len(tide) // 2], marker="x", color=MUTED, s=30,
                   zorder=3, label="high waters, every local maximum")
        ax.scatter(qa[f"{k}_hw_new"], tide[qa[f"{k}_hw_new"] + len(tide) // 2], marker="o", facecolor="none",
                   edgecolor=GREEN, s=50, zorder=3, label="high waters, prominent only")
        ax.text(0.01, 0.97, label, transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set(xlim=(hours[0], hours[-1]), ylabel="m above MSL")
        _style(ax)
        draw_panel_letter(ax, "ace"[row])
        ax = axes[row, 1]
        for name, color, text in (("old", MUTED, "before"), ("new", GREEN, "after")):
            v = qa[f"{k}_ranked_{name}"]
            ax.plot(np.arange(1, len(v) + 1), v, color=color, marker=".", linestyle="none", markersize=4,
                    label=f"{text} QA")
        ax.set(xscale="log", ylabel="skew surge (m)")
        _style(ax)
        draw_panel_letter(ax, "bdf"[row])
    axes[2, 0].set_xlabel("hours from the largest skew surge (before QA)")
    axes[2, 1].set_xlabel("rank")
    axes[0, 1].legend(frameon=False, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, dest / "fig17_level_product_qa.png")
    describe(dest / "fig17_level_product_qa.png", """
        Two failure modes the level product's QA removes, and one it keeps, one station each. Left: the hourly
        FES tide (blue), GTSM-ERA5 surge residual (orange) and their sum (dotted) around the
        station's largest skew surge before QA; grey crosses mark every local maximum of the tide
        (the high waters before QA), green circles the high waters that remain when a maximum must
        have a prominence of at least 10% of the 25-h tidal range. Right: the 300 largest skew
        surges before (grey) and after (green) QA.

        (a, b) Ungava Bay: the shallow-water tide has a second low-water bump. Taken as a high
        water, it makes the next real high water a "surge" of several metres, and those spurious
        skew surges set the GPD tail (shape 1.1 before, -0.1 after). (c, d) Arafura coast: GTSM's
        residual jumps by up to 12 m within an hour (a numerical artefact of the model there).
        Despiking (hours more than 2 m from their 5-h median) removes the shortest spikes but not
        all; stations with 5 or more such jumps take COAST-RP's levels instead. (e, f) Mississippi
        delta: Katrina and other hurricanes are in the ERA5 record and stay in the ETC fit, although
        COAST-RP's synthetic TC part is added on top. Removing them makes TC coasts worse at the
        gauges (fig21), because both TC sources are too low.
        """)


DURATION_RANGES = ((0, 1), (1, 2), (2, 4), (4, 20))
DURATION_SERIES = (("coast_hg", "COAST-HG RP100 (every RP)", INK, "--"), ("rp2", "product RP2", "#9ecae1", "-"),
                   ("rp10", "product RP10", "#6baed6", "-"), ("rp100", "product RP100", "#3182bd", "-"),
                   ("rp1000", "product RP1000", "#08306b", "-"))


def fig18_product_durations(product: Path, dest: Path) -> None:
    d = pd.read_parquet(product / "durations.parquet")
    d = d[d.lat > PRODUCTION_MIN_LAT]
    spring = 2 * d.mhws_m
    fig = plt.figure(figsize=(12, 9.4))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.15))
    for j, (m, label) in enumerate((("t025_hr", "T(0.25) (h)"), ("t05_hr", "T(0.5) (h)"))):
        ax = fig.add_subplot(grid[0, j])
        x = np.arange(len(DURATION_RANGES))
        width = 0.16
        for s, (key, name, color, _) in enumerate(DURATION_SERIES):
            med = [d[f"{key}_{m}"][(spring > lo) & (spring <= hi)].median() for lo, hi in DURATION_RANGES]
            ax.bar(x + (s - 2) * width, med, width * 0.9, color=color if key != "coast_hg" else MUTED, label=name)
        ax.set(xticks=x, xticklabels=[f"{lo}-{hi} m" if hi < 20 else f"> {lo} m" for lo, hi in DURATION_RANGES],
               xlabel="spring tidal range (2 x MHWS, FES)", ylabel=f"median {label}")
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
        _style(ax)
        draw_panel_letter(ax, "ab"[j])
    ratio = (d.rp100_t025_hr / d.coast_hg_t025_hr).replace([np.inf, 0], np.nan).dropna()
    ratio = ratio.iloc[np.argsort(np.abs(np.log(ratio.to_numpy())))]
    ax = fig.add_subplot(grid[1, :], projection=MAP_PROJ)
    ax.set_global()
    _land(ax)
    sc = ax.scatter(d.loc[ratio.index, "lon"], d.loc[ratio.index, "lat"], c=ratio, s=3, cmap="PuOr_r",
                    norm=LogNorm(1 / 4, 4), linewidths=0, transform=PC, zorder=2)
    ax.spines["geo"].set_edgecolor(MUTED)
    draw_panel_letter(ax, "c")
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", extend="both", fraction=0.04, pad=0.03, shrink=0.5)
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=["1/4", "1/2", "1", "2", "4"])
    cb.set_label("T(0.25) at RP100, product / COAST-HG")
    fig.tight_layout()
    _save(fig, dest / "fig18_product_durations.png")
    med = {f"{k}_{m}": d[f"{k}_{m}"].median() for k, *_ in DURATION_SERIES for m in ("t025_hr", "t05_hr")}
    describe(dest / "fig18_product_durations.png", f"""
        Peak durations of the two hydrograph products at the {len(d):,} COAST-RP stations north of
        60 S. T(f) = hours the hydrograph stays within f x peak of its peak (hourly +-72 h profile
        normalised by the peak, relative to local MSL; as in the acceptance test of fig10).
        COAST-HG has one RP100 hydrograph per station, which production uses for every RP; the
        level product has one per RP. (a, b) Median T(0.25) and T(0.5) by spring tidal range.
        (c) T(0.25) at RP100, product over COAST-HG; orange = the product's peak is wider.

        Reading: the product's peaks are wider than COAST-HG's: median T(0.25)
        {med["rp100_t025_hr"]:.0f} h against {med["coast_hg_t025_hr"]:.0f} h at RP100 and T(0.5)
        {med["rp100_t05_hr"]:.0f} h against {med["coast_hg_t05_hr"]:.0f} h. T falls with RP
        (T(0.25) {med["rp2_t025_hr"]:.0f} h at RP2, {med["rp1000_t025_hr"]:.0f} h at RP1000), as the
        gauges show (fig10), which a single COAST-HG shape cannot. The duration correction B reads
        T from these hydrographs, so a longer T lowers its friction multiplier.
        """)


GUESS_NEAR_GAUGE_KM = 150.0


def _guesses(product: Path) -> Path:
    return product / "open_guesses"


def fig19_despike(product: Path, dest: Path) -> None:
    d = pd.read_parquet(_guesses(product) / "despike.parquet")
    d = d[d.lat > PRODUCTION_MIN_LAT]
    fig = plt.figure(figsize=(12, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.2))
    ax = fig.add_subplot(grid[0, 0])
    bins = np.logspace(np.log10(0.01), np.log10(20), 60)
    ax.hist(d.max_jump_m.clip(0.01), bins=bins, color=BLUE)
    for m in (1, 2, 3):
        ax.axvline(m, color=MUTED, linestyle=":" if m != 2 else "--", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="largest hourly change of the GTSM surge residual (m)",
           ylabel="stations")
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = fig.add_subplot(grid[0, 1])
    counts = (1, 2, 3, 5, 10, 20)
    table = np.array([[int((d[f"jumps_gt_{m:g}m"] >= n).sum()) for n in counts] for m in (1.0, 2.0, 3.0)])
    ax.imshow(table, cmap="Blues", norm=LogNorm(1, max(table.max(), 2)), aspect="auto")
    for (r, c), v in np.ndenumerate(table):
        ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=9,
                color="white" if v > np.sqrt(table.max()) else INK, fontweight="bold" if (r, c) == (1, 3) else None)
    ax.set(xticks=range(len(counts)), xticklabels=[f">= {n}" for n in counts], yticks=range(3),
           yticklabels=["> 1 m", "> 2 m", "> 3 m"], xlabel="hourly jumps per station (46 years)",
           ylabel="jump threshold")
    ax.set_title("stations sent to COAST-RP levels (bold: production)", fontsize=9, color=INK)
    draw_panel_letter(ax, "b")
    ax = fig.add_subplot(grid[1, :], projection=MAP_PROJ)
    ax.set_global()
    _land(ax)
    j = d[d.jumps_gt_1m > 0].sort_values("jumps_gt_1m")
    sc = ax.scatter(j.lon, j.lat, c=j.jumps_gt_1m, s=8, cmap="viridis", norm=LogNorm(1, max(j.jumps_gt_1m.max(), 2)),
                    transform=PC, zorder=2, linewidths=0)
    fb = d[d.level_source == "coast_rp"]
    ax.scatter(fb.lon, fb.lat, s=40, facecolor="none", edgecolor=ORANGE, linewidths=1.2, transform=PC, zorder=3)
    ax.spines["geo"].set_edgecolor(MUTED)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.03, shrink=0.5)
    cb.set_label("hourly jumps > 1 m (orange rings: stations on COAST-RP levels)")
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig19_despike.png")
    n1 = int((d.jumps_gt_1m > 0).sum())
    describe(dest / "fig19_despike.png", f"""
        Jumps in the GTSM-ERA5 surge residual at the {len(d):,} stations north of 60 S (1979-2024,
        hourly). (a) Each station's largest change from one hour to the next. (b) How many stations
        a QA rule "at least N jumps above X m" would send to COAST-RP's levels; production uses 5
        jumps above 2 m (bold). (c) Stations with any jump above 1 m, coloured by their number,
        and the stations on COAST-RP levels (orange rings).

        Reading: the median station's largest jump is {d.max_jump_m.median() * 100:.0f} cm and
        {n1:,} stations ever change by more than 1 m within an hour. Above 2 m the jumps are few
        and clustered on the Arafura coast of New Guinea, where they reach {d.max_jump_m.max():.0f} m,
        far beyond any storm. With a 2 m threshold, 3 to 5 required jumps pick the same 15
        stations, so the result does not hang on the exact count; despiking (hours more than 2 m
        from their 5-h median) cleans the 7 with only one or two. A 1 m threshold would also take
        real fast surges: outside the Arafura cluster, the stations with 20 or more jumps above 1 m
        are in the southern North Sea
        and the German Bight, where storm surges rise by a metre within an hour.
        """)


def fig20_gpd_cap(product: Path, dest: Path) -> None:
    g = _guesses(product)
    sc = pd.read_parquet(g / "cap_scores.parquet")
    st = pd.read_parquet(g / "cap_stations.parquet")
    shapes = pd.read_parquet(g / "cap_gauge_shapes.parquet")
    lev = _product(product)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    bins = np.linspace(-1.0, 1.5, 51)
    ax.hist(lev.gpd_shape_mle.dropna().clip(-1, 1.5), bins=bins, density=True, color=BLUE, alpha=0.7,
            label=f"GTSM on FES, {lev.gpd_shape_mle.notna().sum():,} stations")
    ax.hist(shapes.shape_gauge.dropna().clip(-1, 1.5), bins=bins, density=True, histtype="step", color=INK,
            linewidth=1.5, label=f"gauges, {shapes.shape_gauge.notna().sum()}")
    for c in (0.2, 0.25, 0.35, 0.5):
        ax.axvline(c, color=ORANGE if c == 0.35 else MUTED, linestyle="--", linewidth=1)
    ax.set(xlabel="GPD shape (maximum likelihood)", ylabel="density")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = axes[1]
    caps = list(dict.fromkeys(sc.level))
    x = np.arange(len(caps))
    for k, (label, color) in enumerate((("non-TC-dominated gauges", BLUE), ("TC-dominated gauges", ORANGE))):
        s = sc[(sc.gauges == label) & (sc.rp == 100)].set_index("level").reindex(caps)
        ax.plot(x, s["mae"] * 100, color=color, marker="o", label=f"{label}: MAE")
        ax.plot(x, s["median"] * 100, color=color, marker="s", linestyle="--", label=f"{label}: median")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set(xticks=x, xticklabels=[c.replace(" (production)", "\n(production)") for c in caps], xlabel="shape cap",
           ylabel="RP100 station - gauge (cm)")
    ax.legend(frameon=False, fontsize=7, loc="center")
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = axes[2]
    prod = st[st.cap == "0.35 (production)"].set_index("station")
    for c, color in (("none", INK), ("0.5", MUTED), ("0.25", GREEN), ("0.2", BLUE)):
        s = st[st.cap == c].set_index("station")
        dd = (s.rp1000_m - prod.rp1000_m.reindex(s.index)).dropna()
        ax.scatter(prod.shape_mle.reindex(dd.index), dd, s=6, color=color, label=f"cap {c}", alpha=0.7, linewidths=0)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set(xlabel="station GPD shape (maximum likelihood)", ylabel="RP1000 minus production (m)",
           ylim=(-3.5, 10.0))
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    _style(ax)
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig20_gpd_cap.png")
    m = sc[(sc.rp == 100) & (sc.gauges == "all")].set_index("level")
    n35 = int((lev.gpd_shape_mle > 0.35).sum())
    describe(dest / "fig20_gpd_cap.png", f"""
        The GPD shape cap. (a) Maximum-likelihood shapes of the GTSM-on-FES skew surges at all
        stations and of the gauges' own skew surges (test 9), with the tested caps (orange:
        production, 0.35). (b) RP100 level minus the gauge's convolution at the ratio gauges, r
        left one out, for each cap (non-TC- and TC-dominated gauges; MAE solid, median dashed).
        (c) RP1000 at the stations with a shape above 0.2, relative to production, against the
        station's shape.

        Reading: the cap binds at {n35} stations. At the gauges it makes no measurable difference
        up to RP100 (all gauges, MAE {m.loc['none', 'mae'] * 100:.1f} cm without a cap,
        {m.loc['0.35 (production)', 'mae'] * 100:.1f} cm at 0.35, {m.loc['0.2', 'mae'] * 100:.1f} cm
        at 0.2): the gauges rarely sit where GTSM's tail is heavy, and RP100 is still close to the
        data. The cap acts on RP1000 at the heavy-tailed stations, most of them on coasts a major
        hurricane or typhoon struck in 1979-2024: without it RP1000 rises by up to 12.9 m (0.5:
        8.0 m), and 0.2 lowers it by up to 3.9 m. The gauges' own shapes (median
        0.01, standard deviation 0.25) spread more widely than GTSM's (-0.03, 0.19), so they do not
        fix a cap; 0.35 bounds the extreme cases only.
        """)


def fig21_tc_mask(product: Path, dest: Path) -> None:
    sc = pd.read_parquet(_guesses(product) / "tc_mask_scores.parquet")
    variants = list(dict.fromkeys(sc.level))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, letter, label in zip(axes, "ab", ("TC-dominated gauges", "stations with a TC part")):
        y = np.arange(len(variants))
        for k, (rp, color) in enumerate(((10, BLUE), (100, ORANGE))):
            s = sc[(sc.gauges == label) & (sc.rp == rp)].set_index("level").reindex(variants)
            ax.barh(y + (k - 0.5) * 0.38, s["median"] * 100, 0.36, color=color, label=f"RP{rp} median")
            ax.scatter(s["mae"] * 100, y + (k - 0.5) * 0.38, marker="|", s=120, color=INK,
                       label="MAE" if k == 0 else None)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set(yticks=y, yticklabels=variants, xlabel="station - gauge (cm)",
               title=f"{label} (n = {int(sc[(sc.gauges == label)].n.iloc[0])})")
        ax.title.set_fontsize(9)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
        draw_panel_letter(ax, letter)
    fig.tight_layout()
    _save(fig, dest / "fig21_tc_mask.png")
    s = sc[(sc.gauges == "TC-dominated gauges") & (sc.rp == 100)].set_index("level")
    describe(dest / "fig21_tc_mask.png", f"""
        A tropical-cyclone mask, tested and not used in production: at stations with a COAST-RP TC
        part, tidal cycles near an IBTrACS storm would leave the ETC fit, so a cyclone would not
        count twice (in the ERA5 surge record and in COAST-RP's synthetic TC part). Bars: median of station minus gauge at RP10
        and RP100 at the ratio gauges, r left one out; ticks: MAE. (a) TC-dominated gauges, (b)
        gauges whose station has a TC part. Variants: no mask (production); 300 km, +-24 h, >= 34
        kt; 150 and 500 km; +-12 and +-48 h; hurricane strength (>= 64 kt) only.

        Reading: the mask makes the TC coasts worse at the gauges. Without it the TC-dominated
        gauges' RP100 median is {s.loc['no mask (production)', 'median'] * 100:.0f} cm and the MAE
        {s.loc['no mask (production)', 'mae'] * 100:.0f} cm; with a 300 km mask
        {s.loc['300 km, +-24 h, 34 kt', 'median'] * 100:.0f} cm and
        {s.loc['300 km, +-24 h, 34 kt', 'mae'] * 100:.0f} cm. A smaller radius (150 km)
        lies between; the time window and the wind threshold hardly matter. Without the mask a
        cyclone counts in the ERA5 record and again in COAST-RP's TC part, yet the levels match
        the gauges better: both TC sources are too low (GTSM-ERA5 resolves cyclones coarsely,
        COAST-RP's TC part is 0.2-0.25 m below the gauges, §8.1), and the double count makes up
        for it. The price is a wider spread: without the mask one in ten TC-dominated gauges is
        {s.loc['no mask (production)', 'p90'] * 100:.0f} cm or more below its station, with the 300 km
        mask {s.loc['300 km, +-24 h, 34 kt', 'p90'] * 100:.0f} cm, because a cyclone of the 46-year
        record raises the ETC tail where it struck. Production keeps the cyclones in the ETC fit
        until test 10 (STORM-forced GTSM) replaces the TC part.
        """)


def fig22_fes_rule(product: Path, dest: Path) -> None:
    g = _guesses(product)
    gauges = pd.read_parquet(g / "fes_rule_gauges.parquet").dropna()
    st = pd.read_parquet(g / "fes_rule_stations.parquet")
    f = st[st.atlas == "fes2014"].copy()
    d, i = BallTree(np.radians(gauges[["lat", "lon"]].to_numpy()), metric="haversine").query(
        np.radians(f[["lat", "lon"]].to_numpy()), k=1)
    f["gauge_km"], f["gauge_mhws_m"] = d[:, 0] * 6371.0, gauges.mhws_gauge_m.to_numpy()[i[:, 0]]
    f["gauge"] = gauges.index[i[:, 0]]
    near = f[f.gauge_km <= GUESS_NEAR_GAUGE_KM]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    e22 = gauges.mhws_fes2022_m - gauges.mhws_gauge_m
    e14 = gauges.mhws_fes2014_m - gauges.mhws_gauge_m
    flag = np.asarray(version_flag(gauges.mhws_fes2022_m, gauges.mhws_fes2014_m), dtype=bool)
    ax.scatter(e22[~flag], e14[~flag], s=8, color=MUTED, linewidths=0, label="rule silent")
    ax.scatter(e22[flag], e14[flag], s=40, color=ORANGE, label="rule fires")
    lim = (-1.5, 5.0)
    ax.plot(lim, lim, color=MUTED, linewidth=0.8)
    ax.set(xlim=lim, ylim=(-1.5, 1.5), xlabel="FES2022 - gauge MHWS (m)", ylabel="FES2014 - gauge MHWS (m)")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = axes[1]
    for flagged, marker, label in ((True, "o", "rule"), (False, "^", "grown to neighbours")):
        s = near[near.rule_flag == flagged]
        ax.scatter(s.gauge_mhws_m, s.mhws_fes2022_m, marker=marker, s=30, color=BLUE, label=f"FES2022 ({label})")
        ax.scatter(s.gauge_mhws_m, s.mhws_fes2014_m, marker=marker, s=30, facecolor="none", edgecolor=ORANGE,
                   label=f"FES2014 ({label})")
    hi = float(near[["mhws_fes2022_m", "gauge_mhws_m"]].max().max()) * 1.05
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=0.8)
    ax.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"nearest gauge MHWS (m), within {GUESS_NEAR_GAUGE_KM:.0f} km",
           ylabel="station MHWS (m)")
    for gauge, grp in near.groupby("gauge"):
        ax.annotate(gauge.split("-")[0].replace("_", " "), (grp.gauge_mhws_m.iloc[0], grp.mhws_fes2022_m.max()),
                    fontsize=7, color=INK, xytext=(3, 3), textcoords="offset points")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = axes[2]
    far = f[f.gauge_km > GUESS_NEAR_GAUGE_KM]
    ax.scatter(far.mhws_fes2014_m, far.mhws_fes2022_m, s=25, color=INK)
    x = np.linspace(0, 7, 50)
    ax.plot(x, np.maximum(VERSION_RATIO * x, x + VERSION_DIFF_M), color=ORANGE, linewidth=1, label="rule boundary")
    ax.set(xlabel="FES2014 MHWS (m)", ylabel="FES2022 MHWS (m)", title=f"{len(far)} stations without a gauge")
    ax.title.set_fontsize(9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig22_fes_rule.png")
    n_rule, n_grown = int(f.rule_flag.sum()), int((~f.rule_flag).sum())
    describe(dest / "fig22_fes_rule.png", f"""
        The FES version rule (FES2014 where FES2022's MHWS exceeds it by a factor
        {VERSION_RATIO} and by {VERSION_DIFF_M} m) and its growth to neighbours on the same
        artefact. It sets FES2014 at {len(f)} stations ({n_rule} by the rule, {n_grown} by the
        growth). (a) At the {len(gauges)} GESLA gauges: FES2022 and FES2014 MHWS minus the gauge's
        own; orange where the rule would fire. (b) The FES2014 stations with a gauge within
        {GUESS_NEAR_GAUGE_KM:.0f} km: both atlases' MHWS at the station against the gauge's
        (circles: the rule, triangles: grown). (c) The FES2014 stations without such a gauge, with
        the rule's boundary.

        Reading: where the rule can be checked it is right: at the {int(flag.sum())} gauges where
        it fires (Juneau, Skagway, Port Alberni), FES2022 is 1.5-4.7 m too high and FES2014 within
        0.1 m (0.6 m at Juneau). The
        flagged stations near gauges sit with FES2014 at the gauge's level (Varangerfjord against
        Vardo 1.35 m, Le Crouesty, Karachi for Sonmiani, Kukup, Port Stanley for the Falklands).
        Sao Marcos Bay: the Ponta da Madeira gauge inside the bay reads 2.62 m, so the outer
        stations' FES2014 (2.1 m) is the plausible value, not FES2022 (3.2-3.8 m). One flag is
        wrong: at the Saguenay mouth the Tadoussac gauge (2.18 m) sides with FES2022 (2.65 m)
        over FES2014 (1.38 m). Without gauges (c), most flags are FES2022 artefacts of known
        basins (Magellan, Gulf of California, Ungava), but some sit in real macrotidal gulfs
        (Khambhat, Incheon) where neither atlas is checked.
        """)


def fig23_peak_offsets(product: Path, dest: Path) -> None:
    p = pd.read_parquet(_guesses(product) / "peaks.parquet")
    p = p[p.lat > PRODUCTION_MIN_LAT]
    off = p[p.offset_hr != 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    classes = pd.cut(p.form_factor, [0, 0.25, 1.5, 3, 100], labels=["semidiurnal", "mixed, mainly semid.",
                                                                     "mixed, mainly diurnal", "diurnal"])
    share = p.assign(off=p.offset_hr != 0, cls=classes).pivot_table(index="rp", columns="cls", values="off",
                                                                     aggfunc="mean", observed=True) * 100
    for (cls, color) in zip(share.columns, (BLUE, GREEN, ORANGE, INK)):
        ax.plot(np.arange(len(share)), share[cls], marker="o", color=color, label=str(cls))
    ax.set(xticks=np.arange(len(share)), xticklabels=share.index, xlabel="return period (y)",
           ylabel="hydrographs with the maximum off time 0 (%)")
    ax.legend(frameon=False, fontsize=8, title="tide (form factor)", title_fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = axes[1]
    ax.hist(off.offset_hr.clip(-60, 60), bins=np.arange(-60.5, 61, 1), color=BLUE)
    ax.set(xlabel="hours from time 0 to the maximum", ylabel="hydrographs")
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = axes[2]
    ax.hist(off.excess_m.clip(0, 1.0), bins=50, color=BLUE)
    ax.set(xlabel="maximum minus the level at time 0 (m)", ylabel="hydrographs", yscale="log")
    _style(ax)
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig23_peak_offsets.png")
    rp100 = p[p.rp == 100]
    describe(dest / "fig23_peak_offsets.png", f"""
        Hydrographs whose maximum is not at time 0, the high water the generator's surge is
        centred on. (a) Their share by return period and tidal type (form factor (K1 + O1) /
        (M2 + S2) of the FES tide). (b) Where the maximum falls instead. (c) How far it exceeds the
        level at time 0. Every maximum equals the RP level, which is what the SFINCS forcing and
        correction B use.

        Reading: {len(off) / len(p):.1%} of all hydrographs and {(rp100.offset_hr != 0).mean():.1%}
        at RP100 peak elsewhere, most often at RP2 in semidiurnal tides. In 64% the maximum is
        one day away (+-24.8 h), in 19% one tidal cycle (+-12.4 h), in 8% within the hour around
        time 0: the window's centre high water is the median tide at the annual maxima, but a
        neighbouring high water (diurnal inequality, spring-neap cycle) is higher, and the slowly
        varying surge lifts it above the centre. Half of them exceed the time-0 level by less
        than {off.excess_m.median() * 100:.1f} cm, 99% by less than {off.excess_m.quantile(0.99) * 100:.0f} cm.
        The event stays physical: the storm's maximum lands on the highest high water within it,
        and the maximum is the RP level in every case.
        """)


def fig24_datum_terms(processed: Path, dest: Path) -> None:
    mdt = xr.open_dataset(processed / "MDT_mapped_on_coastal_points.nc")
    coast = xr.open_dataset(processed / "COAST-RP_preprocessed.nc")
    product = xr.open_dataset(processed / "GFM_storm_tide_levels_preprocessed.nc")
    t = pd.DataFrame({"lon": mdt.station_x_coordinate.values, "lat": mdt.station_y_coordinate.values,
                      "mdt": mdt.MDT.values, "offset": mdt.SLR_baseline_offset.values,
                      "level_change": product.storm_tide_rp_0100.values - coast.storm_tide_rp_0100.values})
    t = t[t.lat > PRODUCTION_MIN_LAT]
    t["change"] = t.level_change + t.offset
    unchanged = int((t.level_change.abs() < 1e-4).sum())
    panels = (("mdt", Normalize(-1.5, 1.5), "MDT CNES-CLS22, MSL above the GOCO06s geoid (m)",
               [-1.5, -0.75, 0, 0.75, 1.5]),
              ("offset", Normalize(-0.12, 0.12), "2020 offset, MSL rise 1995-2014 to 2020 (m)",
               [-0.12, -0.06, 0, 0.06, 0.12]),
              ("change", Normalize(-0.5, 0.5), "WL_scenarios RP100 at SLR 0, new - former (m)",
               [-0.5, -0.25, 0, 0.25, 0.5]))
    fig = plt.figure(figsize=(10, 19))
    for k, (col, norm, label, ticks) in enumerate(panels):
        v = t[col].dropna()
        v = v.iloc[np.argsort(np.abs(v.to_numpy()))]
        ax = fig.add_axes([0.02, 0.775 - k * 0.235, 0.96, 0.2], projection=MAP_PROJ)
        ax.set_global()
        _land(ax)
        sc = ax.scatter(t.loc[v.index, "lon"], t.loc[v.index, "lat"], c=v, s=3, cmap="RdBu_r", norm=norm,
                        linewidths=0, transform=PC, zorder=2)
        ax.spines["geo"].set_edgecolor(MUTED)
        draw_panel_letter(ax, "abc"[k])
        cax = fig.add_axes([0.25, 0.765 - k * 0.235, 0.5, 0.007])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
        cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.set_ticks(ticks, labels=[f"{x:g}" for x in ticks])
        cb.set_label(label)
    ax = fig.add_axes([0.1, 0.05, 0.85, 0.18])
    for col, color, name in (("offset", GREEN, "2020 offset"), ("level_change", BLUE, "product - COAST-RP"),
                             ("change", INK, "total change")):
        x = np.sort(t[col].dropna().to_numpy())
        ax.plot(x, np.arange(1, x.size + 1) / x.size, color=color, linewidth=2, label=name)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set(xlim=(-0.8, 0.8), xlabel="RP100 at SLR 0 (m)", ylabel="share of stations")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    draw_panel_letter(ax, "d")
    _save(fig, dest / "fig24_datum_terms.png")
    q = t[["mdt", "offset", "level_change", "change"]].quantile([0.05, 0.5, 0.95])
    describe(dest / "fig24_datum_terms.png", f"""
        The terms that turn a storm-tide level (above local MSL) into the total level of the
        WL_scenarios files, at the {len(t):,} stations north of 60 S: total = storm tide + MDT +
        2020 offset + SLR fingerprint (`preparation/prepare_boundary_conditions.py`). (a) The mean
        dynamic topography, local MSL above the geoid of the DEM (CNES-CLS22, 1993-2012 mean). (b)
        The 2020 offset: AR6's global-mean rise from 1995-2014 to 2020 (0.049 m), spread regionally
        with the SLR fingerprint, so that SLR 0 means the sea level of 2020 rather than of the MDT
        epoch. (c) The change of the RP100 level at SLR 0 between the former WL_scenarios (COAST-RP
        + MDT) and the current ones (storm-tide product + MDT + 2020 offset); the MDT is the same in
        both and cancels. (d) Distribution of the change and its two parts.

        Reading: the MDT spans {q.mdt[0.05]:.2f} to {q.mdt[0.95]:.2f} m (5-95%), highest in the
        Indo-Pacific warm pool and lowest in the North Atlantic, the Arctic and the Southern Ocean;
        it is the largest datum term. The offset is {q.offset[0.5] * 100:.1f} cm at the median
        station ({q.offset[0.05] * 100:.1f} to {q.offset[0.95] * 100:.1f} cm): near zero or negative
        where land still rises after the last glaciation (Hudson Bay, the Baltic, Alaska), above the
        global 4.9 cm on the US Gulf and Atlantic coasts and in South-east Asia. The change of the WL_scenarios level is dominated by the
        storm-tide product (median {q.level_change[0.5] * 100:+.1f} cm, 5-95% {q.level_change[0.05]:+.2f}
        to {q.level_change[0.95]:+.2f} m, its pattern as in fig12); the offset adds a few centimetres
        almost everywhere (median total {q.change[0.5] * 100:+.1f} cm). The step at zero in (d) is
        {unchanged:,} stations whose product level equals COAST-RP: the 15 GTSM fallbacks and
        tropical stations where COAST-RP's TC part alone sets RP100 and therefore sets the
        product's RP100 too.
        """)


def fig25_long_record(product: Path, dest: Path) -> None:
    d = pd.read_parquet(_guesses(product) / "long_record" / "long_record.parquet")
    tc = d.tc_part.to_numpy()
    fig = plt.figure(figsize=(17, 5.0))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    ax = fig.add_subplot(grid[0])
    for sel, color, label in ((~tc, MUTED, "no TC part"), (tc, ORANGE, "TC part")):
        ax.scatter(d.gpd_shape_mle[sel], d.long_gpd_shape_mle[sel], s=3, color=color, linewidths=0, alpha=0.5,
                   label=label)
    lim = (-0.8, 1.2)
    ax.plot(lim, lim, color=INK, linewidth=0.6)
    for v in (0.35,):
        ax.axvline(v, color=GRID, linewidth=0.8)
        ax.axhline(v, color=GRID, linewidth=0.8)
    ax.set(xlim=lim, ylim=lim, xlabel="GPD shape (MLE), 1979-2024", ylabel="GPD shape (MLE), 1950-2024")
    ax.legend(frameon=False, fontsize=8, markerscale=4)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = fig.add_subplot(grid[1])
    bins = np.linspace(-1.0, 1.0, 81)
    for sel, color, label in ((~tc, MUTED, "no TC part"), (tc, ORANGE, "TC part")):
        ax.hist((d.long_rp1000_m - d.rp1000_m)[sel].clip(-1, 1), bins=bins, color=color, alpha=0.6, label=label)
    ax.set(xlabel="RP1000 level, 1950-2024 minus 1979-2024 (m; clipped at +-1)", ylabel="stations")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = fig.add_subplot(grid[2], projection=MAP_PROJ)
    ratio = d.early_skew_rp10_m / d.late_skew_rp10_m
    cell = pd.DataFrame({"lon": (d.lon // 5) * 5 + 2.5, "lat": (d.lat // 5) * 5 + 2.5, "r": ratio}).dropna()
    cell = cell.groupby(["lon", "lat"]).r.median().reset_index()
    _land(ax)
    sc = ax.scatter(cell.lon, cell.lat, c=cell.r, s=9, cmap="RdBu_r", norm=Normalize(0.8, 1.2), transform=PC)
    ax.set_global()
    fig.colorbar(sc, ax=ax, shrink=0.8, label="skew-surge RP10, 1950-1978 / 1979-2024")
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig25_long_record.png")
    a, b = d.gpd_shape_mle > 0.35, d.long_gpd_shape_mle > 0.35
    describe(dest / "fig25_long_record.png", f"""
        Every product station rebuilt on GTSM-ERA5 1950-2024 with the production method of the
        1979-2024 product (no cyclone mask, the same gauge ratio r, shape cap 0.35); only the record
        length differs. (a) Maximum-likelihood GPD shape of the declustered skew surges, 1979-2024
        against 1950-2024 (grey lines: 0.35). Stations with a COAST-RP TC part: {int(a[tc].sum())}
        shapes above 0.35 in 46 years, {int(b[tc].sum())} in 75 years, {int((a & b)[tc].sum())} in both;
        without a TC part {int(a[~tc].sum())} and {int(b[~tc].sum())}. The heavy tails at cyclone
        stations stay, but which stations carry them changes: single storms set the shape.
        (b) Change of the product's RP1000 level. (c) Median per 5-degree cell of the ratio of the
        skew-surge RP10 fitted to 1950-1978 and to 1979-2024 (ERA5 before and after satellite
        assimilation; median {np.nanmedian(ratio):.3f}).
        """)


def fig26_pooled_shape(product: Path, dest: Path) -> None:
    g = _guesses(product) / "pooled_shape"
    st = pd.read_parquet(g / "stations.parquet")
    sc = pd.read_parquet(g / "scores.parquet")
    radii = [0, 100, 300, 1000]
    x = np.arange(len(radii))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    ax = axes[0]
    for sel, color, label in ((~st.tc_part, MUTED, "no TC part"), (st.tc_part, ORANGE, "TC part")):
        s = st[sel]
        dd = [np.abs(s[f"shape_{km}km_full"] - s[f"shape_{km}km_late"]) for km in radii]
        ax.plot(x, [v.median() for v in dd], "o-", color=color, label=f"{label}, median")
        ax.plot(x, [v.quantile(0.9) for v in dd], "o--", color=color, label=f"{label}, p90")
    ax.set(xlabel="pooling radius (km; 0 = station alone)", ylabel="|shape 1950-2024 - shape 1979-2024|")
    ax.set_xticks(x, [str(r) for r in radii])
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "a")
    ax = axes[1]
    for sel, color, label in ((~st.tc_part, MUTED, "no TC part"), (st.tc_part, ORANGE, "TC part")):
        s = st[sel]
        ax.plot(x, [(s[f"shape_{km}km_full"] > 0.35).sum() for km in radii], "o-", color=color, label=label)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set(xlabel="pooling radius (km; 0 = station alone)", ylabel="stations with shape > 0.35 (1950-2024)")
    ax.set_xticks(x, [str(r) for r in radii])
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = axes[2]
    levels = sc.level.unique().tolist()
    y = np.arange(len(levels))
    for k, (grp, color) in enumerate((("non-TC-dominated gauges", MUTED), ("TC-dominated gauges", ORANGE))):
        s = sc[(sc.gauges == grp) & (sc.rp == 100)].set_index("level").reindex(levels)
        ax.barh(y + (k - 0.5) * 0.38, s.mae, height=0.36, color=color, label=grp)
    ax.set_yticks(y, levels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="RP100 MAE against the gauges (m)", xlim=(0, 0.5))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style(ax)
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig26_pooled_shape.png")
    describe(dest / "fig26_pooled_shape.png", f"""
        GPD shape pooled over neighbouring stations, on 1950-2024 with the IBTrACS mask (cycles within
        300 km and 24 h of a tropical storm >= 34 kt, or anywhere on a track with no wind on record,
        leave the extratropical fit). A station's pooled shape maximises the summed profile
        likelihood of the stations within the radius; the scale stays per station. (a) How much a
        station's shape changes when 1950-1978 is added (stability against single storms).
        (b) Stations whose shape exceeds 0.35 ({len(st)} stations). (c) RP100 MAE at the ratio
        gauges (r left out) for the production product and the 1950-2024 variants: masking moves
        the TC-dominated gauges low (COAST-RP's TC part alone is too low), pooling and the cap do
        not change the scores.
        """)


def fig27_tide_correction(product: Path, dest: Path) -> None:
    g = _guesses(product) / "tide_correction"
    gt = pd.read_parquet(g / "gauges.parquet")
    sc = pd.read_parquet(g / "scores.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    lim = (0.02, 12.0)
    for ax, col, letter, label in ((axes[0], "amp_fes_m", "a", "FES at the gauge"),
                                   (axes[1], "amp_corrected_25km_m", "b", "FES at the gauge, corrected within 25 km")):
        far = gt.near_km > 10
        ax.scatter(gt.amp_gauge_m[~far], gt[col][~far], s=5, color=MUTED, linewidths=0, label="gauge <= 10 km from a station")
        ax.scatter(gt.amp_gauge_m[far], gt[col][far], s=7, color=BLUE, linewidths=0, label="gauge > 10 km from a station")
        ax.plot(lim, lim, color=INK, linewidth=0.6)
        for f in (0.75, 1.25):
            ax.plot(lim, np.multiply(lim, f), color=GRID, linewidth=0.8)
        ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="gauge tide amplitude (m)",
               ylabel=f"{label} (m)")
        ax.legend(frameon=False, fontsize=8, markerscale=2, loc="upper left")
        _style(ax)
        draw_panel_letter(ax, letter)
    ax = axes[2]
    km = [10, 25, 50, 100, 200]
    for grp, color in (("all", INK), ("gauge tide > 2 m", ORANGE), ("> 10 km from a station", BLUE)):
        s = sc[sc.gauges == grp].set_index("tide")
        ax.plot(km, [s.mae_m[f"FES at the gauge, corrected within {k} km"] for k in km], "o-", color=color, label=grp)
        ax.axhline(s.mae_m["station IDW (current forcing)"], color=color, linestyle="--", linewidth=0.9)
        ax.axhline(s.mae_m["FES at the gauge"], color=color, linestyle=":", linewidth=0.9)
    ax.set(xscale="log", xlabel="correction radius (km)", ylabel="tide amplitude MAE (m)")
    ax.set_xticks(km, [str(k) for k in km])
    ax.legend(frameon=False, fontsize=8, title="solid: corrected; dashed: current forcing;\ndotted: FES uncorrected",
              title_fontsize=7)
    _style(ax)
    draw_panel_letter(ax, "c")
    fig.tight_layout()
    _save(fig, dest / "fig27_tide_correction.png")
    a = sc[sc.gauges == "all"].set_index("tide").mae_m
    describe(dest / "fig27_tide_correction.png", f"""
        FES tide against {len(gt)} usable coastal GESLA sites (first two years of each record,
        least-squares M2, S2, N2, K1, O1 and M4; records that dry at low water, are sparse, or exceed
        a 20 m range (feet) are left out). Tide amplitude = the sum of the M2, S2, N2, K1 and O1
        amplitudes. (a) FES (the atlas of the nearest station) at the gauge against the gauge; grey
        lines at +-25%. (b) The same, scaled by the median gauge/FES ratio of the other gauges
        within 25 km (the gauge and its site left out). (c) MAE against the correction radius; dashed:
        the current forcing (IDW of the 5 nearest stations' FES tide), dotted: FES at the gauge
        uncorrected. All gauges: FES at the gauge {a['FES at the gauge']:.3f} m, current forcing
        {a['station IDW (current forcing)']:.3f} m, stations corrected
        {a['station IDW, stations corrected']:.3f} m, FES at the gauge corrected within 25 km
        {a['FES at the gauge, corrected within 25 km']:.3f} m.
        """)


def fig28_tile_interpolation(product: Path, dest: Path) -> None:
    t = pd.read_parquet(_guesses(product) / "tile_interpolation" / "gauges.parquet")
    variants = {"current": "IDW of station levels (current)", "current_tide_corrected": "same, station tides corrected",
                "decomposed": "IDW surge + local FES tide", "decomposed_tide_corrected": "same, local tide corrected"}
    groups = {"all gauges": np.ones(len(t), bool), "station > 5 km away": (t.station_km > 5).to_numpy(),
              "tide correction > 10%": ((t.tide_factor - 1).abs() > 0.1).to_numpy()}
    colors = (MUTED, INK, ORANGE, BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, rp, letter in ((axes[0], 2, "a"), (axes[1], 100, "b")):
        x = np.arange(len(groups))
        for k, (col, label) in enumerate(variants.items()):
            mae = [(t[f"{col}_rp{rp}_m"] - t[f"gauge_rp{rp}_m"])[sel].abs().mean() for sel in groups.values()]
            ax.bar(x + (k - 1.5) * 0.2, mae, width=0.19, color=colors[k], label=label)
        ax.set_xticks(x, [f"{g}\n(n = {int(s.sum())})" for g, s in groups.items()], fontsize=8)
        ax.set(ylabel=f"RP{rp} MAE against the gauges (m)")
        _style(ax)
        draw_panel_letter(ax, letter)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, dest / "fig28_tile_interpolation.png")
    describe(dest / "fig28_tile_interpolation.png", f"""
        Test 4, how the coastline level between stations is formed, at the {len(t)} ratio gauges.
        Current: inverse-distance weighting (k = 5, power 3) of the nearest stations' RP levels.
        Decomposed: the same weighting of the stations' skew-surge distributions (quantiles and GPD
        parameters averaged), convolved with the FES tide at the gauge. Each with the gauge tide
        correction (median gauge/FES amplitude ratio of the gauges within 25 km, the gauge's own site
        left out) on the stations' or the local tide. The stations' r and surge models are
        production's. The gauges lie {t.station_km.median():.1f} km (median) and at most
        {t.station_km.max():.0f} km from a station, so the estuary heads far from any station are
        not tested here (fig27 tests the tide alone there). (a) RP2, (b) RP100.
        """)


PROCESSED_FIGURES = {"fig24": fig24_datum_terms}
FIGURES = {"fig01": fig01_tide_scatter, "fig02": fig02_tide_ratio_maps, "fig03": fig03_constituents,
           "fig04": fig04_fes_versions, "fig05": fig05_basin_heads, "fig06": fig06_rule_sensitivity,
           "fig07": fig07_coast_rp_regions, "fig08": fig08_profiles_by_rp, "fig09": fig09_surge_by_rp,
           "fig10": fig10_generator_acceptance, "fig11": fig11_generator_profiles}
UNCERTAINTY_COMPONENTS = {"surge_signal": ("gauge ratio r (surge signal)", BLUE), "sampling_pooled": ("GPD sampling", GREEN),
                          "tide": ("tide (FES)", "#8e5bd0"), "tc": ("TC part", ORANGE), "record": ("record length", MUTED)}
RANGE_GROUPS = ["(0.0, 1.0]", "(1.0, 2.0]", "(2.0, 4.0]", "(4.0, 20.0]"]


def _range_bars(ax, df: pd.DataFrame, col: str, offset: float, color: str, label: str) -> None:
    for k, grp in enumerate(RANGE_GROUPS):
        e = df.loc[df.range_class == grp, col].dropna()
        if len(e):
            ax.plot([k + offset] * 2, e.quantile([0.1, 0.9]), color=color, linewidth=1.2)
            ax.plot(k + offset, e.median(), "o", color=color, markersize=5, label=label if k == 0 else None)


def fig29_level_uncertainty(product: Path, dest: Path) -> None:
    st = pd.read_parquet(_guesses(product) / "uncertainty" / "stations.parquet")
    fig = plt.figure(figsize=(16, 8.4))
    norm = LogNorm(0.01, 1.0)
    panels = (("combined_m", "combined (root sum of squares)"), ("surge_signal_m", "gauge ratio r"),
              ("sampling_pooled_m", "GPD sampling, pooled shape"), ("tide_m", "tide (FES vs gauges)"),
              ("tc_m", "TC part (where it dominates RP100)"))
    for n, (col, title) in enumerate(panels):
        ax = fig.add_subplot(2, 3, n + 1, projection=MAP_PROJ)
        _land(ax)
        sc = ax.scatter(st.lon, st.lat, c=st[col].clip(lower=0.01), s=1.5, cmap="YlOrRd", norm=norm, transform=PC,
                        linewidths=0)
        ax.set_global()
        ax.set_title(f"({'abcde'[n]}) {title}: median {st[col].median():.2f} m", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.14, wspace=0.04, hspace=0.12)
    cax = fig.add_axes([0.06, 0.07, 0.55, 0.018])
    fig.colorbar(sc, cax=cax, orientation="horizontal", label="RP100 level uncertainty (m)")
    ax = fig.add_subplot(2, 3, 6, projection=MAP_PROJ)
    _land(ax)
    cell = pd.DataFrame({"lon": (st.lon // 5) * 5 + 2.5, "lat": (st.lat // 5) * 5 + 2.5, "d": st.dominant})
    cell = cell.groupby(["lon", "lat"]).d.agg(lambda s: s.value_counts().index[0]).reset_index()
    for key, (label, color) in UNCERTAINTY_COMPONENTS.items():
        s = cell[cell.d == key]
        ax.scatter(s.lon, s.lat, s=14, marker="s", color=color, transform=PC, label=f"{label} ({len(s)} cells)")
    ax.set_global()
    ax.set_title("(f) largest component per 5-degree cell (most stations)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, -0.02), ncol=2)
    _save(fig, dest / "fig29_level_uncertainty.png")
    q = st[[f"{c}_m" for c in ("sampling_station", *UNCERTAINTY_COMPONENTS, "combined")]].quantile([0.5, 0.9]).round(3)
    share = st.dominant.value_counts(normalize=True).round(2)
    describe(dest / "fig29_level_uncertainty.png", f"""
        Uncertainty of the RP100 storm-tide level at {len(st)} product stations, by source (m; one
        standard error or a typical absolute error; input_analysis/boundary_uncertainty.py).
        (b) Gauge ratio r: the leave-one-out error of r at the ratio gauges around the station
        (median of the 5 nearest within 500 km; beyond, that of the gauges with fewer than 3
        neighbours), times the surge part of the level (RP100 ETC level - MHWS). (c) GPD sampling on
        1950-2024 with the IBTrACS mask and the 300 km pooled shape: delta method on the return
        level, the scale's variance given the shape plus the pooled shape's profile-likelihood
        standard error. (d) Tide: relative amplitude error of the tide gauges around the station
        (gauge-corrected where one lies within 25 km, FES otherwise) times MHWS. (e) TC part: the
        |RP100 error| of the TC-dominated gauges nearby (masked, basin shift left out), at stations
        where the TC part dominates RP100. (a) Root sum of squares of (b)-(e) and the record-length
        change (|RP100 1950-2024 - 1979-2024|). (f) The largest component per 5-degree cell.
        Medians / p90 (m): r {q.loc[0.5, 'surge_signal_m']} / {q.loc[0.9, 'surge_signal_m']},
        sampling pooled {q.loc[0.5, 'sampling_pooled_m']} / {q.loc[0.9, 'sampling_pooled_m']}
        (per-station shape {q.loc[0.5, 'sampling_station_m']} / {q.loc[0.9, 'sampling_station_m']}),
        tide {q.loc[0.5, 'tide_m']} / {q.loc[0.9, 'tide_m']}, TC {q.loc[0.5, 'tc_m']} / {q.loc[0.9, 'tc_m']},
        record {q.loc[0.5, 'record_m']} / {q.loc[0.9, 'record_m']}, combined {q.loc[0.5, 'combined_m']} /
        {q.loc[0.9, 'combined_m']}. Largest component (share of stations): r {share.get('surge_signal', 0)},
        TC {share.get('tc', 0)}, tide {share.get('tide', 0)}, sampling {share.get('sampling_pooled', 0)},
        record {share.get('record', 0)}.
        """)


def fig30_surge_accuracy(product: Path, dest: Path) -> None:
    g = pd.read_parquet(_guesses(product) / "uncertainty" / "gauges_surge.parquet")
    fig = plt.figure(figsize=(16, 8.4))
    ax = fig.add_subplot(2, 2, 1, projection=MAP_PROJ)
    _land(ax)
    sc = ax.scatter(g.lon, g.lat, c=g.ratio_fes, s=10, cmap="RdBu_r", norm=LogNorm(0.5, 2.0), transform=PC,
                    linewidths=0.2, edgecolors=INK)
    ax.set_global()
    fig.colorbar(sc, ax=ax, shrink=0.7, label="r = gauge / GTSM-on-FES skew surge, RP10")
    draw_panel_letter(ax, "a")
    ax = fig.add_subplot(2, 2, 2)
    classes = ["no_tc", "etc_with_tc", "tc_dominated"]
    for k, (col, color, label) in enumerate((("coast_rp", MUTED, "COAST-RP"), ("raw", ORANGE, "GTSM, r = 1, no TC"),
                                             ("product", BLUE, "product (r left out)"),
                                             ("decided", GREEN, "masked + basin shift (left out)"))):
        for j, cls in enumerate(classes):
            e = g.loc[g.station_class == cls, f"{col}_rp100_err_m"]
            x = j + (k - 1.5) * 0.18
            ax.plot([x, x], e.quantile([0.1, 0.9]), color=color, linewidth=1.4)
            ax.plot(x, e.median(), "o", color=color, markersize=5, label=label if j == 0 else None)
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_xticks(range(3), [f"{c}\n(n = {(g.station_class == c).sum()})" for c in classes])
    ax.set(ylabel="RP100 level - gauge (m): median, p10-p90")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "b")
    ax = fig.add_subplot(2, 2, 3)
    bins = np.linspace(-0.8, 0.8, 41)
    for sel, color, label in ((g.neighbours >= 3, BLUE, "3 or more gauges near"),
                              (g.neighbours < 3, ORANGE, "fewer than 3 (median fallback)")):
        ax.hist(g.r_loo_err[sel].clip(-0.8, 0.8), bins=bins, color=color, alpha=0.6,
                label=f"{label}: n {int(sel.sum())}, MAE {g.r_loo_err[sel].abs().mean():.2f}")
    ax.set(xlabel="r predicted without the gauge / gauge's r - 1", ylabel="gauges")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "c")
    ax = fig.add_subplot(2, 2, 4)
    size = g.groupby("region").size()
    keep = size[size >= 10].index
    order = g[g.region.isin(keep)].groupby("region").raw_rp100_err_m.median().sort_values().index
    y = np.arange(len(order))
    for col, color, label, off in (("raw_rp100_err_m", ORANGE, "GTSM, r = 1, no TC", -0.15),
                                   ("decided_rp100_err_m", GREEN, "decided product", 0.15)):
        s = g.groupby("region")[col].quantile([0.1, 0.5, 0.9]).unstack().loc[order]
        ax.hlines(y + off, s[0.1], s[0.9], color=color, linewidth=1.4)
        ax.plot(s[0.5], y + off, "o", color=color, markersize=5, label=label)
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_yticks(y, [f"{r} ({size[r]})" for r in order], fontsize=7)
    ax.set(xlabel="RP100 level - gauge (m): median, p10-p90")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    draw_panel_letter(ax, "d")
    fig.tight_layout()
    _save(fig, dest / "fig30_surge_accuracy.png")
    r = g.ratio_fes
    describe(dest / "fig30_surge_accuracy.png", f"""
        Accuracy of the GTSM-ERA5 surge signal at the {len(g)} ratio gauges (>= 20 complete years since
        1979, <= 10 km from a GTSM station). (a) r = the gauge's skew surge at RP10 / GTSM's skew surge
        on the FES tide at RP10: median {r.median():.2f}, p10-p90 {r.quantile(0.1):.2f}-{r.quantile(0.9):.2f};
        GTSM-ERA5 underestimates the surge at {(r > 1).mean():.0%} of the gauges. (b) RP100 level minus
        the gauge's (test 9 convolution of its own skew surges on its own tide), by gauge class:
        COAST-RP, GTSM alone (r = 1, no TC part), the product with r left out, and the decided
        product (IBTrACS mask, basin shift of COAST-RP_TC, left out per basin). (c) How well r is
        predicted where there is no gauge: r from the neighbouring gauges (gauge left out) against
        the gauge's own. (d) GTSM alone and the decided product by AR6 region (>= 10 gauges).
        """)


def fig31_duration_uncertainty(product: Path, dest: Path) -> None:
    u = _guesses(product) / "uncertainty"
    d = pd.read_parquet(u / "gauges_duration.parquet")
    st = pd.read_parquet(u / "stations.parquet")
    label = {"t025": "T(0.25)", "t05": "T(0.5)"}
    fig = plt.figure(figsize=(16, 8.4))
    sites = d[d.rp == 2].drop_duplicates("file_name")
    for n, f in enumerate(("t025", "t05")):
        ax = fig.add_subplot(2, 2, n + 1)
        _range_bars(ax, d[d.rp == 2], f"{f}_err_rel", -0.24, BLUE, "product RP2 / gauge events - 1")
        _range_bars(ax, d[d.rp == 100], f"{f}_err_rel", -0.08, ORANGE, "product RP100 / gauge events - 1")
        _range_bars(ax, sites.assign(v=sites[f"gtsm_{f}_median_hr"] / sites[f"gesla_{f}_median_hr"] - 1), "v",
                    0.08, GREEN, "GTSM / gauge annual maxima - 1")
        _range_bars(ax, sites, f"gesla_{f}_iqr_rel", 0.24, MUTED, "gauge event spread (IQR / median)")
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_xticks(range(len(RANGE_GROUPS)), [f"MHWS {g} m" for g in RANGE_GROUPS], fontsize=8)
        ax.set(ylabel=f"{label[f]}: relative (median, p10-p90)", ylim=(-1, 1.5))
        if n == 0:
            ax.legend(frameon=False, fontsize=7)
        _style(ax)
        draw_panel_letter(ax, "ab"[n])
    for n, f in enumerate(("t025", "t05")):
        ax = fig.add_subplot(2, 2, n + 3, projection=MAP_PROJ)
        _land(ax)
        ratio = st[f"rp100_{f}_hr"] / st[f"coast_hg_{f}_hr"]
        sc = ax.scatter(st.lon, st.lat, c=ratio, s=1.5, cmap="RdBu_r", norm=LogNorm(0.25, 4.0), transform=PC,
                        linewidths=0)
        ax.set_global()
        fig.colorbar(sc, ax=ax, shrink=0.7, label=f"product RP100 / COAST-HG, {label[f]}")
        draw_panel_letter(ax, "cd"[n])
    fig.tight_layout()
    _save(fig, dest / "fig31_duration_uncertainty.png")
    r2, r100 = d[d.rp == 2], d[d.rp == 100]
    describe(dest / "fig31_duration_uncertainty.png", f"""
        Uncertainty of the hydrograph durations T(0.25) and T(0.5): hours the level stays within 25% /
        50% of its peak (above MSL) of the peak. (a, b) By MHWS class at the GESLA gauges: the
        production generator (13-h smoothed GTSM surge shape on the FES tide) at RP2 ({r2.file_name.nunique()}
        sites) and RP100 ({r100.file_name.nunique()} sites) against the median of the gauge's own events
        in the same RP window; GTSM-ERA5's annual-maximum events against the gauge's; and the spread of
        the gauge's annual maxima (IQR / median), the natural event-to-event variability around any
        design shape. Generator RP2 median T(0.25) {r2.t025_err_rel.median():+.0%} (MAE
        {r2.t025_err_rel.abs().mean():.0%}), T(0.5) {r2.t05_err_rel.median():+.0%}; RP100 T(0.25)
        {r100.t025_err_rel.median():+.0%} (MAE {r100.t025_err_rel.abs().mean():.0%}; the gauge medians rest
        on few events). (c, d) Product RP100 against COAST-HG per station (log scale): the method choice
        alone.
        """)


PRODUCT_FIGURES = {"fig12": fig12_level_product_maps, "fig13": fig13_level_product_by_rp,
                   "fig14": fig14_level_product_gauges, "fig15": fig15_level_product_hydrographs,
                   "fig16": fig16_station_interpolation, "fig17": fig17_level_product_qa,
                   "fig18": fig18_product_durations,
                   "fig19": fig19_despike, "fig20": fig20_gpd_cap, "fig21": fig21_tc_mask, "fig22": fig22_fes_rule,
                   "fig23": fig23_peak_offsets, "fig25": fig25_long_record, "fig26": fig26_pooled_shape,
                   "fig27": fig27_tide_correction, "fig28": fig28_tile_interpolation,
                   "fig29": fig29_level_uncertainty, "fig30": fig30_surge_accuracy,
                   "fig31": fig31_duration_uncertainty}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO / ".cleanup_baseline" / "level_sources"))
    parser.add_argument("--dest", default=str(REPO / "input_analysis" / "boundary_forcing" / "figures"))
    parser.add_argument("--product-dir", default=str(REPO / ".cleanup_baseline" / "level_product"))
    parser.add_argument("--processed-dir", default=None,
                        help="processed_inputs with the boundary_conditions files (fig24); default: product-dir's parent")
    parser.add_argument("--only", nargs="*", choices=sorted({**FIGURES, **PRODUCT_FIGURES, **PROCESSED_FIGURES}),
                        default=None)
    args = parser.parse_args()
    out, dest = Path(args.out_dir), Path(args.dest)
    processed = Path(args.processed_dir) if args.processed_dir else Path(args.product_dir).parent
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.only or {**FIGURES, **PRODUCT_FIGURES, **PROCESSED_FIGURES}:
        if name in FIGURES:
            FIGURES[name](out, dest)
        elif name in PROCESSED_FIGURES:
            PROCESSED_FIGURES[name](processed, dest)
        else:
            PRODUCT_FIGURES[name](Path(args.product_dir), dest)
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
