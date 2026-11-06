"""Draw the figures of the boundary-forcing input analysis, each with a .txt description.

Reads the tables written by the `boundary_test/` scripts (INPUT_ANALYSIS.md §10) from
<out-dir>/ and <out-dir>/analysis/, and writes figNN_*.png + figNN_*.txt into --dest. Figures
carry no titles; everything a reader needs to decode them is in the description file.

Usage:
    python input_analysis/make_boundary_forcing_figures.py [--only fig01 fig07]
        [--out-dir .cleanup_baseline/level_sources] [--dest input_analysis/boundary_forcing/figures]
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
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize, TwoSlopeNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from map_style import LAND_COLOR, draw_caption_box, draw_panel_letter  # noqa: E402
from atlas_rule_sensitivity import VERSION_DIFF_M, VERSION_RATIO, version_flag  # noqa: E402
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
            .str.replace(".", ". ", regex=False).str.replace("-", " ") + " (" + reg.region + ")")
    counts = name.value_counts()
    small = set(counts.index[counts < MIN_REGION_SITES])
    d["label"] = np.where(name.isin(small), f"other ({len(small)} regions)", name)
    d["continent"] = np.where(name.isin(small), "OTHER", reg.continent)

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
    y, labels, ypos, heads = [], [], 0.0, []
    for k, (label, cont) in enumerate(zip(st.index, st.continent)):
        if k > 0 and cont != st.continent.iloc[k - 1]:
            ypos += 0.6
            heads.append((ypos - 0.55, GROUP_LABELS.get(cont, cont.title().replace("-", " "))))
        y.append(ypos)
        labels.append(label)
        ypos += 1
    y = np.array(y)
    lo = float(st[["p25_10", "p25_100"]].min().min())
    hi = float(st[["p75_10", "p75_100"]].max().max())
    pad = 0.06 * (hi - lo)
    fig = plt.figure(figsize=(10.5, 0.25 * ypos + 1.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1, 0.3), wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    for rp, dy in ((10, -0.17), (100, 0.17)):
        ax.hlines(y + dy, st[f"p25_{rp}"], st[f"p75_{rp}"], color=RP_COLORS[rp], linewidth=2, alpha=0.5)
        ax.scatter(st[f"med_{rp}"], y + dy, color=RP_COLORS[rp], s=20, zorder=3, label=f"RP{rp}")
    ax.axvline(0, color=MUTED, linewidth=1)
    for yh, text in heads:
        ax.axhline(yh - 0.05, color=GRID, linewidth=0.8)
        if text:
            ax.text(lo - pad, yh + 0.02, text, fontsize=7.5, color=MUTED, ha="left", va="bottom", style="italic")
    ax.set_yticks(y, labels, fontsize=8)
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
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.65 / (0.25 * ypos + 1.4), top=1 - 0.3 / (0.25 * ypos + 1.4))
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
        Southern), alphabetically within each group; the italic labels name the groups. Each gauge
        takes the nearest AR6 land region within 200 km, otherwise the AR6 ocean region containing
        it. Regions with fewer than {MIN_REGION_SITES} sites are pooled in the last row. The
        numbers are in fig07_coast_rp_vs_gauges_data.csv.

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


def fig05_basin_heads(out: Path, dest: Path) -> None:
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd
    from shapely.geometry import box

    n = _nodes(out)
    g = pd.read_csv(out / "analysis" / "atlas_gauges.csv", index_col=0)
    land = gpd.read_file(shpreader.natural_earth("50m", "physical", "land"))
    basins = []
    for name, (x0, x1, y0, y1) in FIG05_BASINS.items():
        b = n[n.node_lon.between(x0, x1) & n.node_lat.between(y0, y1)]
        gb = g[g.lon.between(x0, x1) & g.lat.between(y0, y1)]
        lons = np.concatenate([b.node_lon, gb.lon])
        lats = np.concatenate([b.node_lat, gb.lat])
        px = max(BASIN_PAD * np.ptp(lons), 0.05)
        py = max(BASIN_PAD * np.ptp(lats), 0.05)
        ext = (lons.min() - px, lons.max() + px, lats.min() - py, lats.max() + py)
        stretch = 1 / np.cos(np.radians(np.mean(ext[2:])))
        aspect = (ext[1] - ext[0]) / ((ext[3] - ext[2]) * stretch)
        basins.append((name, b, gb, ext, stretch, aspect))
    fig_w, left, gap, top, mid, cbar_h = 17.0, 0.25, 0.35, 0.35, 0.32, 0.9
    map_h = (fig_w - 2 * left - gap * (len(basins) - 1)) / sum(b[5] for b in basins)
    fig_h = top + 2 * map_h + mid + cbar_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    x = left
    summary = []
    for j, (name, b, gb, ext, stretch, aspect) in enumerate(basins):
        w = aspect * map_h
        vmax = float(np.nanmax(b[["fes2022_mhws_m", "fes2014_mhws_m"]].to_numpy()))
        norm = Normalize(0, vmax)
        clip = land.clip(box(ext[0], ext[2], ext[1], ext[3]))
        frame = plt.Rectangle(((x - 0.12) / fig_w, (cbar_h - 0.62) / fig_h), (w + 0.24) / fig_w,
                              (2 * map_h + mid + 0.62 + 0.28) / fig_h, transform=fig.transFigure,
                              fill=False, edgecolor=GRID, linewidth=1.2)
        fig.add_artist(frame)
        sc = None
        for i, col in enumerate(("fes2022_mhws_m", "fes2014_mhws_m")):
            y0 = cbar_h + (1 - i) * (map_h + mid)
            ax = fig.add_axes([x / fig_w, y0 / fig_h, w / fig_w, map_h / fig_h])
            clip.plot(ax=ax, color=LAND_COLOR, edgecolor="none", zorder=0)
            keep, rep = b[~b.replaced], b[b.replaced]
            sc = ax.scatter(keep.node_lon, keep.node_lat, c=keep[col], s=14, cmap="viridis", norm=norm,
                            linewidths=0, zorder=2)
            ax.scatter(rep.node_lon, rep.node_lat, c=rep[col], s=28, marker="X", cmap="viridis", norm=norm,
                       linewidths=0.3, edgecolors=INK, zorder=3)
            ax.scatter(gb.lon, gb.lat, c=gb.gesla_harm__mhws_m, s=70, marker="^", cmap="viridis", norm=norm,
                       edgecolors=INK, linewidths=0.8, zorder=4)
            ax.set(xlim=ext[:2], ylim=ext[2:], xticks=[], yticks=[])
            ax.set_aspect(stretch, adjustable="datalim")
            for side in ax.spines.values():
                side.set_edgecolor(MUTED)
            draw_panel_letter(ax, "abcdefgh"[2 * j + i])
        cax = fig.add_axes([(x + 0.1 * w) / fig_w, (cbar_h - 0.42) / fig_h, 0.8 * w / fig_w, 0.12 / fig_h])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_label("MHWS (m)", fontsize=8)
        cb.ax.tick_params(labelsize=7.5)
        summary.append(f"({'aceg'[j]}, {'bdfh'[j]}) {name}: {len(b)} nodes, {int(b.replaced.sum())} replaced; "
                       f"FES2022 MHWS max {b.fes2022_mhws_m.max():.1f} m, FES2014 max {b.fes2014_mhws_m.max():.1f} m"
                       + ("; gauges: " + ", ".join(f"{r.site_name} {r.gesla_harm__mhws_m:.2f} m"
                                                   for r in gb.itertuples()) if len(gb) else "; no gauge"))
        x += w + gap
    _save(fig, dest / "fig05_fes_basin_heads.png")
    describe(dest / "fig05_fes_basin_heads.png", f"""
        MHWS (mean high water springs above local mean sea level) at the FES boundary nodes in
        the four basins where FES2022 has gross artefacts. Each grey frame is one basin and holds
        two maps of the same area: FES2022 on top, FES2014 below, at the same nodes. Circles: nodes
        that keep FES2022 under the version rule; crosses: nodes where the rule replaces FES2022
        by FES2014 (fig04, fig06). Triangles: GESLA tide gauges, coloured by their observed MHWS.
        Both maps of a frame share the colour scale below them. Each map is cropped to its nodes
        and gauges and drawn with the east-west scale corrected for latitude, so shapes are true.
        Land: Natural Earth 1:50m, grey.

        {chr(10).join(summary)}

        Observed references without a gauge in the frame: Minas Basin mean spring range 14.5 m,
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
            level = float(norm(m[i, j]))
            if isinstance(norm, BoundaryNorm):
                level /= max(norm.Ncmap - 1, 1)
            dark = abs(level - 0.5) > 0.3 if isinstance(norm, TwoSlopeNorm) else level > 0.6
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
    grid["net"] = grid.gauges_fixed - grid.gauges_harmed
    best = int(grid.loc[grid.gauges_harmed == 0, "gauges_fixed"].max())
    known = int(grid.known_total.iloc[0])
    grid["known_share"] = grid.known_caught / known
    lim = float(grid.net.abs().max())
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    im = _heat(axes[0], grid, "net", lambda r: f"{int(r.gauges_fixed)}/{int(r.gauges_harmed)}", plt.get_cmap("RdBu"),
               TwoSlopeNorm(0, -lim, lim), mark=lambda r: r.gauges_harmed == 0 and r.gauges_fixed == best)
    cb = fig.colorbar(im, ax=axes[0], orientation="horizontal", pad=0.17, shrink=0.6, aspect=30)
    cb.set_label("gauges improved minus gauges made worse")
    _heat(axes[1], grid, "known_share", lambda r: f"{int(r.known_caught)}", plt.get_cmap("Blues"), Normalize(0, 1),
          mark=lambda r: r.known_caught == known)
    for ax, letter in zip(axes, "ab"):
        draw_panel_letter(ax, letter)
    fig.tight_layout()
    _save(fig, dest / "fig06_version_rule_sensitivity.png")
    adopted = grid[(grid.ratio == VERSION_RATIO) & (grid.diff_m == VERSION_DIFF_M)].iloc[0]
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
        improvements / worsenings. Colour: improvements minus worsenings (blue = net better, red =
        net worse, white = no effect). The gauges with a real FES2022 error (Skagway, Juneau, Port
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

        Reading off the adopted values: (a) leaves the dotted pairs in the column D = 1.25 m (no
        worsening, all {best} improvements; at D <= 1 m Trondheim is switched wrongly, at D = 1.5 m
        Port Alberni is no longer caught). (b) leaves R <= 1.5 (higher R misses artefacts). Of the
        pairs left, R = 1.5 is the largest, i.e. the one that changes the fewest other nodes
        (figS1). At the adopted pair: {int(adopted.gauges_fixed)} improvements,
        {int(adopted.gauges_harmed)} worsenings, {int(adopted.known_caught)} of {known} artefacts
        replaced.

        Behind the D bound: comparing the two atlases at all 905 gauges, binned by how much
        FES2022 exceeds FES2014 in MHWS (gain = |FES2022 - GESLA| - |FES2014 - GESLA|, positive
        when FES2014 is better): {band_lines}. Below about 1 m FES2014 is not the better atlas, so
        switching there does harm. The gauge evidence rests on few sites (5 with a difference above
        1 m), so the known artefacts carry the ratio bound.
        """)


def figS1_nodes_replaced(out: Path, dest: Path) -> None:
    grid = pd.read_csv(out / "analysis" / "atlas_rule_grid.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    _heat(ax, grid, "nodes_replaced", lambda r: f"{int(r.nodes_replaced)}", plt.get_cmap("Oranges"),
          LogNorm(max(grid.nodes_replaced.min(), 1), grid.nodes_replaced.max()))
    fig.tight_layout()
    _save(fig, dest / "figS1_nodes_replaced.png")
    total = len(_nodes(out))
    describe(dest / "figS1_nodes_replaced.png", f"""
        Internal: number of the {total:,} FES boundary nodes of the test tiles that the version rule
        switches from FES2022 to FES2014, for each pair of thresholds (R vertical, D horizontal;
        colour on a logarithmic scale, darker = more nodes). The red frame marks the adopted pair
        R = {VERSION_RATIO}, D = {VERSION_DIFF_M} m. The rule and the two thresholds are described
        with fig06.
        """)


FIGURES = {"fig01": fig01_tide_scatter, "fig02": fig02_tide_ratio_maps, "fig03": fig03_constituents,
           "fig04": fig04_fes_versions, "fig05": fig05_basin_heads, "fig06": fig06_rule_sensitivity,
           "fig07": fig07_coast_rp_regions, "fig08": fig08_profiles_by_rp, "fig09": fig09_surge_by_rp,
           "figS1": figS1_nodes_replaced}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(REPO / ".cleanup_baseline" / "level_sources"))
    parser.add_argument("--dest", default=str(REPO / "input_analysis" / "boundary_forcing" / "figures"))
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = parser.parse_args()
    out, dest = Path(args.out_dir), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in args.only or FIGURES:
        FIGURES[name](out, dest)
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
