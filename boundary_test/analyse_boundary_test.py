"""Plan §5.G: per-stratum scores of the boundary-test runs, the F9 exponent sweep and F2/F4.

Scores per variant, pooled over the tiles of a stratum (areas summed) and as the tile median:
  CSI = matched / (matched + model_only + sfincs_only), HR = matched / SFINCS wet,
  FAR = model_only / eikonal wet, bias = eikonal wet / SFINCS wet.
Only tiles with every variant present count, so the variants share one tile set per stratum.
Each table is also written without the tiles where some boundary cell rises by more than
SUSPECT_SHIFT_M under H1: there FES2022 puts MHWS up to 2x above gauges at the heads of macrotidal
basins (Minas Basin, Bahia Lomas, upper Gulf of California; IMPLEMENTATION_NOTES §2.19).

Writes <out-dir>/analysis/: scores_<vs_standin|vs_sfincs_h1>_<all|no_fes_suspect>.csv,
tiles_<vs_standin|vs_sfincs_h1>.csv, mhws_fes_vs_gtsm.csv, F2_mhws_fes_vs_gtsm.png,
F4_mhws_diff_map.png, F9_exponent_sweep_<subset>.png, H1_shift_map.png.

Usage:
    python boundary_test/analyse_boundary_test.py --out-dir .cleanup_baseline/boundary_test
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

STRATA = ("microtidal", "mesotidal", "macrotidal_open", "macrotidal_estuary", "macrotidal_other", "tc_dominated")
COLORS = dict(zip(STRATA, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
EARTH_RADIUS_KM = 6371.0
NODE_MATCH_KM = 20.0
AREAS = ("matched_km2", "model_only_km2", "sfincs_only_km2")
SUSPECT_SHIFT_M = 2.0


def tile_strata(out: Path) -> pd.Series:
    rep = pd.read_csv(out / "test_tiles_representative.csv", dtype={"tile_id": str})
    prov = pd.read_csv(out / "test_tiles_provisional.csv", dtype={"tile_id": str})
    return pd.concat([rep, prov[~prov.tile_id.isin(rep.tile_id)]]).set_index("tile_id").stratum


def fes_suspect_tiles(out: Path) -> set[str]:
    suspect = set()
    for path in (out / "boundary_product").glob("*.parquet"):
        cells = pd.read_parquet(path, columns=["l_rp100_m", "seed_prod_rp100_m"])
        if (cells.l_rp100_m - cells.seed_prod_rp100_m).max() > SUSPECT_SHIFT_M:
            suspect.add(path.stem)
    return suspect


def load_runs(path: Path, strata: pd.Series) -> pd.DataFrame:
    rows = pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])
    rows = rows[rows.variant.isin(["baseline", "H1", "H2", "H1H2"])].copy()
    rows["run"] = np.where(rows.variant.isin(["H2", "H1H2"]),
                           rows.variant + "_a" + rows.alpha.map("{:g}".format), rows.variant)
    complete = rows.groupby("tile_id").run.nunique()
    rows = rows[rows.tile_id.isin(complete[complete == complete.max()].index)]
    rows["stratum"] = rows.tile_id.map(strata)
    return rows


def metrics(m, mo, so) -> dict:
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"csi": m / (m + mo + so), "hr": m / (m + so), "far": mo / (m + mo), "bias": (m + mo) / (m + so)}


def stratum_scores(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.assign(**metrics(*(rows[c] for c in AREAS)))
    out = []
    for (stratum, run), g in rows.groupby(["stratum", "run"]):
        pooled = metrics(*(g[c].sum() for c in AREAS))
        out.append({"stratum": stratum, "run": run, "n_tiles": len(g), "sfincs_wet_km2": (g.matched_km2 + g.sfincs_only_km2).sum(),
                    **{f"{k}_pooled": v for k, v in pooled.items()},
                    **{f"{k}_tile_median": g[k].median() for k in ("csi", "bias")},
                    "mean_multiplier": g.mean_multiplier.mean(), "seed_shift_median_m": g.seed_shift_median_m.median()})
    return pd.DataFrame(out)


def tile_table(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.assign(**metrics(*(rows[c] for c in AREAS)), eik_wet_km2=rows.matched_km2 + rows.model_only_km2)
    wide = rows.pivot_table(index=["tile_id", "stratum"], columns="run", values=["csi", "bias", "eik_wet_km2"])
    wide.columns = [f"{v}_{r}" for v, r in wide.columns]
    shift = rows.groupby("tile_id").seed_shift_median_m.first()
    return wide.reset_index().assign(seed_shift_median_m=lambda d: d.tile_id.map(shift))


def mhws_comparison(out: Path, strata: pd.Series) -> pd.DataFrame:
    st = pd.read_csv(out / "test_station_tiles.csv", dtype={"tile_id": str}).drop_duplicates("gtsm_index")
    st["stratum"] = st.tile_id.map(strata)
    st = st.merge(pd.read_parquet(out / "tide_stats_gtsm.parquet")[["gtsm_index", "gtsm_mhws_m"]], on="gtsm_index")
    fes = pd.read_parquet(out / "tide_stats_fes2022.parquet")
    fes = fes[fes.valid].drop_duplicates(["node_lon", "node_lat"])
    lon = ((fes.node_lon + 180) % 360) - 180
    d, i = BallTree(np.radians(np.column_stack([fes.node_lat, lon])), metric="haversine").query(
        np.radians(st[["lat", "lon"]].to_numpy()), k=1)
    st["fes_mhws_m"] = np.where(d[:, 0] * EARTH_RADIUS_KM <= NODE_MATCH_KM, fes.mhws_m.to_numpy()[i[:, 0]], np.nan)
    st["node_dist_km"] = d[:, 0] * EARTH_RADIUS_KM
    return st.dropna(subset=["gtsm_mhws_m", "fes_mhws_m"]).reset_index(drop=True)


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_f2(st: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    for s in STRATA:
        g = st[st.stratum == s]
        ax.scatter(g.gtsm_mhws_m, g.fes_mhws_m, s=10, color=COLORS[s], alpha=0.6, linewidths=0, label=f"{s} ({len(g)})")
    lim = [0, float(np.nanmax([st.gtsm_mhws_m.max(), st.fes_mhws_m.max()])) * 1.05]
    ax.plot(lim, lim, color=MUTED, linewidth=1)
    ax.set(xlim=lim, ylim=lim, xlabel="MHWS GTSM tide (m, local MSL)", ylabel="MHWS FES2022, nearest node (m)",
           title="F2: MHWS at test-tile stations")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f4(st: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    diff = st.fes_mhws_m - st.gtsm_mhws_m
    sc = ax.scatter(st.lon, st.lat, c=diff, s=6, cmap="RdBu", vmin=-0.5, vmax=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="MHWS FES - GTSM (m)", shrink=0.8)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="lon", ylabel="lat", title="F4: MHWS FES2022 - GTSM at test-tile stations")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shift_map(tiles: pd.DataFrame, centres: pd.DataFrame, path: Path) -> None:
    t = tiles.merge(centres, on="tile_id")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(t.lon, t.lat, c=t.seed_shift_median_m, s=18, cmap="RdBu", vmin=-1, vmax=1, linewidths=0)
    fig.colorbar(sc, ax=ax, label="median L_RP100 - production seed level (m)", shrink=0.8)
    ax.set(xlim=(-180, 180), ylim=(-60, 80), xlabel="lon", ylabel="lat",
           title=f"H1 boundary shift per tile ({len(t)} tiles)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_f9(scores: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for col, family in enumerate(("H2", "H1H2")):
        ref = "baseline" if family == "H2" else "H1"
        for row, metric in enumerate(("csi_pooled", "bias_pooled")):
            ax = axes[row, col]
            for s in STRATA:
                g = scores[(scores.stratum == s) & scores.run.str.startswith(family + "_a")]
                if g.empty:
                    continue
                alpha = g.run.str.split("_a").str[1].astype(float)
                order = np.argsort(alpha.to_numpy())
                ax.plot(alpha.to_numpy()[order], g[metric].to_numpy()[order], color=COLORS[s], linewidth=2,
                        marker="o", markersize=5, label=f"{s} (n={int(g.n_tiles.iloc[0])})")
                base = scores[(scores.stratum == s) & (scores.run == ref)][metric]
                if len(base):
                    ax.axhline(float(base.iloc[0]), color=COLORS[s], linewidth=1, linestyle=":")
            if metric == "bias_pooled":
                ax.axhline(1.0, color=MUTED, linewidth=1)
            ax.set(title=f"{family}: {metric.split('_')[0].upper()} vs exponent (dotted: {ref}, no H2)",
                   xlabel="exponent alpha" if row else None, ylabel=metric.split("_")[0].upper())
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("F9 vs existing SFINCS runs (stand-in, old boundary levels)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    dest.mkdir(exist_ok=True)
    strata = tile_strata(out)
    suspect = fes_suspect_tiles(out)
    print(f"{len(suspect)} tiles with an H1 shift > {SUSPECT_SHIFT_M} m: {sorted(suspect, key=int)}")
    rep = pd.read_csv(out / "test_tiles_representative.csv", dtype={"tile_id": str})
    centres = pd.concat([rep, pd.read_csv(out / "test_tiles_provisional.csv", dtype={"tile_id": str})]) \
        .drop_duplicates("tile_id")[["tile_id", "lon", "lat"]]

    for name, file in (("vs_standin", "results.jsonl"), ("vs_sfincs_h1", "results_vs_sfincs_h1.jsonl")):
        if not (out / "runs" / file).exists():
            continue
        all_rows = load_runs(out / "runs" / file, strata)
        tiles = tile_table(all_rows).assign(fes_suspect=lambda d: d.tile_id.isin(suspect))
        tiles.to_csv(dest / f"tiles_{name}.csv", index=False)
        for subset, rows in (("all", all_rows), ("no_fes_suspect", all_rows[~all_rows.tile_id.isin(suspect)])):
            scores = stratum_scores(rows).assign(subset=subset)
            scores.to_csv(dest / f"scores_{name}_{subset}.csv", index=False)
            print(f"\n{name}, {subset}: {rows.tile_id.nunique()} tiles")
            print(scores.pivot_table(index="stratum", columns="run", values="csi_pooled").round(3).to_string())
            print(scores.pivot_table(index="stratum", columns="run", values="bias_pooled").round(2).to_string())
            if name == "vs_standin":
                plot_f9(scores, dest / f"F9_exponent_sweep_{subset}.png")
        if name == "vs_standin":
            plot_shift_map(tiles, centres, dest / "H1_shift_map.png")

    st = mhws_comparison(out, strata)
    st.to_csv(dest / "mhws_fes_vs_gtsm.csv", index=False)
    plot_f2(st, dest / "F2_mhws_fes_vs_gtsm.png")
    plot_f4(st, dest / "F4_mhws_diff_map.png")
    d = st.fes_mhws_m - st.gtsm_mhws_m
    print(f"\nMHWS FES - GTSM over {len(st)} stations: median {d.median():+.3f} m, MAE {d.abs().mean():.3f} m")
    print(st.assign(d=d).groupby("stratum").d.describe()[["count", "50%", "25%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
