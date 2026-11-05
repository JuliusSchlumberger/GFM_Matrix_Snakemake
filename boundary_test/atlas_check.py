"""Test 6: EOT20 as a second tide atlas, at the gauges and at the FES2022 boundary nodes.

Gauges (all 1,154; scored on the tide set): EOT20 is extracted at the gauge with
`fes_tide_stats.extract_constants` (bilinear, since EOT20 has land holes; extrapolation to the
nearest valid cell within CUTOFF_KM), predicted over the same 18.61-y 10-min epoch with its own 17 constituents (Sa / Ssa
dropped, as the `no_seasonal` variant) and reduced with `tide_statistics` to MHWS. Compared with
GESLA and FES2022 (`levels_sites.csv`), and per constituent (M2, S2, K1, O1) with the harmonic
constants of `levels_tides.py`.

FES nodes (the test tiles' boundary nodes, `fes_nodes.parquet`): a full nodal-cycle prediction at
~90,000 nodes is too slow, so both atlases are screened with the same amplitude proxy
A = |M2| + |S2| + |K1| + |O1| (about MHWS on semidiurnal and mixed coasts). A node is flagged when
A_FES > FLAG_RATIO x the largest A_EOT20 within NEIGHBOUR_KM, and unverifiable when EOT20 has no
value within NEIGHBOUR_KM.

Writes <out-dir>/analysis/{atlas_gauges.csv, atlas_nodes.parquet, atlas_flagged_nodes.csv} and
atlas_gauges.png, atlas_nodes_map.png, atlas_basins.png.

Usage:
    python boundary_test/atlas_check.py --out-dir .cleanup_baseline/level_sources [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyTMD.io  # noqa: E402
from sklearn.neighbors import BallTree  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "boundary_test"))
from fes_tide_stats import FES_MODEL, STEP_MIN, ensure_local_fes, extract_constants, prediction_basis  # noqa: E402
from levels_tides import FES_CACHE, GROUP_DEG, levels  # noqa: E402
from site_sets import GRID, INK, MUTED, _save, _style  # noqa: E402

EOT20_ROOT = Path("P:/11212688-004-global-floodmaps/modelling/inputs")
EOT20_CACHE = REPO / ".cleanup_baseline" / "eot20_cache"
FES_NODES = REPO / ".cleanup_baseline" / "boundary_test" / "fes_nodes.parquet"
PROXY = ("m2", "s2", "k1", "o1")
SEASONAL = ("sa", "ssa")
FLAG_RATIO = 1.5
NEIGHBOUR_KM = 25.0
EARTH_RADIUS_KM = 6371.0
POINTS_PER_TASK = 25
BASINS = {"Minas Basin / Cobequid Bay": (-65.2, -63.2, 44.9, 45.6),
          "Strait of Magellan / Bahia Lomas": (-71.5, -68.0, -53.6, -52.0),
          "Upper Gulf of California": (-115.3, -113.0, 30.3, 32.2)}
COLORS = {"fes": "#2a78d6", "eot20": "#eb6834", "gtsm": "#898781"}


def grouped_constants(model, lonlat: np.ndarray, method: str = "spline") -> np.ndarray:
    cell = np.floor(lonlat / GROUP_DEG).astype(int)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    groups = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
    hc = extract_constants(model, [lonlat[g] for g in groups], method)
    out = np.empty_like(hc)
    out[np.concatenate(groups)] = hc
    return out


def subset_model(model, names: tuple[str, ...]):
    keep = [k for k, c in enumerate(model.constituents) if c in names]
    model.model_file = [model.model_file[k] for k in keep]
    model.constituents = [model.constituents[k] for k in keep]
    return model


def eot20_basis(model) -> tuple[np.ndarray, np.ndarray]:
    EOT20_CACHE.mkdir(parents=True, exist_ok=True)
    re_path, im_path = EOT20_CACHE / "basis_re.npy", EOT20_CACHE / "basis_im.npy"
    if not re_path.exists():
        b_re, b_im = prediction_basis(list(model.constituents), model)
        np.save(re_path, b_re.astype(np.float32))
        np.save(im_path, b_im.astype(np.float32))
    return re_path, im_path


def _levels_task(args) -> list[dict]:
    re_path, im_path, hc = args
    b_re = np.load(re_path, mmap_mode="r")
    b_im = np.load(im_path, mmap_mode="r")
    ok = np.isfinite(hc).all(axis=1)
    tide = np.asarray(b_re) @ np.nan_to_num(hc.real).T + np.asarray(b_im) @ np.nan_to_num(hc.imag).T
    return [levels(tide[:, j], STEP_MIN * 60.0) if ok[j] else {} for j in range(len(hc))]


def gauge_table(out: Path, model, workers: int) -> pd.DataFrame:
    sites = pd.read_csv(out / "analysis" / "levels_sites.csv", index_col=0)
    pairs = pd.read_parquet(out / "pairs.parquet").set_index("file_name").reindex(sites.index)
    lonlat = pairs[["lon", "lat"]].to_numpy(dtype=float)
    hc = grouped_constants(model, lonlat, "bilinear")
    cons = list(model.constituents)
    hc[:, [cons.index(c) for c in SEASONAL]] = 0.0
    re_path, im_path = eot20_basis(model)
    tasks = [(re_path, im_path, hc[i:i + POINTS_PER_TASK]) for i in range(0, len(hc), POINTS_PER_TASK)]
    with ProcessPoolExecutor(workers) as pool:
        stats = [r for chunk in pool.map(_levels_task, tasks) for r in chunk]
    t = sites[["site_name", "country", "region", "lon", "lat", "quality", "gesla_harm__mhws_m", "gtsm_harm__mhws_m",
               "fes_gauge__mhws_m", "gesla_harm__spring_range_m"]].copy()
    t["eot20__mhws_m"] = [r.get("mhws_m", np.nan) for r in stats]
    t["eot20__spring_range_m"] = [r.get("spring_range_m", np.nan) for r in stats]
    saved = np.load(out / "harmonic_constants.npz", allow_pickle=True)  # own output; file_name is an object array
    pos = pd.Series(range(len(saved["file_name"])), index=saved["file_name"]).reindex(sites.index).to_numpy()
    fes_cons = list(saved["constituents"])
    for c in PROXY:
        t[f"gesla__{c}_m"] = np.abs(saved["gesla_harm"][pos, fes_cons.index(c)])
        t[f"fes__{c}_m"] = np.abs(saved["fes_gauge"][pos, fes_cons.index(c)])
        t[f"eot20__{c}_m"] = np.abs(hc[:, cons.index(c)])
    return t


def node_table(model_eot, model_fes) -> pd.DataFrame:
    nodes = pd.read_parquet(FES_NODES)
    nodes = nodes[nodes.valid & nodes.mhws_m.notna()].reset_index(drop=True)
    lonlat = nodes[["node_lon", "node_lat"]].to_numpy(dtype=float)
    out = nodes[["node_lon", "node_lat", "mhws_m"]].rename(columns={"mhws_m": "fes_mhws_m"})
    for name, model in (("fes", model_fes), ("eot20", model_eot)):
        hc = grouped_constants(model, lonlat, "bilinear" if name == "eot20" else "spline")
        out[f"{name}_proxy_m"] = np.abs(hc).sum(axis=1)
        out.loc[~np.isfinite(hc).all(axis=1), f"{name}_proxy_m"] = np.nan
    ok = out.eot20_proxy_m.notna().to_numpy()
    tree = BallTree(np.radians(out.loc[ok, ["node_lat", "node_lon"]].to_numpy()), metric="haversine")
    neighbours = tree.query_radius(np.radians(out[["node_lat", "node_lon"]].to_numpy()), r=NEIGHBOUR_KM / EARTH_RADIUS_KM)
    eot_ok = out.eot20_proxy_m.to_numpy()[ok]
    out["eot20_max_25km_m"] = [eot_ok[n].max() if len(n) else np.nan for n in neighbours]
    out["ratio_fes_eot20"] = out.fes_proxy_m / out.eot20_max_25km_m
    out["status"] = np.where(out.eot20_max_25km_m.isna(), "unverifiable",
                             np.where(out.ratio_fes_eot20 > FLAG_RATIO, "flagged", "ok"))
    return out


def plot_gauges(t: pd.DataFrame, dest: Path) -> None:
    q = t[t.quality]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (col, label) in zip(axes[:2], (("fes_gauge__mhws_m", "FES2022"), ("eot20__mhws_m", "EOT20"))):
        ax.scatter(q.gesla_harm__mhws_m, q[col], s=10, color=COLORS["fes" if label == "FES2022" else "eot20"],
                   alpha=0.7, linewidths=0)
        hi = float(np.nanmax(q.gesla_harm__mhws_m)) * 1.25
        ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1)
        d = (q[col] - q.gesla_harm__mhws_m).dropna()
        ratio = (q[col] / q.gesla_harm__mhws_m).median()
        off = int((q[col] > hi).sum())
        ax.set(xlim=(0, hi), ylim=(0, hi), xlabel="GESLA (gauge) MHWS (m)", ylabel=f"{label} at the gauge MHWS (m)",
               title=f"{label} vs gauge: median ratio {ratio:.2f}, MAE {d.abs().mean():.3f} m, "
                     f"within 0.1 m {(d.abs() <= 0.1).mean():.0%} (n={len(d)})"
                     + (f"\n{off} off scale (EOT20 S2 artefacts, St Lawrence)" if off else ""))
        ax.title.set_fontsize(9)
        _style(ax)
    ax = axes[2]
    data, labels, colors = [], [], []
    for c in PROXY:
        for src in ("fes", "eot20"):
            r = (q[f"{src}__{c}_m"] / q[f"gesla__{c}_m"]).replace([np.inf, -np.inf], np.nan).dropna()
            data.append(r[q.loc[r.index, f"gesla__{c}_m"] > 0.05])
            labels.append(f"{c.upper()}\n{'FES' if src == 'fes' else 'EOT20'}")
            colors.append(COLORS[src])
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, alpha=0.5, edgecolor=color)
    for median in bp["medians"]:
        median.set(color=INK)
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.set(ylim=(0.6, 1.4), ylabel="amplitude / gauge amplitude",
           title="Constituent amplitude ratio at the gauges (gauge amplitude > 5 cm)")
    ax.title.set_fontsize(9)
    _style(ax)
    fig.suptitle("Tide set (quality sites): FES2022 and EOT20 against the gauges, same prediction and statistics",
                 color=INK)
    fig.tight_layout()
    _save(fig, dest / "atlas_gauges.png")


def plot_nodes_map(nodes: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ok = nodes[nodes.status == "ok"]
    ax.scatter(ok.node_lon, ok.node_lat, s=2, color=GRID, linewidths=0, label=f"ok (n={len(ok)})")
    for status, color, size in (("unverifiable", MUTED, 8), ("flagged", "#e34948", 18)):
        s = nodes[nodes.status == status]
        ax.scatter(s.node_lon, s.node_lat, s=size, color=color, linewidths=0, label=f"{status} (n={len(s)})")
    for name, (x0, x1, y0, y1) in BASINS.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=INK, linewidth=0.8))
    ax.set(xlim=(-180, 180), ylim=(-60, 85), xlabel="longitude", ylabel="latitude",
           title=f"FES2022 boundary nodes of the test tiles: flagged = FES proxy > {FLAG_RATIO} x the largest EOT20 "
                 f"proxy within {NEIGHBOUR_KM:.0f} km")
    ax.title.set_fontsize(10)
    ax.legend(frameon=False, fontsize=8, loc="lower left", markerscale=2)
    _style(ax)
    fig.tight_layout()
    _save(fig, dest / "atlas_nodes_map.png")


def plot_basins(nodes: pd.DataFrame, gauges: pd.DataFrame, dest: Path) -> None:
    fig, axes = plt.subplots(2, len(BASINS), figsize=(5.4 * len(BASINS), 9.4))
    for j, (name, (x0, x1, y0, y1)) in enumerate(BASINS.items()):
        b = nodes[nodes.node_lon.between(x0, x1) & nodes.node_lat.between(y0, y1)]
        vmax = float(np.nanmax(b[["fes_proxy_m", "eot20_proxy_m"]].to_numpy())) if len(b) else 1.0
        for i, col in enumerate(("fes_proxy_m", "eot20_proxy_m")):
            ax = axes[i, j]
            sc = ax.scatter(b.node_lon, b.node_lat, c=b[col], s=14, cmap="viridis", vmin=0, vmax=vmax, linewidths=0)
            flagged = b[b.status == "flagged"]
            if i == 0 and len(flagged):
                ax.scatter(flagged.node_lon, flagged.node_lat, s=40, facecolors="none", edgecolors="#e34948",
                           linewidths=1.0, label=f"flagged (n={len(flagged)})")
                ax.legend(frameon=False, fontsize=8, loc="lower left")
            g = gauges[gauges.lon.between(x0, x1) & gauges.lat.between(y0, y1)]
            ax.scatter(g.lon, g.lat, marker="^", s=40, color=INK)
            for _, r in g.iterrows():
                ax.annotate(f"{r.site_name}: gauge {r.gesla_harm__mhws_m:.2f} m", (r.lon, r.lat), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color=INK)
            fig.colorbar(sc, ax=ax, shrink=0.8, label="|M2|+|S2|+|K1|+|O1| (m)")
            ax.set(xlim=(x0, x1), ylim=(y0, y1), title=f"{name}: {'FES2022' if i == 0 else 'EOT20'}")
            ax.title.set_fontsize(10)
            _style(ax)
    fig.suptitle("Amplitude proxy at the FES boundary nodes (triangles: GESLA gauges with their MHWS)", color=INK)
    fig.tight_layout()
    _save(fig, dest / "atlas_basins.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plots-only", action="store_true", help="redraw the figures from the saved tables")
    args = parser.parse_args()
    out = Path(args.out_dir)
    dest = out / "analysis"
    if args.plots_only:
        gauges = pd.read_csv(dest / "atlas_gauges.csv", index_col=0)
        nodes = pd.read_parquet(dest / "atlas_nodes.parquet")
        plot_gauges(gauges, dest)
        plot_nodes_map(nodes, dest)
        plot_basins(nodes, gauges, dest)
        return
    eot = pyTMD.io.model(str(EOT20_ROOT)).elevation("EOT20")
    gauges = gauge_table(out, eot, args.workers)
    gauges.round(4).to_csv(dest / "atlas_gauges.csv")
    q = gauges[gauges.quality]
    for col, label in (("fes_gauge__mhws_m", "FES2022"), ("eot20__mhws_m", "EOT20")):
        d = (q[col] - q.gesla_harm__mhws_m).dropna()
        print(f"{label} vs gauge MHWS: median ratio {(q[col] / q.gesla_harm__mhws_m).median():.3f}, "
              f"MAE {d.abs().mean():.3f} m, within 0.1 m {(d.abs() <= 0.1).mean():.2f}, "
              f"> 0.25 m {(d.abs() > 0.25).sum()} (n={len(d)})")
    miss = q[(q.fes_gauge__mhws_m - q.gesla_harm__mhws_m).abs() > 0.25]
    print("\nat the FES-miss sites\n" + miss[["site_name", "gesla_harm__mhws_m", "fes_gauge__mhws_m", "eot20__mhws_m",
                                               "gtsm_harm__mhws_m"]].round(2).to_string())

    fes = subset_model(pyTMD.io.model(str(ensure_local_fes(FES_CACHE))).elevation(FES_MODEL), PROXY)
    eot_proxy = subset_model(pyTMD.io.model(str(EOT20_ROOT)).elevation("EOT20"), PROXY)
    nodes = node_table(eot_proxy, fes)
    nodes.to_parquet(dest / "atlas_nodes.parquet")
    flagged = nodes[nodes.status == "flagged"].sort_values("ratio_fes_eot20", ascending=False)
    flagged.round(3).to_csv(dest / "atlas_flagged_nodes.csv", index=False)
    print("\nnode status\n" + nodes.status.value_counts().to_string())
    for name, (x0, x1, y0, y1) in BASINS.items():
        b = nodes[nodes.node_lon.between(x0, x1) & nodes.node_lat.between(y0, y1)]
        print(f"\n{name}: {len(b)} nodes, status {b.status.value_counts().to_dict()}, "
              f"FES MHWS max {b.fes_mhws_m.max():.2f} m, FES proxy max {b.fes_proxy_m.max():.2f} m, "
              f"EOT20 proxy max {b.eot20_proxy_m.max():.2f} m")
    print("\nflagged nodes, top 25\n" + flagged.head(25).round(2).to_string(index=False))
    plot_gauges(gauges, dest)
    plot_nodes_map(nodes, dest)
    plot_basins(nodes, gauges, dest)


if __name__ == "__main__":
    main()
