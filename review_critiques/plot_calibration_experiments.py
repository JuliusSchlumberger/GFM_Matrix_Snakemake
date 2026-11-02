"""Figures and tables for the calibration experiments (critiques 2 and 3).

Pooled metrics sum area counts over tiles before forming CSI / bias, as calibration_results.

Usage:
    python review_critiques/plot_calibration_experiments.py --results-dir <dir> --fig-dir <dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

BASE = Path("P:/11212688-004-global-floodmaps/modelling/calibration_roughness_SFINCS")
VARIANT_STYLE = {
    "landcover": ("land-cover friction", "k", "-o"),
    "A_volume_cap": ("A: basin volume cap", "tab:blue", "-s"),
    "B_duration_a0.5": ("B: duration friction a=0.5", "tab:orange", "-^"),
    "B_duration_a1": ("B: duration friction a=1", "tab:red", "-v"),
}


def metrics(df: pd.DataFrame) -> pd.Series:
    tp, fp, fn = df.matched_km2.sum(), df.model_only_km2.sum(), df.sfincs_only_km2.sum()
    return pd.Series({"CSI": tp / (tp + fp + fn), "bias": (tp + fp) / (tp + fn), "HT": tp / (tp + fn),
                      "FAR": fp / (tp + fp) if tp + fp else np.nan, "n_tiles": df.tile_id.nunique()})


def tile_csi(df: pd.DataFrame) -> pd.Series:
    tot = df.matched_km2 + df.model_only_km2 + df.sfincs_only_km2
    return (df.matched_km2 / tot.replace(0, np.nan)).set_axis(df.tile_id)


def plot_v0(df: pd.DataFrame, fig_dir: Path) -> None:
    ref = pd.read_csv(BASE / "friction_sweep_per_tile_metrics.csv")
    ref = ref[ref.friction_scale_factor == 12.0].assign(tile_id=lambda d: d.tile_id.astype(str)).set_index("tile_id")
    ours = df[(df.variant == "landcover") & (df.param == 12.0)].set_index("tile_id")
    j = ours.join(ref, rsuffix="_ref", how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (a, b, name) in zip(axes, [("matched_km2", "matched_km2_ref", "matched"),
                                       ("model_only_km2", "eikonal_only_km2", "eikonal only"),
                                       ("sfincs_only_km2", "sfincs_only_km2_ref", "SFINCS only")]):
        ax.scatter(j[b], j[a], s=10)
        lim = max(j[a].max(), j[b].max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        rel = (j[a].sum() - j[b].sum()) / j[b].sum() * 100
        ax.set_title(f"{name} (km²), pooled diff {rel:+.2f}%")
        ax.set_xlabel("stored calibration result (HPC)")
        ax.set_ylabel("this harness")
    fig.suptitle(f"Harness check: land-cover friction fsf 12, {len(j)} tiles")
    fig.tight_layout()
    fig.savefig(fig_dir / "V0_harness_vs_stored.png", dpi=140)
    plt.close(fig)


def plot_c2(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    lu = df[df.variant == "landcover"]
    uni = df[df.variant == "uniform"]
    lu_rate = lu.groupby("param").mean_m_per_km.median()
    lu_m = lu.groupby("param").apply(metrics).assign(rate=lu_rate)
    uni_m = uni.groupby("param").apply(metrics).assign(rate=lambda d: d.index)
    eq = metrics(df[df.variant == "uniform_equal_mean"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key in zip(axes, ["CSI", "bias"]):
        ax.plot(lu_m.rate, lu_m[key], "-o", color="k", label="land-cover friction (x = median tile mean rate)")
        for fsf, row in lu_m.iterrows():
            ax.annotate(f"fsf {fsf:g}", (row.rate, row[key]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.plot(uni_m.rate, uni_m[key], "-s", color="tab:green", label="uniform attenuation")
        ax.plot([df[df.variant == "uniform_equal_mean"].param.median()], [eq[key]], "D", color="tab:purple",
                ms=9, label="uniform at each tile's land-cover mean (fsf 12)")
        ax.axvline(0.2, color="0.5", ls=":", label="0.2 m/km (Vafeidis, Flow-tub)")
        ax.set_xscale("log")
        ax.set_xlabel("attenuation rate (m/km)")
        ax.set_ylabel(f"pooled {key}")
        ax.grid(alpha=0.3)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Critique 2: land-cover vs uniform attenuation ({lu.tile_id.nunique()} tiles vs SFINCS RP100)")
    fig.tight_layout()
    fig.savefig(fig_dir / "C2_1_pooled_landcover_vs_uniform.png", dpi=140)
    plt.close(fig)

    best_rate = float(uni_m.CSI.idxmax())
    csi_lu = tile_csi(lu[lu.param == 12.0])
    csi_un = tile_csi(uni[uni.param == best_rate])
    csi_eq = tile_csi(df[df.variant == "uniform_equal_mean"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(csi_lu, csi_un.reindex(csi_lu.index), s=12)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("CSI land-cover fsf 12")
    axes[0].set_ylabel(f"CSI uniform {best_rate:g} m/km (best pooled)")
    axes[1].hist((csi_lu - csi_un.reindex(csi_lu.index)).dropna(), bins=40, color="tab:green", alpha=0.7,
                 label=f"vs uniform {best_rate:g} m/km")
    axes[1].hist((csi_lu - csi_eq.reindex(csi_lu.index)).dropna(), bins=40, color="tab:purple", alpha=0.5,
                 label="vs uniform equal mean")
    axes[1].axvline(0, color="k")
    axes[1].set_xlabel("CSI land-cover - CSI uniform (per tile)")
    axes[1].legend()
    best_per_tile = uni.assign(csi=tile_csi(uni).values).sort_values("csi").groupby("tile_id").tail(1)
    axes[2].hist(best_per_tile.param, bins=np.array(sorted(uni.param.unique()) + [1.2]) * 0.999, color="tab:green")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("per-tile best uniform rate (m/km)")
    axes[2].set_ylabel("tiles")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "C2_2_per_tile_landcover_vs_uniform.png", dpi=140)
    plt.close(fig)
    table = pd.concat([lu_m.assign(variant="land-cover", param=lu_m.index),
                       uni_m.assign(variant="uniform", param=uni_m.index),
                       pd.DataFrame([eq]).assign(variant="uniform equal mean")], ignore_index=True)
    table.to_csv(fig_dir / "C2_table_A3_candidates.csv", index=False)
    return table


def plot_c3(df: pd.DataFrame, worst: list[str], fig_dir: Path) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    for variant, (label, color, style) in VARIANT_STYLE.items():
        sub = df[df.variant == variant]
        if sub.empty:
            continue
        m = sub.groupby("param").apply(metrics)
        axes[0].plot(m.index, m.CSI, style, color=color, label=label)
        axes[1].plot(m.index, m.bias, style, color=color, label=label)
        worst_sub = sub[sub.tile_id.isin(worst)]
        if not worst_sub.empty:
            mw = worst_sub.groupby("param").apply(metrics)
            axes[2].plot(mw.index, mw.CSI, style, color=color, label=label)
        rows += [{"variant": variant, "fsf": p, **r} for p, r in m.iterrows()]
    for ax, title in zip(axes, ["pooled CSI, all sampled tiles", "pooled bias", "pooled CSI, 12 worst tiles"]):
        ax.set_xlabel("friction scale factor")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle("Critique 3: event-volume corrections, recalibrated over the friction scale factor")
    fig.tight_layout()
    fig.savefig(fig_dir / "C3_1_pooled_corrections_vs_fsf.png", dpi=140)
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(fig_dir / "C3_table_corrections.csv", index=False)

    if not worst:
        return table
    best = table.loc[table.groupby("variant").CSI.idxmax()].set_index("variant").fsf
    fig, ax = plt.subplots(figsize=(15, 5))
    width = 0.8 / len(best)
    for i, (variant, fsf) in enumerate(best.items()):
        sub = df[(df.variant == variant) & (df.param == fsf) & df.tile_id.isin(worst)]
        csi = tile_csi(sub).reindex(sorted(worst))
        ax.bar(np.arange(len(worst)) + i * width, csi.values, width, color=VARIANT_STYLE[variant][1],
               label=f"{VARIANT_STYLE[variant][0]} (fsf {fsf:g})")
    ax.set_xticks(np.arange(len(worst)) + 0.4 - width / 2)
    ax.set_xticklabels(sorted(worst))
    ax.set_xlabel("worst tile (fsf 12 panel selection)")
    ax.set_ylabel("CSI vs SFINCS")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "C3_2_worst_tiles_csi.png", dpi=140)
    plt.close(fig)
    return table


def plot_c3_basins(results_dir: Path, fig_dir: Path) -> None:
    path = results_dir / "basins.jsonl"
    if not path.exists():
        return
    b = pd.read_json(path, lines=True)
    b = b[(b.eikonal_m3 > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(b.eikonal_m3, b.inflow_m3.clip(lower=1), s=4, c=np.where(b.capped, "tab:blue", "0.6"))
    lim = [1, max(b.eikonal_m3.max(), b.inflow_m3.max())]
    axes[0].plot(lim, lim, "k--", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("eikonal basin volume (m³)")
    axes[0].set_ylabel("weir inflow over the sill (m³)")
    axes[0].set_title(f"{len(b)} basins (fsf 12), blue = capped ({b.capped.mean():.0%})")
    removed = (b.eikonal_m3 - np.minimum(b.inflow_m3, b.eikonal_m3)).groupby(b.tile_id).sum()
    total = b.eikonal_m3.groupby(b.tile_id).sum()
    axes[1].hist((removed / total).dropna() * 100, bins=40)
    axes[1].set_xlabel("share of basin volume removed by the cap, per tile (%)")
    axes[1].set_ylabel("tiles")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "C3_3_basin_volumes.png", dpi=140)
    plt.close(fig)


def plot_maps(results_dir: Path, fig_dir: Path) -> None:
    panels = [("sfincs", "SFINCS hmax"), ("landcover_fsf12", "land-cover fsf 12"), ("A_fsf12", "A: volume cap"),
              ("B_fsf12_a1", "B: duration a=1"), ("uniform_0.2", "uniform 0.2 m/km")]
    for npz in sorted((results_dir / "maps").glob("*.npz")):
        d = np.load(npz)
        score = d["score"]
        basins = ((d["fill"] - d["dem"]) > 0.05) & np.isfinite(d["fill"])
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.2))
        for ax, (key, title) in zip(axes, panels):
            if key not in d:
                ax.set_axis_off()
                continue
            arr = d[key].astype(np.float32)
            wet = np.nan_to_num(arr) > 0.10
            ax.imshow(np.where(score, 0.3, 0.0), cmap="Greys", vmin=0, vmax=1)
            ax.imshow(np.where(wet, np.nan_to_num(arr), np.nan), cmap="Blues", vmin=0, vmax=3)
            ax.contour(ndimage.binary_opening(basins, iterations=1), levels=[0.5], colors="tab:red", linewidths=0.5)
            ax.set_title(title)
            ax.set_axis_off()
        fig.suptitle(f"Tile {npz.stem}: depth (m, 0-3); red = enclosed basins (spill level > ground); grey = scored land")
        fig.tight_layout()
        fig.savefig(fig_dir / f"C3_4_maps_tile{npz.stem}.png", dpi=100)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    args = parser.parse_args()
    results_dir, fig_dir = Path(args.results_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(results_dir / "results.jsonl", lines=True, dtype={"tile_id": str})
    errors = df[df.variant.isin(["error", "no_seeds"])]
    if len(errors):
        errors.to_csv(fig_dir / "calibration_errors.csv", index=False)
    df = df[~df.variant.isin(["error", "no_seeds"])]
    sample = json.loads((results_dir / "sample.json").read_text())
    worst = sample["worst"]
    # Keep the sampled tiles and only (variant, param) pairs run on every one of them, so
    # pooled curves never mix tile sets.
    df = df[df.tile_id.isin(sample["tiles"])]
    n_tiles = df.tile_id.nunique()
    coverage = df.groupby(["variant", "param"]).tile_id.transform("nunique")
    df = df[coverage == n_tiles]
    plot_v0(df, fig_dir)
    plot_c2(df, fig_dir)
    plot_c3(df, [w for w in worst if w in set(df.tile_id)], fig_dir)
    plot_c3_basins(results_dir, fig_dir)
    plot_maps(results_dir, fig_dir)
    print(f"{df.tile_id.nunique()} tiles, {len(errors)} errors; figures in {fig_dir}")


if __name__ == "__main__":
    main()
