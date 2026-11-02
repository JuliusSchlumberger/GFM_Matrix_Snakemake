"""Figures comparing the flood-fraction and water-level protection rules.

P1: effective protection RP vs SLR over all COAST-RP stations (water-level rule).
P2: case-study EAI per measure under both rules (exposure CSVs of compute_exposure_analysis).
P3: maps of cells where the rules disagree, for selected (RP, SLR) events.

Usage:
    python review_critiques/plot_protection_rule.py --fig-dir <dir>
        [--ff-dir <exposure out, flood_fraction rule>] [--wl-dir <exposure out, water_level rule>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

from config_utils import load_config  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml"
COASTRP_RPS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
DESIGN_RPS = [2, 10, 100, 1000]
SLR_M = np.linspace(0.0, 2.0, 41)


def effective_protection_rp(storm_tide: np.ndarray, design_rp: int, slr_m: float) -> np.ndarray:
    """Return per station the RP whose storm tide equals design level - SLR (log-RP interpolation).

    `storm_tide` is (n_stations, len(COASTRP_RPS)) in m. Below RP1 the standard is lost
    entirely and reported as 1 (a defence that fails every year).
    """
    log_rp = np.log(np.array(COASTRP_RPS, dtype=float))
    target = storm_tide[:, COASTRP_RPS.index(design_rp)] - slr_m
    out = np.empty(len(storm_tide))
    for i, (row, tgt) in enumerate(zip(storm_tide, target)):
        out[i] = np.exp(np.interp(tgt, row, log_rp, left=0.0, right=log_rp[-1]))
    return out


def plot_p1(cfg: dict, fig_dir: Path) -> None:
    path = Path(cfg["paths"]["processed_inputs_dir"]) / "COAST-RP_preprocessed.nc"
    with xr.open_dataset(path) as ds:
        st = np.column_stack([ds[f"storm_tide_rp_{rp:04d}"].values for rp in COASTRP_RPS]).astype(float)
    st = st[np.all(np.isfinite(st), axis=1) & np.all(np.diff(st, axis=1) >= 0, axis=1)]
    sample = st[:: max(1, len(st) // 4000)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(DESIGN_RPS)))
    for d, col in zip(DESIGN_RPS, colors):
        eff = np.array([effective_protection_rp(sample, d, s) for s in SLR_M])
        q25, q50, q75 = np.percentile(eff, [25, 50, 75], axis=1)
        axes[0].plot(SLR_M, q50, color=col, lw=2, label=f"design RP{d}")
        axes[0].fill_between(SLR_M, q25, q75, color=col, alpha=0.2)
        lost = (eff <= 1.0 + 1e-9).mean(axis=1)
        axes[1].plot(SLR_M, 100 * lost, color=col, lw=2, label=f"design RP{d}")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.9, 1200)
    axes[0].set_xlabel("SLR (m)")
    axes[0].set_ylabel("effective protection RP (years)")
    axes[0].set_title("Water-level rule: protection standard eroded by SLR\n(median, IQR over COAST-RP stations)")
    axes[0].legend()
    axes[1].set_xlabel("SLR (m)")
    axes[1].set_ylabel("stations where the defence fails every year (%)")
    axes[1].set_title("Design level below annual storm tide (RP1) + SLR")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "P1_effective_protection_rp_vs_slr.png", dpi=150)
    plt.close(fig)
    print(f"  wrote P1 ({len(sample)} of {len(st)} stations)")


def _read_base(d: Path, label: str) -> pd.Series:
    df = pd.read_csv(d / f"exposure_{label}_base.csv", index_col=0)
    return df.sum(axis=0)


def plot_p2(ff_dir: Path, wl_dir: Path, fig_dir: Path, slr_intensities: list[str]) -> None:
    labels = ["baseline"] + [f"protect_{s}" for s in slr_intensities] + [f"retreat_{s}" for s in slr_intensities]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    groups = [["baseline"], [f"protect_{s}" for s in slr_intensities], [f"retreat_{s}" for s in slr_intensities]]
    titles = ["No adaptation", "Protect (design raised to SLR intensity)", "Retreat"]
    for ax, group, title in zip(axes, groups, titles):
        for i, label in enumerate(group):
            ff, wl = _read_base(ff_dir, label), _read_base(wl_dir, label)
            x = [int(c.split("_")[1]) / 1000 for c in ff.index]
            col = plt.cm.tab10(i)
            ax.plot(x, ff.values, "--o", color=col, label=f"{label} flood fraction")
            ax.plot(x, wl.values, "-s", color=col, label=f"{label} water level")
        ax.set_title(title)
        ax.set_xlabel("SLR (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("EAI (people / yr), case-study countries summed")
    fig.suptitle("Bangkok / Chao Phraya: EAI under the flood-fraction (dashed) vs water-level (solid) rule")
    fig.tight_layout()
    fig.savefig(fig_dir / "P2_case_eai_by_rule.png", dpi=150)
    plt.close(fig)
    rows = []
    for label in labels:
        ff, wl = _read_base(ff_dir, label), _read_base(wl_dir, label)
        for slr in ff.index:
            rows.append({"scenario": label, "slr": slr, "eai_flood_fraction": ff[slr], "eai_water_level": wl[slr]})
    pd.DataFrame(rows).to_csv(fig_dir / "P2_case_eai_by_rule.csv", index=False)
    print("  wrote P2")


def plot_p3(cfg: dict, fig_dir: Path, events: list[tuple[int, str]]) -> None:
    from compute_exposure_analysis import _exceedance, _load_chunk, load_analysis_context
    from _common import discover_chunk_ids
    from exposure_analysis import build_adapt_protection_fraction

    cfg = {**cfg, "protection": {**cfg["protection"], "rule": "water_level"}}
    ctx = load_analysis_context(cfg)
    chunk_ids, _ = discover_chunk_ids(ctx.flood_frac_dir, ctx.chunks_dir)
    cmap = ListedColormap(["#f0f0f0", "#9ecae1", "#de2d26", "#fdae6b", "#3182bd"])
    names = ["dry", "protected (both)", "exposed (both)", "exposed only water level", "exposed only flood fraction"]
    for cid in chunk_ids:
        c = _load_chunk(cid, ctx.chunks_dir, ctx.flood_frac_dir, ctx.return_periods, ctx.slr_scenarios,
                        ctx.iso_lookup, ctx.water_level_protection)
        if not any((g > 0).any() for g in c.ff.values()):
            continue
        apf = build_adapt_protection_fraction(c.ff, ctx.return_periods, ctx.slr_baseline, ctx.rp_applied, c.geo)
        exc = _exceedance(c, ctx.water_level_protection, ctx.slr_baseline)
        fig, axes = plt.subplots(1, len(events), figsize=(5 * len(events), 5))
        for ax, key in zip(np.atleast_1d(axes), events):
            if key not in c.ff:
                ax.set_axis_off()
                continue
            ff = c.ff[key]
            wet = ff > 0
            ff_exp, wl_exp = wet & (ff > apf), wet & exc[key]
            cls = np.zeros(ff.shape, dtype=int)
            cls[wet & ~ff_exp & ~wl_exp] = 1
            cls[ff_exp & wl_exp] = 2
            cls[wl_exp & ~ff_exp] = 3
            cls[ff_exp & ~wl_exp] = 4
            ax.imshow(cls, cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
            pop_diff = ((wl_exp.astype(float) - ff_exp.astype(float)) * ff * c.pop).sum()
            ax.set_title(f"RP{key[0]} {key[1]}\nexposed pop. water level - flood fraction: {pop_diff:,.0f}")
            ax.set_axis_off()
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(5)]
        fig.legend(handles, names, loc="lower center", ncol=5)
        fig.suptitle(f"Chunk {cid}: baseline protection, where the two rules disagree")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(fig_dir / f"P3_rule_disagreement_{cid}.png", dpi=130)
        plt.close(fig)
        print(f"  wrote P3 {cid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fig-dir", required=True)
    parser.add_argument("--ff-dir", default=None)
    parser.add_argument("--wl-dir", default=None)
    args = parser.parse_args()
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    plot_p1(cfg, fig_dir)
    if args.ff_dir and args.wl_dir:
        plot_p2(Path(args.ff_dir), Path(args.wl_dir), fig_dir, cfg["adaptation"]["slr_intensities"])
        plot_p3(cfg, fig_dir, [(2, "SLR_1000"), (100, "SLR_0"), (1000, "SLR_0"), (100, "SLR_1000")])


if __name__ == "__main__":
    main()
