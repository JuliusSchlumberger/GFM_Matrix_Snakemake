"""Adaptation pathways map ("metro map"): a curated subset of the chains
compute_adaptation_pathways.py evaluated, drawn as one horizontal lane per
chain - shared "do nothing" trunk from 2020 to the entity's own baseline
tipping year, then one coloured segment per stage (colour = measure family,
linewidth = design intensity), ending either in an arrow (the chain holds
beyond --to-year) or an X at the year it ultimately fails.

Selecting ~10 of 324 chains per (entity, ssp): all 8 single-stage chains
(every measure/intensity tried alone - the branching choice right after
baseline tips) plus up to 2 more multi-stage chains - the shortest SUCCESSFUL
multi-stage chain if any singles don't already succeed and multi-stage ones
do, else (the --ssp5-style "nothing fully succeeds" case) the best-performing
FAILING multi-stage chains, ranked by how late they push the eventual
tipping year. Override the total with --n-chains.

Reads compute_adaptation_pathways.py's own output CSVs directly (pathways_countries.csv,
pathways_continents.csv) - no EAI recomputation, this is pure post-processing,
so it's fast regardless of how long the underlying computation took.

Writes one combined figure (one panel per entity, --ssp only) to --outdir:
  adaptation_pathways_{ssp}.png
Entities whose baseline never tips ("no_adaptation_needed") are skipped - no
pathway to show.

Usage:
    python analysis/plot_adaptation_pathways.py \\
        --countries-csv <figures>/pathways_countries.csv --continents-csv <figures>/pathways_continents.csv \\
        --ssp SSP5 [--entities NGA THA EGY Africa Asia] [--n-chains 10] [--to-year 2150] [--outdir <figures>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MEASURE_COLORS = {"avoid": "#4C72B0", "protect": "#DD8452", "retreat": "#55A868", "protect_avoid": "#8172B2"}
INTENSITY_LW = {"small": 2.5, "large": 5.0}
MEASURE_LABEL = {"avoid": "Avoid", "protect": "Protect", "retreat": "Retreat", "protect_avoid": "Protect + Avoid"}


def _block_parts(block: str) -> tuple[str, str]:
    measure, _, intensity = block.rpartition("_")
    return measure, intensity


def _n_stages(chain: str) -> int:
    return chain.count(">") + 1


def select_chains(g: pd.DataFrame, n_total: int = 10) -> pd.DataFrame:
    g = g.copy()
    g["n_stages"] = g["chain"].apply(_n_stages)
    singles = g[g["n_stages"] == 1]
    multi = g[g["n_stages"] > 1]
    remaining = n_total - len(singles)
    if remaining <= 0:
        return singles.head(n_total)

    succ_multi = multi[multi["outcome"].str.startswith("success")]
    if not succ_multi.empty:
        extra = succ_multi.sort_values("n_stages").head(remaining)
    else:
        fails_multi = multi[multi["outcome"] == "fails"].copy()
        fails_multi["final_tip_year"] = fails_multi["final_tip_year"].astype(float)
        extra = fails_multi.sort_values("final_tip_year", ascending=False).head(remaining)
    return pd.concat([singles, extra], ignore_index=True)


def chain_segments(row: pd.Series, to_year: int) -> tuple[list[tuple[str, float, float]], bool]:
    """[(block, start_year, end_year), ...] for one chain row, plus whether it succeeds."""
    blocks = [b.strip() for b in row["chain"].split(">")]
    raw = row["stage_years"]
    switch_years = [float(y) for y in str(raw).split(";") if y.strip()] if pd.notna(raw) and str(raw).strip() else []
    success = str(row["outcome"]).startswith("success")
    boundaries = [float(row["baseline_tip_year"])] + switch_years + ([float(to_year)] if success else [])
    return list(zip(blocks, boundaries[:-1], boundaries[1:])), success


def _sort_key(row: pd.Series) -> tuple:
    success = str(row["outcome"]).startswith("success")
    return (0 if success else 1, row["n_stages"], -(float(row["final_tip_year"]) if not success else 0))


def plot_entity_panel(ax: plt.Axes, entity: str, g: pd.DataFrame, n_chains: int, to_year: int) -> None:
    baseline_tip = float(g["baseline_tip_year"].iloc[0])
    picked = select_chains(g, n_chains)
    picked = picked.loc[picked.apply(_sort_key, axis=1).sort_values().index]

    n = len(picked)
    for row_i, (_, row) in enumerate(picked.iterrows()):
        y = n - row_i  # best/first chain at the top
        ax.plot([2020, baseline_tip], [y, y], color="0.6", lw=2.0, zorder=2, solid_capstyle="butt")
        segments, success = chain_segments(row, to_year)
        for block, y0, y1 in segments:
            measure, intensity = _block_parts(block)
            ax.plot([y0, y1], [y, y], color=MEASURE_COLORS.get(measure, "0.3"),
                    lw=INTENSITY_LW.get(intensity, 3.0), zorder=3, solid_capstyle="butt")
            ax.plot(y0, y, "o", ms=5, mfc="white", mec=MEASURE_COLORS.get(measure, "0.3"), mew=1.6, zorder=4)
        end_x = segments[-1][2]
        if success:
            ax.plot(end_x, y, ">", ms=9, color=MEASURE_COLORS.get(_block_parts(segments[-1][0])[0], "0.3"), zorder=5)
        else:
            ax.plot(end_x, y, "x", ms=9, mew=2.2, color="0.15", zorder=5)
        ax.annotate(row["chain"], xy=(to_year, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, va="center", ha="left", color="0.2")

    ax.axvline(baseline_tip, color="0.5", lw=0.8, ls=":")
    ax.set_xlim(2020, to_year + 1)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    ax.set_title(entity, fontsize=13, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries-csv", required=True)
    parser.add_argument("--continents-csv", required=True)
    parser.add_argument("--ssp", required=True)
    parser.add_argument("--entities", nargs="+", default=None,
                        help="default: every entity with a real pathway (baseline tips) under --ssp")
    parser.add_argument("--n-chains", type=int, default=10)
    parser.add_argument("--to-year", type=int, default=2150)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(args.countries_csv), pd.read_csv(args.continents_csv)], ignore_index=True)
    sub = df[(df["ssp"] == args.ssp) & (df["outcome"] != "no_adaptation_needed")]

    entities = args.entities or list(dict.fromkeys(sub["entity"]))
    entities = [e for e in entities if e in set(sub["entity"])]
    if not entities:
        sys.exit(f"No entities with a real pathway under {args.ssp} (everyone's baseline already holds to {args.to_year}?)")

    ncols = min(3, len(entities))
    nrows = int(np.ceil(len(entities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols + 1.0, 4.8 * nrows), squeeze=False)
    for ax, entity in zip(axes.flat, entities):
        plot_entity_panel(ax, entity, sub[sub["entity"] == entity], args.n_chains, args.to_year)
        ax.set_xlabel("Year", fontsize=11)
    for ax in axes.flat[len(entities):]:
        ax.set_visible(False)

    handles = [Line2D([], [], color=c, lw=4, label=MEASURE_LABEL[m]) for m, c in MEASURE_COLORS.items()]
    handles += [Line2D([], [], color="0.3", lw=INTENSITY_LW["small"], label="design: small (SLR 0.5m)"),
                Line2D([], [], color="0.3", lw=INTENSITY_LW["large"], label="design: large (SLR 1.0m)")]
    handles += [Line2D([], [], color="0.6", lw=2.0, label="baseline (no adaptation)"),
                Line2D([], [], ls="none", marker=">", color="0.3", ms=9, label=f"holds beyond {args.to_year}"),
                Line2D([], [], ls="none", marker="x", color="0.15", mew=2.2, ms=9, label="pathway fails here")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Adaptation pathways - {args.ssp}", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adaptation_pathways_{args.ssp}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
