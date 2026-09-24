"""Two grid plots of the tiles with the most extreme eikonal-vs-SFINCS
total-flooded-area disagreement: the N worst underestimations (eikonal
floods much LESS than SFINCS) and the N worst overestimations (eikonal
floods much MORE), ranked by the ratio eikonal_km2/sfincs_km2 among tiles
where EITHER model's own flooded area exceeds MIN_FLOODED_KM2_DEFAULT
(2026-09-24, user direction - excludes tiles whose extreme ratio comes from
a trivially small absolute base, e.g. "5 agreed cells, 5 extra" looking
like a 2x factor despite not being a meaningful disagreement; no gate on
the earlier "agreed/matched area" - gate is on either model's own total).

Reuses plot_worst_tiles_comparison.py's own build_rgb() (agree/SFINCS-only/
eikonal-only categorical map on the SFINCS subgrid) and all_tiles_summary.csv
(aggregate_tile_summaries.py's output - already has eikonal_km2/sfincs_km2
per tile from postprocess_tile_summary.py, no separate merge needed).

Run under gfm_python_preprocessing (NOT hydromt-sfincs-dev) -
matplotlib.pyplot.savefig() crashes with exit code 127 under
hydromt-sfincs-dev, a real documented issue this session (broken native
BLAS/font-rendering backend in that env).

Usage:
    python plot_eikonal_disagreement_extremes.py
    python plot_eikonal_disagreement_extremes.py --base-dir-name validation_sfincs_v3 --n-tiles 9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gfm_config import read_root  # noqa: E402
from plot_worst_tiles_comparison import (  # noqa: E402
    COLOR_AGREE, COLOR_DRY, COLOR_EIKONAL_ONLY, COLOR_OCEAN, COLOR_SFINCS_ONLY, COLOR_WATERBODY, build_rgb,
)

N_TILES_DEFAULT = 9
NCOLS_DEFAULT = 3
MIN_FLOODED_KM2_DEFAULT = 0.5  # candidate gate (2026-09-24, user direction): only consider tiles
# where EITHER model's own flooded area exceeds this - excludes tiles whose extreme ratio comes
# from a trivially small absolute base (e.g. "5 agreed cells, 5 extra" looks like a 2x factor but
# isn't a meaningful disagreement).


def _plot_grid(df: pd.DataFrame, root: Path, base_dir_name: str, title: str, out_path: Path, ncols: int) -> None:
    n = len(df)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.8))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, df.iterrows()):
        tile_id = str(int(row["tile_id"]))
        try:
            rgb = build_rgb(tile_id, root, base_dir_name)
        except Exception as e:
            ax.text(0.5, 0.5, f"{tile_id}\nERROR: {type(e).__name__}", ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.imshow(rgb, origin="upper")
        ax.set_title(
            f"tile {tile_id} (set {row.get('set', '?')})\n"
            f"eikonal={row['eikonal_km2']:.2f} sfincs={row['sfincs_km2']:.2f} km2, ratio={row['ratio']:.2f}",
            fontsize=9,
        )
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[n:]:
        ax.set_visible(False)

    handles = [
        mpatches.Patch(facecolor=COLOR_OCEAN, edgecolor="black", label="ocean"),
        mpatches.Patch(facecolor=COLOR_WATERBODY, edgecolor="black", label="lake/river"),
        mpatches.Patch(facecolor=COLOR_DRY, edgecolor="black", label="dry land"),
        mpatches.Patch(facecolor=COLOR_AGREE, edgecolor="black", label="agree (both wet)"),
        mpatches.Patch(facecolor=COLOR_SFINCS_ONLY, edgecolor="black", label="SFINCS only"),
        mpatches.Patch(facecolor=COLOR_EIKONAL_ONLY, edgecolor="black", label="eikonal only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs_v3")
    parser.add_argument("--n-tiles", type=int, default=N_TILES_DEFAULT)
    parser.add_argument("--min-flooded-km2", type=float, default=MIN_FLOODED_KM2_DEFAULT)
    parser.add_argument("--ncols", type=int, default=NCOLS_DEFAULT)
    args = parser.parse_args()

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name

    df = pd.read_csv(base_dir / "all_tiles_summary.csv")
    df = df.dropna(subset=["eikonal_km2", "sfincs_km2"])
    df = df[df["sfincs_km2"] > 0].copy()
    df["ratio"] = df["eikonal_km2"] / df["sfincs_km2"]
    print(f"{len(df)} tile(s) with valid eikonal_km2/sfincs_km2 (sfincs_km2 > 0)")

    n_before_gate = len(df)
    df = df[(df["eikonal_km2"] > args.min_flooded_km2) | (df["sfincs_km2"] > args.min_flooded_km2)].copy()
    print(f"{len(df)} of {n_before_gate} tile(s) have eikonal_km2 or sfincs_km2 > {args.min_flooded_km2} km2")

    under = df.sort_values("ratio").head(args.n_tiles).reset_index(drop=True)
    over = df.sort_values("ratio", ascending=False).head(args.n_tiles).reset_index(drop=True)
    print(f"worst {len(under)} underestimation(s): ratio {under['ratio'].min():.3f} to {under['ratio'].max():.3f}")
    print(f"worst {len(over)} overestimation(s): ratio {over['ratio'].min():.3f} to {over['ratio'].max():.3f}")

    _plot_grid(
        under, root, args.base_dir_name,
        f"Worst {len(under)} eikonal underestimations vs SFINCS (lowest eikonal_km2/sfincs_km2)",
        base_dir / "worst_eikonal_underestimation.png", args.ncols,
    )
    _plot_grid(
        over, root, args.base_dir_name,
        f"Worst {len(over)} eikonal overestimations vs SFINCS (highest eikonal_km2/sfincs_km2)",
        base_dir / "worst_eikonal_overestimation.png", args.ncols,
    )


if __name__ == "__main__":
    main()
