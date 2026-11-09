"""Plot depth accuracy and extent agreement vs SFINCS across each swept variant's runs (report only).

Reads {base}/variant_runs_depth_accuracy.csv and variant_runs_pooled.csv (compute_variant_metrics.py)
and writes {base}/figures/depth_accuracy_by_run[_<group>].png: per variant with more than one run,
against its friction value (alpha for the duration variants), the share of SFINCS-wet cells within
10/20/50 cm, the bias per SFINCS depth class on both-wet cells, and CSI. The selected run is marked.

Usage:
    python plot_depth_accuracy.py --base-dir-name calibration_roughness_SFINCS [--group calibration]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import DISPLAY_LABEL, GROUPS  # noqa: E402
from gfm_config import read_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from map_style import draw_panel_letter  # noqa: E402

TAU_COLORS = {0.1: "#9ecae1", 0.2: "#4292c6", 0.5: "#08306b"}
BIAS_CLASSES = {"0.1-0.3": "#fdae6b", "0.5-1": "#e6550d", "2-3": "#7f2704"}
WEIGHT_STYLE = {"area": "-", "pop": "--"}


def swept_axis(variant: str) -> tuple[str, str]:
    if variant.endswith("_duration"):
        return "alpha", "duration exponent alpha"
    if "landcover" in variant:
        return "parameter", "friction scale factor (per m)"
    return "parameter", "attenuation rate (m/km)"


def plot(depth: pd.DataFrame, pooled: pd.DataFrame, selection: dict, group: str, out: Path) -> None:
    variants = [v for v in DISPLAY_LABEL if pooled[pooled.variant == v].run_key.nunique() > 1]
    if not variants:
        print("no variant with more than one run: nothing to plot")
        return
    fig, axes = plt.subplots(3, len(variants), figsize=(3.4 * len(variants) + 2.4, 8.4), squeeze=False,
                             sharey="row", layout="constrained")
    for j, variant in enumerate(variants):
        col, xlabel = swept_axis(variant)
        d = depth[depth.variant == variant].sort_values(col)
        p = pooled[pooled.variant == variant].sort_values(col)
        ref = d[(d.cells == "reference_wet") & (d.sfincs_class_m == "all")]
        for weight, ls in WEIGHT_STYLE.items():
            w = ref[ref.weight == weight]
            for tau, color in TAU_COLORS.items():
                axes[0, j].plot(w[col], w[f"within_{tau:g}m"], color=color, ls=ls, lw=1.6, marker="o", ms=3,
                                label=f"within {tau * 100:.0f} cm, {'area' if weight == 'area' else 'population'}")
        both = d[(d.cells == "both_wet") & (d.weight == "area")]
        for cls, color in BIAS_CLASSES.items():
            b = both[both.sfincs_class_m == cls]
            axes[1, j].plot(b[col], b.bias_m, color=color, lw=1.6, marker="o", ms=3, label=f"SFINCS {cls} m")
        axes[1, j].axhline(0, color="#525252", lw=0.8)
        axes[2, j].plot(p[col], p.CSI, color="#08519c", lw=1.6, marker="o", ms=3, label="area")
        if "CSI_pop" in p:
            axes[2, j].plot(p[col], p.CSI_pop, color="#08519c", ls="--", lw=1.6, marker="o", ms=3, label="population")
        chosen = (selection.get("variants") or {}).get(variant)
        if chosen and chosen.get(col) is not None:
            for ax in axes[:, j]:
                ax.axvline(chosen[col], color="#bdbdbd", lw=1.0, zorder=0)
        axes[0, j].set_title(DISPLAY_LABEL[variant], fontsize=9, pad=14)
        axes[2, j].set_xlabel(xlabel)
    axes[0, 0].set(ylabel="share of SFINCS-wet cells within", ylim=(0, 1))
    axes[1, 0].set_ylabel("bias, eikonal - SFINCS (m)\nboth-wet cells, by SFINCS depth")
    axes[2, 0].set(ylabel="CSI (extent)", ylim=(0, 1))
    for ax in axes[:, -1]:
        ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    for k, ax in enumerate(axes.T.ravel()):
        draw_panel_letter(ax, "abcdefghijklmnopqrstuvwxyz"[k] if k < 26 else None)
    fig.suptitle(f"Depth and extent agreement vs SFINCS per run, {group} group\n"
                 "grey line: selected run (by CSI; depth is reported only)", fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="calibration")
    args = parser.parse_args()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    depth = pd.read_csv(base_dir / "variant_runs_depth_accuracy.csv")
    pooled = pd.read_csv(base_dir / "variant_runs_pooled.csv")
    selection_path = base_dir / SELECTION_NAME
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.group == "calibration" else f"_{args.group}"
    plot(depth[depth.group == args.group], pooled[pooled.group == args.group], selection, args.group,
         fig_dir / f"depth_accuracy_by_run{sfx}.png")


if __name__ == "__main__":
    main()
