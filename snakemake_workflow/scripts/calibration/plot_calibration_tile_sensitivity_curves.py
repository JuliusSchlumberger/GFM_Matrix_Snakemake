"""Distribution-curve (ECDF) version of plot_calibration_tile_sensitivity_boxplot.py's
own per-tile flooded-area sensitivity to max_rounds and
obstacle_coupling.max_outer_iterations (2026-09-24, user direction: curves
instead of box-whisker).

ECDF (empirical CDF), not a KDE - the per-tile %% change data has an exact
spike at 0 (many tiles completely unaffected) plus a long, sparse tail (a
handful of tiles changing 10-20%%); a KDE's bandwidth would either smear the
spike into a false bump or need per-parameter tuning, whereas the ECDF is
bandwidth-free and shows the exact spike as a vertical jump. Also directly
answers the calibration-relevant question "what fraction of tiles change by
more than X%%?" by reading straight off the curve.

Reuses calibration_tile_sensitivity.csv (written by
plot_calibration_tile_sensitivity_boxplot.py) rather than re-reading every
tile's raster again - run that script first if the CSV doesn't exist yet.

Usage:
    python plot_calibration_tile_sensitivity_curves.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]

PARAM_REF = {"max_rounds": "max_rounds_4", "obstacle_coupling": "obstacle_coupling_off"}
PARAM_VALUE_ORDER = {"max_rounds": ["8", "20"], "obstacle_coupling": ["1", "3", "10"]}
COLORS = {"8": "#1f77b4", "20": "#d62728", "1": "#1f77b4", "3": "#ff7f0e", "10": "#d62728"}


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, (param, ref_suffix) in zip(axes, PARAM_REF.items()):
        sub = df[df["param"] == param]
        for value in PARAM_VALUE_ORDER[param]:
            s = sub.loc[sub["value"] == value, "pct_change"].dropna().to_numpy()
            if s.size == 0:
                continue
            xs, ys = _ecdf(s)
            ax.step(xs, ys, where="post", color=COLORS.get(value, "black"), linewidth=2, label=f"{value} (n={s.size})")
        ax.axvline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{param}\n(reference: {ref_suffix})", fontsize=11)
        ax.set_xlabel("% change in flooded area vs reference")
        ax.set_ylabel("cumulative fraction of tiles")
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(title=param.replace("_", " "), loc="lower right")

    fig.suptitle(
        "Eikonal flooded-area sensitivity to max_rounds and obstacle_coupling.max_outer_iterations\n"
        "(esp_fra_rp100 + nor_rp250 tiles pooled - ECDF of per-tile % change vs each parameter's own reference run)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    config_path = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    cfg = load_config(str(config_path))
    sweep_root = Path(cfg["paths"]["root"]) / "calibration_esp_fra_nor"

    csv_path = sweep_root / "calibration_tile_sensitivity.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found - run plot_calibration_tile_sensitivity_boxplot.py first "
            f"to collect the per-tile data."
        )
    df = pd.read_csv(csv_path, dtype={"value": str})

    fig_path = sweep_root / "calibration_tile_sensitivity_curves.png"
    plot_distributions(df, fig_path)


if __name__ == "__main__":
    main()
