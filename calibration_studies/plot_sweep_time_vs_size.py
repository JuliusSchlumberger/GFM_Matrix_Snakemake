"""Single figure for the sweep-budget calibration study: wall-clock time vs.
number of sweep rounds, coloured by tile size (see
calibration_studies/test_sweep_budget_calibration.py and
docs/methods_03_calibration_sensitivity.md section 2) - the companion to
plot_sweep_calibration_bands.py's own round-indexed BAND panels, this one
shows each tile as its own point instead of a pooled band, with tile size
(native cell count) as the third dimension via colour.

Reuses plot_sweep_budget_convergence.collect() directly - no new data
collection, that function already extracts exactly (n_cells, n_rounds_used,
time_to_converge_s, converged) per tile from the same raw per-tile sweep
CSVs the round-based figure reads. Unlike plot_sweep_calibration_bands.py,
this does NOT restrict to the wet-tile subset (aggregate_wet_tiles.py) -
wall-clock cost is real for a dry tile too, and narrowing to only wet tiles
would bias the picture.

A tile that never converges within N_COMPLETE_ROUNDS (right-censored, same
population the ECDF panel flags) is plotted as a distinct marker (triangle,
vs. a converged tile's circle) at its own (n_rounds_available, elapsed time
so far) - a real data point, not an extrapolation, just not a true
"rounds-to-converge" value.

No fitted trend line - raw per-tile points only.

Usage:
    python plot_sweep_time_vs_size.py <sweep_budget_dir> <figures_dir> [--epsilon 0.03]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from plot_sweep_budget_convergence import N_COMPLETE_ROUNDS, collect  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent


def make_figure(df: pd.DataFrame, epsilon: float, out_path: Path) -> None:
    converged = df[df["converged"]]
    censored = df[~df["converged"]]
    vmin, vmax = df["n_cells"].min() / 1e6, df["n_cells"].max() / 1e6

    fig, ax = plt.subplots(figsize=(10, 7.5))

    sc = ax.scatter(
        converged["n_rounds_used"], converged["time_to_converge_s"],
        c=converged["n_cells"] / 1e6, cmap="viridis", vmin=vmin, vmax=vmax,
        s=28, alpha=0.8, marker="o", edgecolors="none",
        label=f"converged (n={len(converged)})",
    )
    if len(censored):
        ax.scatter(
            censored["n_rounds_available"], censored["time_to_converge_s"],
            c=censored["n_cells"] / 1e6, cmap="viridis", vmin=vmin, vmax=vmax,
            s=55, alpha=0.9, marker="^", edgecolors="black", linewidths=0.5,
            label=f"not converged within {N_COMPLETE_ROUNDS} rounds (n={len(censored)})",
        )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("tile size (million native cells)")

    ax.set_yscale("log")
    ax.set_xlabel("sweep rounds (to converge, or run so far if not converged)")
    ax.set_ylabel("wall-clock time (s, log)")
    ax.set_title(f"Sweep wall-clock time vs. rounds, coloured by tile size - n={len(df)} tile(s), epsilon={epsilon}m")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_budget_dir")
    parser.add_argument("figures_dir")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--epsilon", type=float, default=None, help="default: config simulation.flooding.waterlevel_epsilon_m")
    args = parser.parse_args()

    cfg = load_config(args.config)
    epsilon = args.epsilon if args.epsilon is not None else float(cfg["simulation"]["flooding"]["waterlevel_epsilon_m"])

    sweep_budget_dir = Path(args.sweep_budget_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = collect(sweep_budget_dir, epsilon)
    df = df.dropna(subset=["n_cells"])
    print(f"{len(df)} tile(s) with a usable sweep trace "
          f"({int(df['converged'].sum())} converged within {N_COMPLETE_ROUNDS} rounds)")

    # Censored tiles have no real time_to_converge_s (None) - use their own
    # elapsed time through n_rounds_available instead, a real data point.
    # _tile_convergence doesn't return that elapsed time directly for the
    # censored case, so re-derive it the same way: cum_time_s at the last
    # available round's last sweep. Cheap to redo here (one extra read per
    # censored tile only).
    if (~df["converged"]).any():
        def _censored_elapsed(row) -> float:
            csv_path = sweep_budget_dir / f"{int(row['tile'])}.csv"
            raw = pd.read_csv(csv_path)
            last_idx = row["n_rounds_available"] * 4 - 1
            return float(raw["cum_time_s"].iloc[last_idx])

        censored_mask = ~df["converged"]
        df.loc[censored_mask, "time_to_converge_s"] = df.loc[censored_mask].apply(_censored_elapsed, axis=1)

    make_figure(df, epsilon, figures_dir / "sweep_time_vs_size.png")


if __name__ == "__main__":
    main()
