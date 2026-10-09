"""Bangkok / Chao Phraya case study: exposed-population-vs-return-period
curve, one line per SLR scenario. Reads
analysis/compute_bangkok_case_exposure.py's own output CSV - no raster I/O
here, this is a pure plotting script.

Usage:
    python plot_bangkok_exposure_vs_rp.py \\
        --exposure-csv P:/.../bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv \\
        --outdir P:/.../bangkok_chao_phraya/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exposure-csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.exposure_csv)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    slr_order = sorted(df["waterlevel_name"].unique(), key=lambda s: int(s.split("_")[1]))
    cmap = plt.get_cmap("Blues")
    colors = {slr: cmap(0.35 + 0.6 * i / max(1, len(slr_order) - 1)) for i, slr in enumerate(slr_order)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for slr in slr_order:
        sub = df[df["waterlevel_name"] == slr].sort_values("return_period")
        ax.plot(sub["return_period"], sub["exposed_population"], marker="o", markersize=4,
                color=colors[slr], linewidth=2, label=slr)

    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Exposed population (depth >= threshold)")
    ax.set_title("Bangkok / Chao Phraya case study: exposed population vs. return period")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="SLR scenario", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "bangkok_exposure_vs_rp.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
