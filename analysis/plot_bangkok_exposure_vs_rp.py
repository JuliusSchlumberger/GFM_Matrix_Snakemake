"""Plot Bangkok / Chao Phraya exposed population vs return period per SLR scenario.

Reads compute_bangkok_case_exposure.py's CSV.

Usage:
    python plot_bangkok_exposure_vs_rp.py \\
        --exposure-csv P:/.../bangkok_chao_phraya/merged_results/exposure/bangkok_case_exposure.csv \\
        --out-dir P:/.../bangkok_chao_phraya/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import return_period_years  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exposure-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.exposure_csv)
    df["return_period"] = df["return_period"].map(return_period_years)  # "RP100" -> 100
    out_dir = Path(args.out_dir)
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
