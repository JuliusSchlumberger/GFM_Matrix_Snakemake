"""Bar charts comparing EAI across adaptation scenarios at each SLR level.

Reads File 1 (exposure_{label}_base.csv — 2020-population EAI at each
discrete modelled SLR scenario, plain SLR_X-named columns) from
compute_exposure_analysis.py and plots grouped bar charts comparing baseline
vs protect vs retreat at each SLR scenario — globally and per country. Avoid
has no File 1 (its "redirected growth" term is inherently growth-dependent,
so there's no growth-free base to bar-chart) and is not included here.

Usage:
    python snakemake_workflow/analysis/plot_adaptation_bars.py \\
        --config  snakemake_workflow/config/config.yml \\
        --expdir  D:/GFM/merged_results/exposure \\
        [--outdir D:/GFM/figures/adaptation_bars]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import merged_slr_scenarios  # noqa: E402
from visualization import plot_adaptation_bars  # noqa: E402


def _expand(s, root, code_root=""):
    return str(s).replace("{root}", root).replace("{code_root}", code_root or root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    root = cfg.get("paths", {}).get("root", "")
    code_root = cfg.get("paths", {}).get("code_root", root)
    ex = lambda s: _expand(s, root, code_root)

    viz = cfg.get("visualization", {})
    bc = cfg["boundary_conditions"]
    adapt_cfg = cfg.get("adaptation", {})

    slr_scenarios = merged_slr_scenarios(bc, adapt_cfg)
    slr_mm = [float(s.split("_")[1]) for s in slr_scenarios]
    slr_intensities = adapt_cfg.get("slr_intensities", [])

    exp_dir = Path(ex(args.expdir))
    out_dir = Path(ex(args.outdir or str(
        Path(ex(viz.get("output_dir", "{root}/figures"))) / "adaptation_bars"
    )))
    out_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(viz.get("dpi", 200))

    # Load File 1 (base EAI, plain SLR_X-named columns) for every scenario
    # that has one — baseline + protect/retreat, not avoid.
    eai_dict: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(exp_dir.glob("exposure_*_base.csv")):
        label = csv_path.stem.replace("exposure_", "").replace("_base", "")
        eai_dict[label] = pd.read_csv(csv_path, index_col=0)  # index=ISO, cols=SLR names

    if not eai_dict:
        print(f"No exposure_*_base.csv files found in {exp_dir}")
        sys.exit(0)

    def _save(sub_dict: dict[str, pd.DataFrame], label: str, fname: Path) -> None:
        # sum over countries/geogunits to get aggregate totals per scenario per SLR
        summed = {k: v.sum(axis=0) for k, v in sub_dict.items()}
        fig = plot_adaptation_bars(
            {k: pd.DataFrame([s.to_dict()]) for k, s in summed.items()},
            slr_scenarios, slr_mm,
            entity_label=label,
            measure_colors=viz.get("measure_colors"),
            dpi=dpi,
        )
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    # Global
    _save(eai_dict, "Global", out_dir / "adaptation_bars_global.png")
    print("Global done.")

    # Per country
    all_isos = next(iter(eai_dict.values())).index.tolist()
    country_dir = out_dir / "country"
    country_dir.mkdir(exist_ok=True)
    for iso in sorted(all_isos):
        sub = {k: v.loc[[iso]] for k, v in eai_dict.items() if iso in v.index}
        if not sub:
            continue
        _save(sub, iso, country_dir / f"adaptation_bars_{iso}.png")
    print(f"Per-country done → {country_dir}")

    print(f"\nAll bar charts written to: {out_dir}")


if __name__ == "__main__":
    main()
