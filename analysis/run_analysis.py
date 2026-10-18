"""Run the post-simulation analysis pipeline steps enabled by config.yml analysis.*.

Each step runs in its own subprocess, so a crash does not abort later steps unless
--fail-fast. Switches: compute_exposure, compute_flood_totals (raw per-country totals for
flood_extent_validation), plot_burning_ember, plot_timeseries, plot_tipping_points,
plot_extended_pathways, plot_slr_vs_population, plot_world_maps (slow globally). All
figures go directly into --fig-dir.

Usage:
    python analysis/run_analysis.py \\
        [--config  snakemake_workflow/config/config.yml] \\
        [--expdir  D:/GFM/merged_results/exposure] \\
        [--fig-dir D:/GFM/figures] \\
        [--fail-fast] [--only-exposure | --only-plots] [--skip-world-maps]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402
from step_runner import run_step  # noqa: E402


SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(script: Path, extra_args: list[str], label: str, fail_fast: bool) -> bool:
    return run_step([PYTHON, str(script)] + extra_args, label, fail_fast)


# (results key, analysis.* switch, script, kind, title, skip label), in run
# order. "compute" gets --out-dir <expdir>; "plot" gets --expdir <expdir> --out-dir <fig-dir>.
_STEPS = [
    ("exposure", "compute_exposure", "compute_exposure_analysis.py", "compute",
     "Exposure analysis (baseline / protect / retreat / avoid / protect + avoid)", "Exposure analysis"),
    ("flood_totals", "compute_flood_totals", "compute_flood_totals.py", "compute",
     "Flood totals (per-country flooded area / exposed population per RP, SLR)", "Flood totals"),
    ("burning_ember", "plot_burning_ember", "plot_burning_ember.py", "plot",
     "Burning-ember / impact-matrix figures", "Burning-ember plots"),
    ("timeseries", "plot_timeseries", "plot_timeseries.py", "plot",
     "EAI time-series along SSP trajectories", "Time-series plots"),
    ("tipping_points", "plot_tipping_points", "plot_tipping_points.py", "plot",
     "Tipping-point figure (visualization.tipping_points)", "Tipping-point figure"),
    ("extended_pathways", "plot_extended_pathways", "plot_extended_pathways.py", "plot",
     "Pathways beyond 2100 (visualization.extended_pathways)", "Pathways beyond 2100"),
    ("slr_vs_population", "plot_slr_vs_population", "plot_slr_vs_population.py", "plot",
     "SLR vs population: waterfalls + scatter (visualization.slr_vs_population)", "SLR vs population"),
    ("world_maps", "plot_world_maps", "plot_world_map.py", "plot",
     "World choropleth maps", "World maps"),
]


def main() -> None:
    """Run the enabled analysis steps; exit 1 if any step failed."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(SCRIPTS_DIR.parent / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--expdir", default=None,
                        help="exposure CSV directory (default: merged_outputs/exposure)")
    parser.add_argument("--fig-dir", "--figdir", dest="figdir", default=None,
                        help="figures root directory (default: visualization.output_dir)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort on first failed step")
    parser.add_argument("--only-exposure",    action="store_true")
    parser.add_argument("--only-plots",       action="store_true")
    parser.add_argument("--skip-exposure",    action="store_true")
    parser.add_argument("--skip-flood-totals", action="store_true",
                        help="skip compute_flood_totals.py even if analysis.compute_flood_totals is true "
                             "(e.g. it already ran standalone - see analysis/compute_flood_totals.py's own "
                             "--workers for a much faster parallel run of just that step)")
    parser.add_argument("--skip-world-maps",  action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    viz        = cfg["visualization"]
    merged_out = cfg["postprocessing"]["merged_outputs"]

    exp_dir = Path(args.expdir or f"{merged_out}/exposure")
    fig_dir = Path(args.figdir or viz["output_dir"])

    sw = cfg["analysis"]
    enabled = {key: sw[switch] for key, switch, *_ in _STEPS}
    if args.only_exposure:  # flood totals keep their config switch
        for key, _sw, _script, kind, *_ in _STEPS:
            if kind == "plot":
                enabled[key] = False
    if args.only_plots or args.skip_exposure:
        enabled["exposure"] = False
    if args.skip_flood_totals:
        enabled["flood_totals"] = False
    if args.skip_world_maps:
        enabled["world_maps"] = False

    cfg_arg = ["--config", str(config_path)]

    results: dict[str, bool] = {}
    t_start = time.time()

    for key, switch, script, kind, title, skip_label in _STEPS:
        if not enabled[key]:
            print(f"\n  [ SKIP ] {skip_label} (analysis.{switch} = false)")
            continue
        io_args = ["--out-dir", str(exp_dir)] if kind == "compute" else ["--expdir", str(exp_dir), "--out-dir", str(fig_dir)]
        results[key] = _run(SCRIPTS_DIR / script, cfg_arg + io_args, title, args.fail_fast)

    total = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  Analysis pipeline complete  ({total / 60:.1f} min)")
    print(f"{'═' * 60}")
    for step, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {step}")
    if not all(results.values()):
        print("\n  Some steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
