"""Single entry point for the full post-simulation analysis pipeline.

Reads config.yml to decide which steps to run (analysis.* switches) and
calls each downstream script in sequence.  Each step runs in its own Python
subprocess so modules are properly isolated and crashes in one step do not
abort subsequent steps (unless --fail-fast is set).

Analysis switches in config.yml:
  analysis.compute_exposure     — baseline / protect / retreat / avoid EAI CSVs
  analysis.plot_burning_ember   — impact-matrix (SLR × growth) figures
  analysis.plot_adaptation_bars — grouped bar comparison per scenario
  analysis.plot_timeseries      — EAI along SSP SLR trajectories over time
  analysis.plot_world_maps      — world choropleth maps (slow for global runs)

Usage:
    python snakemake_workflow/analysis/run_analysis.py \\
        [--config  snakemake_workflow/config/config.yml] \\
        [--expdir  D:/GFM/merged_results/exposure] \\
        [--figdir  D:/GFM/figures] \\
        [--fail-fast]

Override switches without editing config.yml by passing --skip-* / --only-*:
    --only-exposure          run only the exposure calculation
    --only-plots             run only the plots (all enabled plot types)
    --skip-world-maps        skip world maps even if enabled in config
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _expand(s: str, root: str, code_root: str = "") -> str:
    return str(s).replace("{root}", root).replace("{code_root}", code_root or root)


def _run(
    script: Path,
    extra_args: list[str],
    label: str,
    fail_fast: bool,
) -> bool:
    """Run `script` in a subprocess; return True on success."""
    cmd = [PYTHON, str(script)] + extra_args
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'═' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s — {label}")
        if fail_fast:
            print("  Aborting (--fail-fast).")
            sys.exit(result.returncode)
        return False
    print(f"\n  ✓ Done in {elapsed:.0f}s — {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(SCRIPTS_DIR.parent / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg,
                        help=f"path to config.yml (default: {_default_cfg})")
    parser.add_argument("--expdir", default=None,
                        help="exposure CSV directory (default: merged_outputs/exposure)")
    parser.add_argument("--figdir", default=None,
                        help="figures root directory (default: visualization.output_dir)")
    parser.add_argument("--fail-fast", action="store_true",
                        help="abort on first failed step")
    # Convenience overrides
    parser.add_argument("--only-exposure",    action="store_true")
    parser.add_argument("--only-plots",       action="store_true")
    parser.add_argument("--skip-exposure",    action="store_true")
    parser.add_argument("--skip-world-maps",  action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    root      = cfg.get("paths", {}).get("root", "")
    code_root = cfg.get("paths", {}).get("code_root", root)
    ex = lambda s: _expand(s, root, code_root)

    viz        = cfg.get("visualization", {})
    merged_out = ex(cfg["postprocessing"]["merged_outputs"])

    exp_dir = Path(args.expdir or f"{merged_out}/exposure")
    fig_dir = Path(args.figdir or ex(viz.get("output_dir", "{root}/figures")))

    # ── Resolve which steps to run from config + CLI overrides ────────────────
    sw = cfg.get("analysis", {})

    do_exposure    = sw.get("compute_exposure",     True)
    do_ember       = sw.get("plot_burning_ember",   True)
    do_bars        = sw.get("plot_adaptation_bars", True)
    do_timeseries  = sw.get("plot_timeseries",      True)
    do_world_maps  = sw.get("plot_world_maps",      False)

    if args.only_exposure:
        do_ember = do_bars = do_timeseries = do_world_maps = False
    if args.only_plots:
        do_exposure = False
    if args.skip_exposure:
        do_exposure = False
    if args.skip_world_maps:
        do_world_maps = False

    cfg_arg = ["--config", str(config_path)]

    results: dict[str, bool] = {}
    t_start = time.time()

    # ── Step 1: Compute exposure (baseline / protect / retreat / avoid) ────────
    if do_exposure:
        success = _run(
            SCRIPTS_DIR / "compute_exposure_analysis.py",
            cfg_arg + ["--outdir", str(exp_dir)],
            "Exposure analysis (baseline / protect / retreat / avoid)",
            args.fail_fast,
        )
        results["exposure"] = success
    else:
        print("\n  [ SKIP ] Exposure analysis (analysis.compute_exposure = false)")

    # ── Step 2: Burning-ember charts ──────────────────────────────────────────
    if do_ember:
        success = _run(
            SCRIPTS_DIR / "plot_burning_ember.py",
            cfg_arg + ["--expdir", str(exp_dir),
                        "--outdir", str(fig_dir / "burning_ember")],
            "Burning-ember / impact-matrix figures",
            args.fail_fast,
        )
        results["burning_ember"] = success
    else:
        print("\n  [ SKIP ] Burning-ember plots (analysis.plot_burning_ember = false)")

    # ── Step 3: Adaptation bar charts ────────────────────────────────────────
    if do_bars:
        success = _run(
            SCRIPTS_DIR / "plot_adaptation_bars.py",
            cfg_arg + ["--expdir", str(exp_dir),
                        "--outdir", str(fig_dir / "adaptation_bars")],
            "Adaptation strategy bar charts",
            args.fail_fast,
        )
        results["adaptation_bars"] = success
    else:
        print("\n  [ SKIP ] Adaptation bar charts (analysis.plot_adaptation_bars = false)")

    # ── Step 4: Time-series ───────────────────────────────────────────────────
    if do_timeseries:
        success = _run(
            SCRIPTS_DIR / "plot_timeseries.py",
            cfg_arg + ["--expdir", str(exp_dir),
                        "--outdir", str(fig_dir / "timeseries")],
            "EAI time-series along SSP trajectories",
            args.fail_fast,
        )
        results["timeseries"] = success
    else:
        print("\n  [ SKIP ] Time-series plots (analysis.plot_timeseries = false)")

    # ── Step 5: World maps ────────────────────────────────────────────────────
    if do_world_maps:
        success = _run(
            SCRIPTS_DIR / "plot_world_map.py",
            cfg_arg + ["--expdir", str(exp_dir),
                        "--outdir", str(fig_dir / "world_maps")],
            "World choropleth maps",
            args.fail_fast,
        )
        results["world_maps"] = success
    else:
        print("\n  [ SKIP ] World maps (analysis.plot_world_maps = false)")

    # ── Summary ───────────────────────────────────────────────────────────────
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
