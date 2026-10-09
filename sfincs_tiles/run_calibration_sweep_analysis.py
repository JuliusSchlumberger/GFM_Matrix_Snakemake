"""THE single entry point for the sfincs_calibration friction-sweep
analysis/postprocessing pipeline - one command, five steps, each a real
subprocess call to an existing, independently-tested script (nothing
reimplemented here, stdout/stderr stream straight through so you see the
same live progress running any of them standalone would give you):

  0. report_calibration_tile_status.py - which tiles are fully done, which
     have no COAST-RP station (can never complete), which need a full
     rebuild vs. just the sweep, and writes missing_eikonal_pairs.csv (the
     direct resubmission target for the fixable ones). Informational only
     - never blocks the steps below, which just work with whatever sweep
     data already exists on disk.
  1. compute_friction_sweep_metrics.py - pooled CSI/bias per
     friction_scale_factor against SFINCS.
  2. Pick the sweep point with the BEST CSI (or --select-by bias).
  3. plot_worst_tiles_panel.py + plot_validation_results.py, both pointed
     at that winning friction_scale_factor - worst-tile panels AND the full
     validation_sfincs_v5-style figure set (extent scatter, agreement
     histogram, depth scatter, agreement map, agreement-vs-tile-size,
     depth-joint correlation/category-alignment heatmaps) all for the one
     friction factor that actually performs best, not the production
     default.
  5. plot_sfincs_tile_diagnostics.py --figures coastrp_forcing for every
     tile in those two worst-tile panels (plot_worst_tiles_panel.py's
     worst_tiles_selection_fsf{X}.csv) - written to figures/ as
     {overprediction,underprediction}_{tile_id}_forcing_comparison.png
     (eikonal over-/under-predicting relative to SFINCS).

Usage:
    python run_calibration_sweep_analysis.py --base-dir-name sfincs_calibration --max-outer-iterations 5
    # score only a specific tile subset instead of the study's full tile_ids.txt:
    python run_calibration_sweep_analysis.py --base-dir-name sfincs_calibration \\
        --tile-ids-file sfincs_calibration/tile_ids_done.txt --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
from gfm_config import read_root  # noqa: E402

FRICTION_SCALE_FACTORS_DEFAULT = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]

# worst-tile panel -> figure-name prefix, from the eikonal's point of view
_EIKONAL_PREDICTION = {"eikonal_overpredicts": "overprediction", "sfincs_overpredicts": "underprediction"}


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--tile-ids-file", default=None,
                         help="one tile_id per line, passed to compute_friction_sweep_metrics.py - "
                              "default: {base-dir-name}/tile_ids.txt (the study's own full tile list; "
                              "tiles with no data yet are skipped automatically)")
    parser.add_argument("--friction-scale-factors", type=float, nargs="+", default=FRICTION_SCALE_FACTORS_DEFAULT)
    parser.add_argument("--max-outer-iterations", type=int, default=4,
                         help="must match whatever the sweep was actually run with (sfincs_calibration used 5)")
    parser.add_argument("--n-tiles", type=int, default=9, help="worst-tile panel size, forwarded to plot_worst_tiles_panel.py")
    parser.add_argument("--min-union-km2", type=float, default=1.0, help="forwarded to plot_worst_tiles_panel.py")
    parser.add_argument(
        "--select-by", choices=["CSI", "bias"], default="CSI",
        help="'CSI': highest pooled CSI (default). 'bias': |bias-1| closest to 0 (least over/under-prediction overall).",
    )
    args = parser.parse_args()

    python = sys.executable
    root = read_root(Path(args.config))
    tile_ids_file = args.tile_ids_file or str(root / args.base_dir_name / "tile_ids.txt")

    # Step 0: tile completion status (informational - writes calibration_tile_status.csv
    # + missing_eikonal_pairs.csv, never blocks the steps below).
    _run([
        python, str(_THIS_DIR / "report_calibration_tile_status.py"),
        "--config", args.config,
        "--base-dir-name", args.base_dir_name,
        "--friction-scale-factors", *[str(f) for f in args.friction_scale_factors],
        "--max-outer-iterations", str(args.max_outer_iterations),
    ])

    # Step 1: score the sweep (writes {base_dir}/friction_sweep_{per_tile,pooled}_metrics.csv)
    _run([
        python, str(_THIS_DIR / "compute_friction_sweep_metrics.py"),
        "--config", args.config,
        "--base-dir-name", args.base_dir_name,
        "--tile-ids-file", tile_ids_file,
        "--friction-scale-factors", *[str(f) for f in args.friction_scale_factors],
        "--max-outer-iterations", str(args.max_outer_iterations),
    ])

    # Step 2: pick the best sweep point from what step 1 just wrote.
    pooled_path = root / args.base_dir_name / "friction_sweep_pooled_metrics.csv"
    pooled = pd.read_csv(pooled_path)

    if args.select_by == "CSI":
        best = pooled.loc[pooled["CSI"].idxmax()]
    else:
        best = pooled.loc[(pooled["bias"] - 1.0).abs().idxmin()]
    best_fsf = float(best["friction_scale_factor"])
    print(f"\nBest sweep point by {args.select_by}: friction_scale_factor={best_fsf:g} "
          f"({best['fraction_of_current']:.1f}x current), CSI={best['CSI']:.4f}, bias={best['bias']:.4f}, "
          f"n_tiles_scored={int(best['n_tiles_scored'])}")

    # Step 3: worst-tile panels for that best point.
    _run([
        python, str(_THIS_DIR / "plot_worst_tiles_panel.py"),
        "--base-dir-name", args.base_dir_name,
        "--friction-scale-factor", str(best_fsf),
        "--max-outer-iterations", str(args.max_outer_iterations),
        "--n-tiles", str(args.n_tiles),
        "--min-union-km2", str(args.min_union_km2),
    ])

    # Step 4: full validation figure set (extent scatter, agreement histogram, depth
    # scatter, agreement map, agreement-vs-tile-size) for that same best point.
    _run([
        python, str(_THIS_DIR / "plot_validation_results.py"),
        "--base-dir-name", args.base_dir_name,
        "--friction-scale-factor", str(best_fsf),
        "--max-outer-iterations", str(args.max_outer_iterations),
    ])

    # Step 5: COAST-RP forcing figure for every tile in the two worst-tile panels
    # (the selection step 3 wrote). One subprocess per tile; a failing tile is
    # reported at the end instead of aborting the rest.
    fig_dir = root / args.base_dir_name / "figures"
    selection = pd.read_csv(fig_dir / f"worst_tiles_selection_fsf{best_fsf:g}.csv")
    failed = []
    for row in selection.itertuples():
        cmd = [
            python, str(_THIS_DIR / "plot_sfincs_tile_diagnostics.py"),
            "--config", args.config,
            "--tile-id", str(row.tile_id),
            "--base-dir-name", args.base_dir_name,
            "--figures", "coastrp_forcing",
            "--fig-dir", str(fig_dir),
            "--out-name", f"{_EIKONAL_PREDICTION[row.panel]}_{row.tile_id}_forcing_comparison.png",
        ]
        print(f"\n$ {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failed.append(f"{row.panel}/{row.tile_id}")
    if failed:
        print(f"\nCOAST-RP forcing figure FAILED for {len(failed)} tile(s): {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
