"""THE single entry point for the final SFINCS calibration analysis
(calibration_roughness_SFINCS) - one command, each step a real subprocess
call to an independently runnable script, all reading the per-tile
outputs/calibration_result.json files (compute_calibration_result.py), so it
can be rerun at any time on whatever tiles have finished:

  1. compute_friction_sweep_metrics.py - eikonal vs SFINCS per tile and
     pooled per friction scale factor (friction_sweep_{per_tile,pooled}_metrics.csv).
  2. Pick the friction scale factor with the best pooled CSI (or --select-by bias).
  3. plot_worst_tiles_panel.py - worst-tile panels at that point.
  4. plot_validation_results.py - extent/depth/agreement figure set and
     metrics_overview_table_fsf<X>.csv (HC-bathtub and EA-bathtub vs SFINCS) at that point.
  5. plot_sfincs_tile_diagnostics.py --figures coastrp_forcing boundary_forcing
     for EVERY tile scored at that point, --forcing-checks-workers at a time,
     into figures/forcing_checks/ as
     {overprediction,underprediction,agreement}_{tile_id}_forcing_comparison.png and
     ..._boundary_forcing.png (eikonal over-/under-predicting relative to SFINCS).
     --forcing-checks-only redoes just this step from the existing metrics.

Usage:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS --forcing-checks-only
    # the overlap-corrected (dedup) results - maps and inputs read from the original study:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS_dedup \\
        --raster-base-dir-name calibration_roughness_SFINCS
    # a tile subset instead of every tile with a result:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS --tile-ids-file <ids.txt>
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
from gfm_config import read_root  # noqa: E402


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--tile-ids-file", default=None,
                        help="restrict every step to these tiles (default: every tile with a calibration_result.json)")
    parser.add_argument("--n-tiles", type=int, default=6, help="worst-tile panel size, forwarded to plot_worst_tiles_panel.py")
    parser.add_argument("--min-union-km2", type=float, default=50.0, help="forwarded to plot_worst_tiles_panel.py")
    parser.add_argument(
        "--select-by", choices=["CSI", "bias"], default="CSI",
        help="'CSI': highest pooled CSI (default). 'bias': |bias-1| closest to 0 (least over/under-prediction overall).",
    )
    parser.add_argument("--forcing-checks-workers", type=int, default=4,
                        help="step 5: tiles drawn in parallel (one subprocess each)")
    parser.add_argument("--forcing-checks-worst-only", action="store_true",
                        help="step 5 only for the tiles of the worst-tile panels (worst_tiles_selection_fsf<X>.csv)")
    parser.add_argument("--forcing-checks-only", action="store_true",
                        help="only step 5 (forcing figures), from the existing pooled + per-tile metrics")
    parser.add_argument("--raster-base-dir-name", default=None,
                        help="base dir holding the per-tile rasters/inputs when --base-dir-name only holds result "
                             "JSONs (calibration_roughness_SFINCS_dedup -> calibration_roughness_SFINCS): steps 3 "
                             "and 5 read the maps from there, figures still go to --base-dir-name")
    args = parser.parse_args()

    python = sys.executable
    root = read_root(Path(args.config))
    common = ["--config", args.config, "--base-dir-name", args.base_dir_name]
    subset = ["--tile-ids-file", args.tile_ids_file] if args.tile_ids_file else []
    raster = ["--raster-base-dir-name", args.raster_base_dir_name] if args.raster_base_dir_name else []
    if args.raster_base_dir_name:  # step 4 needs the tiles' lon/lat from the selection metadata
        meta = root / args.base_dir_name / "tile_selection_metadata.csv"
        if not meta.exists():
            shutil.copyfile(root / args.raster_base_dir_name / "tile_selection_metadata.csv", meta)
            print(f"copied tile_selection_metadata.csv from {args.raster_base_dir_name}")
    if args.forcing_checks_only:
        _forcing_checks(python, args, root, _best_fsf(root, args))
        return

    _run([python, str(_THIS_DIR / "compute_friction_sweep_metrics.py"), *common, *subset])
    best_fsf = _best_fsf(root, args)
    _run([python, str(_THIS_DIR / "plot_worst_tiles_panel.py"), *common, *raster,
          "--friction-scale-factor", f"{best_fsf:g}", "--n-tiles", str(args.n_tiles),
          "--min-union-km2", str(args.min_union_km2)])
    _run([python, str(_THIS_DIR / "plot_validation_results.py"), *common, *subset,
          "--friction-scale-factor", f"{best_fsf:g}"])
    _forcing_checks(python, args, root, best_fsf)


def _best_fsf(root: Path, args) -> float:
    pooled = pd.read_csv(root / args.base_dir_name / "friction_sweep_pooled_metrics.csv")
    if args.select_by == "CSI":
        best = pooled.loc[pooled["CSI"].idxmax()]
    else:
        best = pooled.loc[(pooled["bias"] - 1.0).abs().idxmin()]
    best_fsf = float(best["friction_scale_factor"])
    print(f"\nBest sweep point by {args.select_by}: friction_scale_factor={best_fsf:g}, CSI={best['CSI']:.4f}, "
          f"bias={best['bias']:.4f}, n_tiles_scored={int(best['n_tiles_scored'])}")
    return best_fsf


def _eikonal_prediction(row) -> str:
    """Figure-name prefix from the eikonal's point of view at the chosen sweep point."""
    if row.eikonal_only_km2 > row.sfincs_only_km2:
        return "overprediction"
    if row.sfincs_only_km2 > row.eikonal_only_km2:
        return "underprediction"
    return "agreement"


def _forcing_checks(python: str, args, root: Path, best_fsf: float) -> None:
    """COAST-RP forcing comparison + boundary forcing figure for EVERY tile scored at the chosen
    sweep point (friction_sweep_per_tile_metrics.csv) into
    figures/forcing_checks/, named {over,under}prediction_<tile>_... from the
    eikonal's point of view. --forcing-checks-workers tiles at a time, one
    subprocess each; a failing tile is reported at the end instead of aborting
    the rest."""
    base = root / args.base_dir_name
    out_dir = base / "figures" / "forcing_checks"
    per_tile = pd.read_csv(base / "friction_sweep_per_tile_metrics.csv")
    rows = per_tile[per_tile["friction_scale_factor"] == best_fsf].sort_values("tile_id")
    if args.tile_ids_file:
        rows = rows[rows["tile_id"].astype(str).isin(Path(args.tile_ids_file).read_text().split())]
    if args.forcing_checks_worst_only:  # only the tiles in the worst-tile panels (plot_worst_tiles_panel.py)
        sel = pd.read_csv(base / "figures" / f"worst_tiles_selection_fsf{best_fsf:g}.csv")
        rows = rows[rows["tile_id"].isin(sel["tile_id"])]
    rows = list(rows.itertuples())
    figures = ["coastrp_forcing", "boundary_forcing"]

    def run_one(row) -> tuple[str, bool]:
        prefix = f"{_eikonal_prediction(row)}_{int(row.tile_id)}"
        cmd = [python, str(_THIS_DIR / "plot_sfincs_tile_diagnostics.py"),
               "--config", args.config, "--tile-id", str(int(row.tile_id)),
               "--base-dir-name", args.raster_base_dir_name or args.base_dir_name,
               "--figures", *figures, "--fig-dir", str(out_dir), "--out-prefix", prefix]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"  tile {int(row.tile_id)} FAILED: {(res.stderr or '').strip().splitlines()[-1:]}", flush=True)
        return prefix, res.returncode == 0

    print(f"\nforcing figures ({', '.join(figures)}) for {len(rows)} tile(s) -> {out_dir}", flush=True)
    failed = []
    with ThreadPoolExecutor(args.forcing_checks_workers) as ex:
        for k, (prefix, ok) in enumerate(ex.map(run_one, rows), 1):
            if not ok:
                failed.append(prefix)
            if k % 25 == 0 or k == len(rows):
                print(f"  [{k}/{len(rows)}] done, {len(failed)} failed", flush=True)
    print(f"\nforcing figures for {len(rows) - len(failed)}/{len(rows)} tile(s) in {out_dir}")
    if failed:
        print(f"FAILED for {len(failed)} tile(s): {', '.join(failed[:50])}{' ...' if len(failed) > 50 else ''}")
        sys.exit(1)


if __name__ == "__main__":
    main()
