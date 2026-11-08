"""Run the SFINCS calibration analysis on whatever tiles have a calibration_result.json.

Steps (subprocesses): 1 compute_variant_metrics.py (run tables, variant_selection.json);
2 the best variant's run by pooled CSI (or --select-by bias); 3 plot_worst_tiles_panel.py; 4
plot_validation_results.py; 5 plot_sfincs_tile_diagnostics.py forcing figures for
the worst tiles (or all, --forcing-checks-all) into figures/forcing_checks/.

Usage:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS --forcing-checks-only
    # forcing figures for every scored tile instead of only the worst ones:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS --forcing-checks-all
    # the overlap-corrected (dedup) results - maps and inputs read from the original study:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS_dedup \\
        --raster-base-dir-name calibration_roughness_SFINCS
    # a tile subset instead of every tile with a result:
    python run_calibration_sweep_analysis.py --base-dir-name calibration_roughness_SFINCS --tile-ids-file <ids.txt>
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent

sys.path.insert(0, str(_THIS_DIR))
from calibration_paths import SELECTION_NAME  # noqa: E402
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
    parser.add_argument("--n-panel-tiles", type=int, default=6, help="worst-tile panel size, forwarded to plot_worst_tiles_panel.py")
    parser.add_argument("--min-union-km2", type=float, default=50.0, help="forwarded to plot_worst_tiles_panel.py")
    parser.add_argument(
        "--select-by", choices=["CSI", "bias"], default="CSI",
        help="'CSI': highest pooled CSI (default). 'bias': |bias-1| closest to 0 (least over/under-prediction overall).",
    )
    parser.add_argument("--forcing-checks-workers", type=int, default=4,
                        help="step 5: tiles drawn in parallel (one subprocess each)")
    parser.add_argument("--forcing-checks-all", action="store_true",
                        help="step 5 for every scored tile (default: only the tiles of the worst-tile panels, "
                             "worst_tiles_selection_fsf<X>.csv from step 3)")
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
        _forcing_checks(python, args, root, _best_run(root, args))
        return

    _run([python, str(_THIS_DIR / "compute_variant_metrics.py"), *common, *subset, "--select-by", args.select_by])
    best = _best_run(root, args)
    _run([python, str(_THIS_DIR / "plot_worst_tiles_panel.py"), *common, *raster,
          "--run-key", best, "--n-panel-tiles", str(args.n_panel_tiles),
          "--min-union-km2", str(args.min_union_km2)])
    _run([python, str(_THIS_DIR / "plot_validation_results.py"), *common, *subset, "--run-key", best])
    _forcing_checks(python, args, root, best)


def _best_run(root: Path, args) -> str:
    selection = json.loads((root / args.base_dir_name / SELECTION_NAME).read_text())
    variant = selection["best_variant"]
    e = selection["variants"][variant]
    print(f"\nBest variant by {selection['select_by']} on the calibration group: {variant}, run {e['run_key']}, "
          f"CSI={e['calibration']['CSI']:.4f}, bias={e['calibration']['bias']:.4f}")
    return e["run_key"]


def _model_prediction(row) -> str:
    if row.model_only_km2 > row.sfincs_only_km2:
        return "overprediction"
    if row.sfincs_only_km2 > row.model_only_km2:
        return "underprediction"
    return "agreement"


def _forcing_checks(python: str, args, root: Path, key: str) -> None:
    # A failing tile is reported at the end (exit 1) instead of aborting the rest.
    base = root / args.base_dir_name
    out_dir = base / "figures" / "forcing_checks"
    per_tile = pd.read_csv(base / "variant_runs_per_tile.csv")
    rows = per_tile[per_tile["run_key"] == key].sort_values("tile_id")
    if args.tile_ids_file:
        rows = rows[rows["tile_id"].astype(str).isin(Path(args.tile_ids_file).read_text().split())]
    if not args.forcing_checks_all:
        sel_path = base / "figures" / f"worst_tiles_selection_{key}.csv"
        if not sel_path.exists():
            raise SystemExit(f"{sel_path} missing - run the full analysis (step 3) first, or use --forcing-checks-all")
        rows = rows[rows["tile_id"].isin(pd.read_csv(sel_path)["tile_id"])]
    rows = list(rows.itertuples())
    figures = ["coastrp_forcing", "boundary_forcing"]

    def run_one(row) -> tuple[str, bool]:
        prefix = f"{_model_prediction(row)}_{int(row.tile_id)}"
        cmd = [python, str(_THIS_DIR / "plot_sfincs_tile_diagnostics.py"),
               "--config", args.config, "--tile-id", str(int(row.tile_id)),
               "--base-dir-name", args.raster_base_dir_name or args.base_dir_name,
               "--figures", *figures, "--fig-dir", str(out_dir), "--out-prefix", prefix, "--run-key", key]
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
