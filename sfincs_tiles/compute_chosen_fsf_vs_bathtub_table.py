"""Pooled 2-row comparison table: HC-bathtub vs eikonal at production's
chosen friction_scale_factor, for a friction-sweep calibration study
(sfincs_calibration by default).

Why this needs its own script rather than just filtering
friction_sweep_pooled_metrics.csv / metrics_overview_table.csv down to one
row each and pasting them together: there is no summary_eikonal.json for a
non-default-tagged fsf point - postprocess_tile_summary.py only ever writes
the untagged DEFAULT point's own eikonal stats (run_eikonal_on_sfincs_
subgrid.py's FRICTION_SCALE_FACTOR_DEFAULT, 30.0, frozen for sweep-file-
identity reasons - see that constant's own comment - independent of
production's actual friction_scale_factor, now 9.0). So this pools the
chosen point's depth-agreement stats directly from each tile's own
sweep_comparison_cache.json (tile_sweep_cache.py) instead, re-implementing
flood_agreement.pool_depth_joint's exact sufficient-stat summation against
that source. Extent metrics (HT/FAR/CSI/bias) are summed the same way,
straight from each tile's cache - not read back from friction_sweep_
pooled_metrics.csv, so this script's own result does not depend on that
file being up to date.

HC-bathtub's row is unchanged from compute_metrics_overview_table.py's own
pooled_row() (summary_bathtub.json, no raster I/O, reused directly).

Loads each tile's chosen-fsf cache entry via tile_sweep_cache.load_or_build_
cache(), which (2026-10-08 fix) rebuilds live for any tile whose on-disk
cache doesn't yet cover the requested point - cheap here since only ONE fsf
point is requested, unlike compute_friction_sweep_metrics.py's own full
10-point sweep (which, for this same reason, needed a much larger live
rebuild after the same bug - see tile_sweep_cache.py's own comment on
load_or_build_cache for the root cause).

Usage:
    python compute_chosen_fsf_vs_bathtub_table.py --base-dir-name sfincs_calibration
    python compute_chosen_fsf_vs_bathtub_table.py --base-dir-name sfincs_calibration \\
        --chosen-friction-scale-factor 9 --max-outer-iterations 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_friction_sweep_metrics import MAX_OUTER_ITERATIONS_DEFAULT  # noqa: E402
from compute_metrics_overview_table import collect_rows, pooled_row  # noqa: E402
from flood_agreement import depth_error_metrics_from_pooled, metrics_from_counts  # noqa: E402
from gfm_config import read_root  # noqa: E402
from tile_sweep_cache import load_or_build_cache  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402


def pool_chosen_fsf(base_dir: Path, tile_ids: list[str], chosen_fsf: float, max_outer_iterations: int) -> dict:
    """Same pooling flood_agreement.pool_depth_joint/metrics_from_counts do
    for a summary_{model}.json source, but reading each tile's chosen-fsf
    point straight from its own sweep_comparison_cache.json instead (see
    module docstring for why there is no summary json for this)."""
    matched = model_only = sfincs_only = 0.0
    n = sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
    hist_fine = hist_category = None
    n_tiles_pooled = 0

    for tid in tile_ids:
        tile_dir = base_dir / tid
        cache = load_or_build_cache(tile_dir, [chosen_fsf], max_outer_iterations)
        if cache is None:
            continue
        point = cache["points"].get(f"{chosen_fsf:g}")
        if point is None:
            continue
        n_tiles_pooled += 1
        matched += point["matched_km2"]
        model_only += point["model_only_km2"]
        sfincs_only += point["sfincs_only_km2"]

        joint = point.get("depth_joint")
        if not joint:
            continue
        n += joint["n"]
        sum_x += joint["sum_x"]
        sum_y += joint["sum_y"]
        sum_x2 += joint["sum_x2"]
        sum_y2 += joint["sum_y2"]
        sum_xy += joint["sum_xy"]
        hf = np.asarray(joint["hist_fine"])
        hc = np.asarray(joint["hist_category"])
        hist_fine = hf if hist_fine is None else hist_fine + hf
        hist_category = hc if hist_category is None else hist_category + hc

    pooled_joint = {
        "n": n, "sum_x": sum_x, "sum_y": sum_y, "sum_x2": sum_x2, "sum_y2": sum_y2, "sum_xy": sum_xy,
        "hist_fine": hist_fine, "hist_category": hist_category,
    }
    return {
        "model": f"Eikonal (friction_scale_factor={chosen_fsf:g})",
        "n_tiles_pooled": n_tiles_pooled,
        "total_km2": float("nan"),  # not tracked per-point in the sweep cache - HC-bathtub's own total_km2
        # below is cell_km2 summed over the full tile grid regardless of wet/dry, not a wet-cell count,
        # so there is no equivalent single number to compute from matched/only/sfincs_only alone here.
        "matched_km2": float(matched), "model_only_km2": float(model_only), "sfincs_only_km2": float(sfincs_only),
        **metrics_from_counts(matched, model_only, sfincs_only),
        **depth_error_metrics_from_pooled(pooled_joint),
    }


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="sfincs_calibration")
    parser.add_argument(
        "--chosen-friction-scale-factor", type=float, default=None,
        help="default: simulation.flooding.friction_scale_factor from --config (production's own value)",
    )
    parser.add_argument("--max-outer-iterations", type=int, default=MAX_OUTER_ITERATIONS_DEFAULT)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chosen_fsf = args.chosen_friction_scale_factor
    if chosen_fsf is None:
        chosen_fsf = float(cfg["simulation"]["flooding"]["friction_scale_factor"])

    root = read_root(Path(args.config))
    base_dir = root / args.base_dir_name
    tile_ids = [line.strip() for line in (base_dir / "tile_ids.txt").read_text().splitlines() if line.strip()]
    print(f"{len(tile_ids)} tile(s) from tile_ids.txt - chosen friction_scale_factor={chosen_fsf:g}")

    bathtub_df = collect_rows(base_dir, tile_ids)
    bathtub_row = pooled_row(bathtub_df, base_dir, "bathtub")

    print("Pooling chosen-fsf eikonal point from each tile's sweep_comparison_cache.json "
          "(rebuilding live for any tile missing it)...")
    eikonal_row = pool_chosen_fsf(base_dir, tile_ids, chosen_fsf, args.max_outer_iterations)

    table = pd.DataFrame([bathtub_row, eikonal_row])
    out_path = base_dir / "chosen_fsf_vs_bathtub_table.csv"
    table.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    display_cols = [
        "model", "n_tiles_pooled", "matched_km2", "model_only_km2", "sfincs_only_km2",
        "HT", "FAR", "CSI", "bias", "r", "bias_m", "rmse_m", "median_error_m", "pct_within_0.2m",
    ]
    rounded = table[display_cols].copy()
    for c in rounded.columns:
        if c != "model":
            rounded[c] = rounded[c].astype(float).round(2)
    with pd.option_context("display.width", 180):
        print(rounded.to_string(index=False))


if __name__ == "__main__":
    main()
