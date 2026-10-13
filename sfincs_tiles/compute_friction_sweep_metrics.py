"""Friction-sweep scores of the final SFINCS calibration: eikonal vs SFINCS
per (tile, friction scale factor) and pooled per friction scale factor, read
from every tile's outputs/calibration_result.json (calibration_results.py).

Writes {base}/friction_sweep_per_tile_metrics.csv and
{base}/friction_sweep_pooled_metrics.csv and prints the best point by CSI and
by |bias - 1|.

Usage:
    python compute_friction_sweep_metrics.py --base-dir-name calibration_roughness_SFINCS [--tile-ids-file <ids>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_results import load_results, sweep_tables  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--tile-ids-file", default=None, help="restrict to these tiles (default: every tile with a result)")
    args = parser.parse_args()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    ids = Path(args.tile_ids_file).read_text().split() if args.tile_ids_file else None
    results = load_results(base_dir, ids)
    n_listed = len((base_dir / "tile_ids.txt").read_text().split()) if (base_dir / "tile_ids.txt").exists() else None
    print(f"{len(results)} tile(s) with a calibration_result.json"
          + (f" (of {n_listed} in tile_ids.txt)" if n_listed else ""), flush=True)
    if not results:
        raise SystemExit("nothing to score yet")

    per_tile, pooled = sweep_tables(results)
    retry_transient_io(per_tile.to_csv, base_dir / "friction_sweep_per_tile_metrics.csv", index=False)
    retry_transient_io(pooled.to_csv, base_dir / "friction_sweep_pooled_metrics.csv", index=False)
    for r in pooled.itertuples():
        print(f"friction_scale_factor={r.friction_scale_factor:g}: n={r.n_tiles_scored}, CSI={r.CSI:.4f}, "
              f"bias={r.bias:.4f}, HT={r.HT:.4f}, FAR={r.FAR:.4f}")
    print(f"\nWrote {base_dir / 'friction_sweep_per_tile_metrics.csv'} ({len(per_tile)} row(s)) and "
          f"friction_sweep_pooled_metrics.csv")
    best_csi = pooled.loc[pooled["CSI"].idxmax()]
    best_bias = pooled.loc[(pooled["bias"] - 1.0).abs().idxmin()]
    print(f"Best CSI: friction_scale_factor={best_csi['friction_scale_factor']:g}, CSI={best_csi['CSI']:.4f}, "
          f"bias={best_csi['bias']:.4f}")
    print(f"Best |bias-1|: friction_scale_factor={best_bias['friction_scale_factor']:g}, CSI={best_bias['CSI']:.4f}, "
          f"bias={best_bias['bias']:.4f}")


if __name__ == "__main__":
    main()
