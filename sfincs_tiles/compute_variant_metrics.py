"""Score every flood-model run vs SFINCS, pooled per group, and pick each variant's run.

Writes {base}/variant_runs_{per_tile,pooled}.csv and {base}/variant_selection.json: per
variant the calibration-group run with the best pooled CSI (or |bias - 1|), its validation
score, and the best variant. run_calibration_models.py reads the selection for its
duration and selected phases.

Usage:
    python compute_variant_metrics.py --base-dir-name calibration_roughness_SFINCS [--select-by CSI]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_paths import SELECTION_NAME  # noqa: E402
from calibration_results import load_results, run_tables, select_runs  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    parser.add_argument("--tile-ids-file", default=None, help="restrict to these tiles (default: every tile with a result)")
    parser.add_argument("--select-by", choices=["CSI", "bias"], default="CSI")
    args = parser.parse_args()

    base_dir = read_root(Path(args.config)) / args.base_dir_name
    ids = Path(args.tile_ids_file).read_text().split() if args.tile_ids_file else None
    results = load_results(base_dir, ids)
    print(f"{len(results)} tile(s) with a calibration_result.json", flush=True)
    if not results:
        raise SystemExit("nothing to score yet")

    per_tile, pooled = run_tables(results)
    retry_transient_io(per_tile.to_csv, base_dir / "variant_runs_per_tile.csv", index=False)
    retry_transient_io(pooled.to_csv, base_dir / "variant_runs_pooled.csv", index=False)
    cal = pooled[pooled.group == "calibration"].sort_values(["variant", "parameter", "alpha"])
    for r in cal.itertuples():
        print(f"{r.run_key}: n={r.n_tiles_scored}, CSI={r.CSI:.4f}, bias={r.bias:.4f}, HT={r.HT:.4f}, FAR={r.FAR:.4f}")

    selection = select_runs(pooled, args.select_by)
    retry_transient_io((base_dir / SELECTION_NAME).write_text, json.dumps(selection, indent=1), encoding="utf-8")
    print(f"\nSelected by {args.select_by} on the calibration group (validation in brackets):")
    for variant, e in selection["variants"].items():
        c, v = e["calibration"], e["validation"]
        val = f"CSI {v['CSI']:.4f}, bias {v['bias']:.4f}, n {v['n_tiles_scored']}" if v else "not run"
        print(f"  {variant}: {e['run_key']} - CSI {c['CSI']:.4f}, bias {c['bias']:.4f}, n {c['n_tiles_scored']} "
              f"[{val}]")
    if "best_variant" in selection:
        print(f"best variant: {selection['best_variant']}")
    print(f"\nWrote variant_runs_per_tile.csv, variant_runs_pooled.csv and {SELECTION_NAME} in {base_dir}")


if __name__ == "__main__":
    main()
