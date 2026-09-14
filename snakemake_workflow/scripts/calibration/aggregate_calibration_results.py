"""Aggregate every materialized run's validate_country.py output into one
`calibration_results.csv`, for the ESP/FRA/NOR HPC calibration sweep
(docs/calibration_sweep_plan.md).

Walks `{root}/calibration_esp_fra_nor/*/validation/*/metrics_*.csv` (one
directory per run_tag = "{group}__{sweep_point}", written by
scripts/calibration/build_run_config.py's path isolation), concatenates,
and tags each row with `group`/`sweep_point`/`swept_param`/`swept_value`
columns - parsed via an EXPLICIT lookup table keyed on the real sweep-point
filenames under config/calibration/sweep/, not regex-guessing (a wrong
guess here would silently mislabel a whole run's results).

run_tag is read from each CSV's own `run_tag` column if validate_country.py
was run with `--run-tag` (recommended); falls back to inferring it from the
file's own path (the `{run_tag}` path segment) otherwise, so this still
works even for a run whose validate_country.py invocation forgot the flag.

Usage:
    python aggregate_calibration_results.py [--config <config.yml>]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]

# (swept_param, swept_value) per sweep-point filename (without .yml) - must
# stay in sync with config/calibration/sweep/*.yml's actual contents; not
# derived from them automatically, so a new sweep-point file needs an entry
# added here too, deliberately (a silent auto-derivation could mislabel a
# result if a sweep file's own structure ever changes).
SWEEP_POINTS: dict[str, tuple[str | None, object]] = {
    "baseline": (None, None),
    "friction_0.5": ("friction_scale_factor", 0.5),
    "friction_2.0": ("friction_scale_factor", 2.0),
    "max_rounds_4": ("max_rounds", 4),
    "max_rounds_8": ("max_rounds", 8),
    "max_rounds_20": ("max_rounds", 20),
    "obstacle_coupling_off": ("obstacle_coupling.enabled", False),
    "obstacle_coupling_iter1": ("obstacle_coupling.max_outer_iterations", 1),
    "obstacle_coupling_iter3": ("obstacle_coupling.max_outer_iterations", 3),
    "obstacle_coupling_iter10": ("obstacle_coupling.max_outer_iterations", 10),
    "waterlevel_eps_0.01": ("waterlevel_epsilon_m", 0.01),
    "waterlevel_eps_0.10": ("waterlevel_epsilon_m", 0.10),
}


def _split_run_tag(run_tag: str) -> tuple[str, str]:
    group, _, sweep_point = run_tag.partition("__")
    if not sweep_point:
        raise ValueError(f"run_tag {run_tag!r} doesn't match the expected '{{group}}__{{sweep_point}}' shape")
    return group, sweep_point


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    cfg = load_config(args.config)
    sweep_root = Path(cfg["paths"]["root"]) / "calibration_esp_fra_nor"

    metrics_paths = sorted(sweep_root.glob("*/validation/*/metrics_*.csv"))
    if not metrics_paths:
        print(f"No metrics_*.csv found under {sweep_root} - nothing to aggregate.")
        return

    rows = []
    unknown_sweep_points: set[str] = set()
    for path in metrics_paths:
        df = pd.read_csv(path)
        if "run_tag" in df.columns and df["run_tag"].notna().all():
            run_tag = df["run_tag"].iloc[0]
        else:
            # Fallback: .../{run_tag}/validation/{country}/metrics_*.csv
            run_tag = path.parents[2].name
            df["run_tag"] = run_tag
        group, sweep_point = _split_run_tag(run_tag)
        df["group"] = group
        df["sweep_point"] = sweep_point
        if sweep_point in SWEEP_POINTS:
            swept_param, swept_value = SWEEP_POINTS[sweep_point]
        else:
            unknown_sweep_points.add(sweep_point)
            swept_param, swept_value = None, None
        df["swept_param"] = swept_param
        df["swept_value"] = swept_value
        rows.append(df)

    if unknown_sweep_points:
        print(f"WARNING: {len(unknown_sweep_points)} sweep-point name(s) not in SWEEP_POINTS "
              f"(swept_param/swept_value left blank for these): {sorted(unknown_sweep_points)}")

    combined = pd.concat(rows, ignore_index=True)
    out_path = sweep_root / "calibration_results.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} row(s) from {len(metrics_paths)} file(s), "
          f"{combined['run_tag'].nunique()} run_tag(s))")


if __name__ == "__main__":
    main()
