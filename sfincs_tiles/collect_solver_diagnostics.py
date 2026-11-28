"""Tabulate the eikonal solver diagnostics of every calibration run, one row per tile and run.

Reads each tile's model_runs.json and writes <base>/variant_runs_solver.csv plus a per-variant
summary, variant_runs_solver_summary.csv. Bathtub and straight-line runs have no sweeps: their
round and outer columns stay empty. solve_s is the solve alone (a capped run shares its base
run's solve); runtime_s adds writing and the volume cap.

Usage:
    python collect_solver_diagnostics.py --config <resolved_config.yml> --base-dir-name calibration_roughness_SFINCS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from calibration_paths import RUNS_NAME  # noqa: E402
from gfm_config import read_root  # noqa: E402

SOLVER_NAME = "variant_runs_solver.csv"
SUMMARY_NAME = "variant_runs_solver_summary.csv"


def _join(values) -> str | None:
    return None if values is None else " ".join(str(v) for v in values)


def solver_rows(tile_id: str, runs: dict) -> list[dict]:
    """Return one row per run of a tile's model_runs.json."""
    rows = []
    for key, run in sorted(runs.items()):
        s = run.get("solver") or {}
        rounds = s.get("inner_rounds_per_outer")
        rows.append({
            "tile_id": tile_id, "run_key": key, "variant": run["variant"],
            "parameter": run.get("parameter"), "alpha": run.get("alpha"),
            "outer_iterations_used": s.get("outer_iterations_used"),
            "outer_converged": s.get("outer_converged"),
            "inner_rounds_per_outer": _join(rounds),
            "inner_converged_per_outer": _join(s.get("inner_converged_per_outer")),
            "max_inner_rounds": max(rounds) if rounds else None,
            "all_inner_converged": all(s["inner_converged_per_outer"]) if rounds else None,
            "newly_blocked_per_outer": _join(s.get("newly_blocked_per_outer")),
            "last_blocking_outer": s.get("last_blocking_outer"),
            "solve_s": run.get("solve_s"), "runtime_s": run.get("runtime_s"),
        })
    return rows


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per variant: runs, rounds and outer-iteration distribution, solve time."""
    def one(variant: str, g: pd.DataFrame) -> dict:
        eik = g.dropna(subset=["max_inner_rounds"])
        return {
            "variant": variant, "n_runs": len(g),
            "max_inner_rounds_median": eik["max_inner_rounds"].median(),
            "max_inner_rounds_p90": eik["max_inner_rounds"].quantile(0.9),
            "max_inner_rounds_max": eik["max_inner_rounds"].max(),
            "n_inner_not_converged": int((eik["all_inner_converged"] == False).sum()),  # noqa: E712
            "outer_iterations_max": eik["outer_iterations_used"].max(),
            "n_outer_not_converged": int((eik["outer_converged"] == False).sum()),  # noqa: E712
            "last_blocking_outer_max": eik["last_blocking_outer"].max(),
            "solve_s_median": g["solve_s"].median(),
            "solve_s_p90": g["solve_s"].quantile(0.9),
            "solve_s_max": g["solve_s"].max(),
        }
    return pd.DataFrame([one(v, g) for v, g in table.groupby("variant", sort=True)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    args = parser.parse_args()

    base = read_root(Path(args.config)) / args.base_dir_name
    rows = []
    for runs_path in sorted(base.glob(f"*/outputs/{RUNS_NAME}")):
        rows += solver_rows(runs_path.parent.parent.name, json.loads(runs_path.read_text()))
    if not rows:
        raise SystemExit(f"no {RUNS_NAME} under {base}")
    table = pd.DataFrame(rows)
    table.to_csv(base / SOLVER_NAME, index=False)
    summary = summarize(table)
    summary.to_csv(base / SUMMARY_NAME, index=False)
    print(f"{len(table)} run(s) on {table['tile_id'].nunique()} tile(s) -> {base / SOLVER_NAME}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
