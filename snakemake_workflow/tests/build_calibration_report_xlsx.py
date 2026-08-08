"""Post-hoc re-analysis of the 40-tile obstacle-coupling calibration study.

Reads the already-collected per-tile CSVs written by
test_sweep_budget_calibration.py and test_obstacle_coupling_calibration.py
and assembles a single XLSX workbook suitable for a paper appendix. Does
NOT re-run any solver - every diagnostic here is derived from data already
on disk.

Convergence definitions (see "Notes" sheet in the output workbook for the
paper-facing version of this text):

- Sweep-level: a tile is "converged" at sweep S if max_depth_change_abs
  stays below WATERLEVEL_EPSILON_M (0.03 m, the same threshold production's
  round-based solver uses) for every sweep from S through the last sweep
  run (sustained, not a one-off dip - false plateaus followed by a later
  jump are common in this dataset). The "reference sweep" used for the
  jump/max-depth-change columns is the sweep BEFORE S (the last real
  change before it settled), or S itself if S == 1 (no earlier sweep
  exists). If no such S exists, the tile is "not converged" and the
  reference sweep is the last sweep actually run.
- Outer-loop level: convergence is read directly from the calibration
  script's own outer_stopped_early flag on the last real (non-summary)
  round. The reference outer iteration is the one BEFORE the converged
  iteration, or the final iteration itself if it hit the outer-iteration
  cap without converging.
"""

import csv
from pathlib import Path

import pandas as pd

RUN_DIR = Path(r"C:\Users\Schlu005\GFM\tests\obstacle_coupling_calibration_40")
SWEEP_DIR = RUN_DIR / "sweep_budget"
OUTER_DIR = RUN_DIR / "obstacle_coupling"
OUT_XLSX = RUN_DIR / "calibration_report.xlsx"

WATERLEVEL_EPSILON_M = 0.03  # matches src/flood_model.py:WATERLEVEL_EPSILON_M


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_wet_tiles() -> list[str]:
    with (SWEEP_DIR / "wet_tiles_selected.txt").open(encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def analyze_sweep_tile(tile: str) -> tuple[dict, list[dict]]:
    rows = _read_rows(SWEEP_DIR / f"{tile}.csv")
    for r in rows:
        r["sweep_count"] = int(r["sweep_count"])
        for k in (
            "cum_time_s", "max_depth_change_abs", "median_depth_change_abs", "mean_depth_change_abs",
            "pct_cells_changed", "depth_mean", "depth_sum", "depth_max",
        ):
            r[k] = float(r[k])
        for k in ("n_inundated", "n_newly_flooded", "n_cells_changed", "n_cells"):
            r[k] = int(r[k])
    rows.sort(key=lambda r: r["sweep_count"])

    for i, r in enumerate(rows):
        prev_n = rows[i - 1]["n_inundated"] if i > 0 else 0
        r["delta_n_inundated"] = r["n_inundated"] - prev_n
        r["pct_change_n_inundated"] = (
            round(100.0 * r["delta_n_inundated"] / prev_n, 4) if prev_n > 0 else None
        )

    n_sweeps_run = rows[-1]["sweep_count"]

    convergence_sweep = None
    for r in reversed(rows):
        if r["max_depth_change_abs"] < WATERLEVEL_EPSILON_M:
            convergence_sweep = r["sweep_count"]
        else:
            break
    converged = convergence_sweep is not None

    if converged:
        reference_sweep = convergence_sweep - 1 if convergence_sweep > 1 else 1
    else:
        reference_sweep = n_sweeps_run

    by_sweep = {r["sweep_count"]: r for r in rows}
    ref_row = by_sweep[reference_sweep]
    prev_row = by_sweep.get(reference_sweep - 1)

    if prev_row is not None and prev_row["n_inundated"] > 0:
        flood_extent_jump_pct = round(
            100.0 * (ref_row["n_inundated"] - prev_row["n_inundated"]) / prev_row["n_inundated"], 4
        )
    else:
        flood_extent_jump_pct = None  # reference sweep is sweep 1 - no prior sweep to compare to

    final_row = rows[-1]
    total_time_s = final_row["cum_time_s"]

    summary = {
        "tile": tile,
        "n_cells": ref_row["n_cells"],
        "n_sweeps_run": n_sweeps_run,
        "total_time_s": round(total_time_s, 3),
        "avg_time_per_sweep_s": round(total_time_s / n_sweeps_run, 4),
        "converged": converged,
        "convergence_sweep": convergence_sweep,
        "reference_sweep": reference_sweep,
        "cells_changed_pct_at_reference": ref_row["pct_cells_changed"],
        "flood_extent_jump_pct_at_reference": flood_extent_jump_pct,
        "max_depth_change_abs_at_reference_m": ref_row["max_depth_change_abs"],
        "n_inundated_final": final_row["n_inundated"],
        "depth_mean_final_m": final_row["depth_mean"],
        "depth_max_final_m": final_row["depth_max"],
    }
    return summary, rows


def analyze_outer_tile(tile: str) -> tuple[dict, list[dict]] | tuple[dict, None]:
    rows = _read_rows(OUTER_DIR / f"{tile}.csv")
    error_row = next((r for r in rows if r["status"].startswith("error")), None)
    if error_row is not None:
        return {
            "tile": tile,
            "usable": False,
            "exclusion_reason": error_row.get("status", ""),
        }, None

    summary_row = next(r for r in rows if r["status"] == "summary")
    real_rows = [r for r in rows if r["status"] == "ok"]

    for r in real_rows:
        r["n_outer"] = int(r["n_outer"])
        r["n_cells"] = int(r["n_cells"])
        r["outer_stopped_early"] = r["outer_stopped_early"] == "True"
        for k in ("cum_time_s", "iter_time_s", "pct_blocked_cumulative",
                  "pct_newly_blocked", "max_depth_change_abs"):
            r[k] = float(r[k])
        for k in ("n_newly_flooded", "n_no_longer_flooded", "n_inundated"):
            r[k] = int(r[k])
    real_rows.sort(key=lambda r: r["n_outer"])

    for i, r in enumerate(real_rows):
        if i == 0:
            r["delta_n_inundated"] = 0
            r["pct_change_n_inundated"] = None
        else:
            prev_n = real_rows[i - 1]["n_inundated"]
            r["delta_n_inundated"] = r["n_inundated"] - prev_n
            r["pct_change_n_inundated"] = (
                round(100.0 * r["delta_n_inundated"] / prev_n, 4) if prev_n > 0 else None
            )

    baseline_row = next((r for r in real_rows if r["n_outer"] == 0), None)
    iter_rows = [r for r in real_rows if r["n_outer"] > 0]
    baseline_time_s = round(baseline_row["iter_time_s"], 3) if baseline_row else None
    avg_time_per_outer_iteration_s = (
        round(sum(r["iter_time_s"] for r in iter_rows) / len(iter_rows), 4) if iter_rows else None
    )

    last_row = real_rows[-1]
    converged = last_row["outer_stopped_early"]
    n_outer_run = last_row["n_outer"]

    if converged:
        reference_iter = n_outer_run - 1
    else:
        reference_iter = n_outer_run

    by_iter = {r["n_outer"]: r for r in real_rows}
    if reference_iter in by_iter:
        ref_row = by_iter[reference_iter]
    else:
        # reference_iter == 0 (baseline, unblocked) - no change metrics recorded there
        ref_row = {
            "pct_newly_blocked": 0.0, "max_depth_change_abs": 0.0,
            "n_newly_flooded": 0, "n_no_longer_flooded": 0,
        }

    summary = {
        "tile": tile,
        "usable": True,
        "n_cells": last_row["n_cells"],
        "pct_removed_by_static_filter_alone": float(summary_row["pct_removed_by_static_filter_alone"]),
        "pct_additional_removed_by_full_outer_loop": float(summary_row["pct_additional_removed_by_full_outer_loop"]),
        "outer_iterations_run": n_outer_run,
        "baseline_time_s": baseline_time_s,
        "avg_time_per_outer_iteration_s": avg_time_per_outer_iteration_s,
        "total_time_s": round(last_row["cum_time_s"], 3),
        "converged": converged,
        "reference_outer_iteration": reference_iter,
        "pct_newly_blocked_at_reference": ref_row["pct_newly_blocked"],
        "max_depth_change_abs_at_reference_m": ref_row["max_depth_change_abs"],
        "n_newly_flooded_at_reference": ref_row["n_newly_flooded"],
        "n_no_longer_flooded_at_reference": ref_row["n_no_longer_flooded"],
        "pct_blocked_final": last_row["pct_blocked_cumulative"],
    }
    return summary, real_rows


def main() -> None:
    wet_tiles = load_wet_tiles()

    sweep_summaries, sweep_detail_rows = [], []
    for tile in wet_tiles:
        summary, rows = analyze_sweep_tile(tile)
        sweep_summaries.append(summary)
        sweep_detail_rows.extend(rows)

    outer_summaries, outer_detail_rows = [], []
    for tile in wet_tiles:
        summary, rows = analyze_outer_tile(tile)
        outer_summaries.append(summary)
        if rows:
            outer_detail_rows.extend(rows)

    df_sweep_summary = pd.DataFrame(sweep_summaries)
    df_sweep_summary["convergence_sweep"] = df_sweep_summary["convergence_sweep"].astype("Int64")

    df_outer_summary = pd.DataFrame(outer_summaries)
    df_outer_summary["converged"] = df_outer_summary["converged"].astype("boolean")
    for col in ("outer_iterations_run", "reference_outer_iteration",
                "n_newly_flooded_at_reference", "n_no_longer_flooded_at_reference"):
        df_outer_summary[col] = df_outer_summary[col].astype("Int64")

    df_sweep_detail = pd.DataFrame(sweep_detail_rows)
    df_outer_detail = pd.DataFrame(outer_detail_rows)

    df_sweep_change = df_sweep_detail[[
        "tile", "sweep_count", "n_inundated", "delta_n_inundated", "pct_change_n_inundated",
        "max_depth_change_abs", "median_depth_change_abs", "mean_depth_change_abs",
    ]].rename(columns={
        "delta_n_inundated": "flooded_area_change_cells",
        "pct_change_n_inundated": "flooded_area_change_pct",
        "max_depth_change_abs": "max_inundation_change_m",
        "median_depth_change_abs": "median_inundation_change_m",
        "mean_depth_change_abs": "mean_inundation_change_m",
    })

    df_outer_change = df_outer_detail[[
        "tile", "n_outer", "n_inundated", "delta_n_inundated", "pct_change_n_inundated",
        "n_newly_flooded", "n_no_longer_flooded", "max_depth_change_abs",
    ]].rename(columns={
        "delta_n_inundated": "flooded_area_change_cells",
        "pct_change_n_inundated": "flooded_area_change_pct",
        "max_depth_change_abs": "max_inundation_change_m",
    })

    notes = pd.DataFrame({"Notes": [
        "40-tile obstacle-coupling calibration study - tightened bounds for tractability "
        "(not production defaults): sweep-level cap 24 sweeps (6 rounds x 4); outer-loop "
        "cap 3 iterations, each with an inner solve capped at 6 rounds (24 sweeps).",
        "",
        "Sweep-level convergence: a tile is 'converged' at sweep S if max_depth_change_abs "
        "stays below WATERLEVEL_EPSILON_M = 0.03 m (production's own round-based convergence "
        "threshold) for every sweep from S through the last sweep run (sustained, not a "
        "one-off dip - false plateaus followed by a later jump were observed in this dataset).",
        "",
        "Sweep-level 'reference sweep': the sweep the jump/max-depth-change columns are read "
        "from. If converged at sweep S, this is sweep S-1 (the last real change before it "
        "settled - S itself is trivially near-zero by construction). If S == 1, there is no "
        "earlier sweep, so sweep 1 itself is used. If not converged, this is the last sweep "
        "actually run (24).",
        "",
        "'flood_extent_jump_pct_at_reference' is the sweep-over-sweep relative % change in "
        "n_inundated (flooded cell count) at the reference sweep. Left blank when the "
        "reference sweep is sweep 1 (no prior sweep to compare against - the entire count is "
        "new by construction).",
        "",
        "'cells_changed_pct_at_reference' is pct_cells_changed at the reference sweep - the % "
        "of all tile cells (not just newly-flooded ones) whose depth changed that sweep.",
        "",
        "Outer-loop convergence: read directly from the calibration script's own "
        "outer_stopped_early flag on the last outer iteration actually run. The reference "
        "outer iteration is the one before the converged iteration (or the final iteration "
        "itself if the 3-iteration cap was hit without converging).",
        "",
        "Two tiles (2351, 2365) had zero real storm-surge boundary stations - water level "
        "stays at 0 (unseeded), which still exceeds some legitimately-negative-elevation DEM "
        "cells, so they showed nonzero flooding in the sweep-level run despite no real "
        "forcing. They are valid for the sweep-count analysis (pure numerics) but were "
        "excluded from the outer-loop analysis (marked usable=False below) since 'does "
        "re-blocking help' is not physically meaningful without real forcing.",
        "",
        "'avg_time_per_sweep_s' = total sweep-solve wall time / sweeps run. "
        "'avg_time_per_outer_iteration_s' = mean wall time per outer re-blocking iteration, "
        "excluding the baseline (unblocked) solve, which is reported separately as "
        "'baseline_time_s'. All timings are wall-clock from this calibration run's own "
        "single-tile-at-a-time execution, not representative of production's parallel "
        "Snakemake scheduling.",
        "",
        "'Sweep - Flooded Area & Depth Change' and 'Outer - Flooded Area & Depth Change' give "
        "the full per-sweep / per-outer-iteration trace (not just the reference-step summary "
        "above): flooded_area_change is the change in flooded CELL COUNT (n_inundated) versus "
        "the previous step; max/median/mean_inundation_change_m is the change in flood DEPTH "
        "(metres) versus the previous step - this is the per-step change, not the peak depth "
        "ever reached (depth_max, in the Detail sheets).",
    ]})

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="Notes", index=False)
        df_sweep_summary.to_excel(writer, sheet_name="Sweep Summary", index=False)
        df_outer_summary.to_excel(writer, sheet_name="Outer-Loop Summary", index=False)
        df_sweep_change.to_excel(writer, sheet_name="Sweep - Change Trace", index=False)
        df_outer_change.to_excel(writer, sheet_name="Outer - Change Trace", index=False)
        df_sweep_detail.to_excel(writer, sheet_name="Sweep Detail", index=False)
        df_outer_detail.to_excel(writer, sheet_name="Outer-Loop Detail", index=False)

    print(f"written {OUT_XLSX}")
    print(f"  Sweep Summary: {len(df_sweep_summary)} tiles")
    print(f"  Outer-Loop Summary: {len(df_outer_summary)} tiles "
          f"({df_outer_summary['usable'].sum()} usable)")
    print(f"  Sweep - Change Trace: {len(df_sweep_change)} rows")
    print(f"  Outer - Change Trace: {len(df_outer_change)} rows")
    print(f"  Sweep Detail: {len(df_sweep_detail)} rows")
    print(f"  Outer-Loop Detail: {len(df_outer_detail)} rows")


if __name__ == "__main__":
    main()
