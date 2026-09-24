"""Per-tile sensitivity of eikonal flooded area to max_rounds ("sweeps") and
obstacle_coupling.max_outer_iterations ("external rounds") - box-whisker of
the per-tile % change in flooded area vs each parameter's own reference
setting (2026-09-24, user direction):
  - max_rounds: reference = 4 (the sweep's lowest value), compared against
    8 and 20.
  - obstacle_coupling: reference = "off" (disabled), compared against
    iter1/iter3/iter10 (enabled, max_outer_iterations=1/3/10).

This is NOT calibration_results.csv's own region-level HR/FAR/CSI-vs-
benchmark comparison (see plot_calibration_sensitivity.py for that) - this
reads each sweep point's own per-tile waterdepth_RP*_SLR_0.tif directly
(model_outputs/<tile_id>/results/) and measures the eikonal model's
self-sensitivity to these two hyperparameters, independent of any benchmark
ground truth.

Pools esp_fra_rp100 (RP100, 78 tiles) and nor_rp250 (RP250, up to 41 tiles)
together per user direction - one box per parameter value, across every
tile present in both that value's own run and its reference run (some
sweep points, e.g. nor_rp250__obstacle_coupling_iter3, only finished a
subset of tiles - handled by intersecting tile sets per comparison, not
by erroring).

Usage:
    python plot_calibration_tile_sensitivity_boxplot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]

WET_THRESHOLD_M = 0.05  # matches flood_agreement.py's own WET_THRESHOLD_M convention
WATERDEPTH_SCALE = 100.0  # raw int16 = cm, matches rasters.py's encode_waterdepth_cm convention
WATERDEPTH_NODATA = 32767

# (group, RP suffix) - the two sweep families in calibration_esp_fra_nor/
GROUPS = [("esp_fra_rp100", "RP100"), ("nor_rp250", "RP250")]

# param -> (reference sweep-point suffix, [(display value, sweep-point suffix), ...])
PARAM_RUNS = {
    "max_rounds": ("max_rounds_4", [("8", "max_rounds_8"), ("20", "max_rounds_20")]),
    "obstacle_coupling": (
        "obstacle_coupling_off",
        [("1", "obstacle_coupling_iter1"), ("3", "obstacle_coupling_iter3"), ("10", "obstacle_coupling_iter10")],
    ),
}


def _pixel_area_km2_by_row(transform, height: int, crs) -> np.ndarray:
    """Latitude-corrected per-row pixel area (km2) for an EPSG:4326 raster -
    same convention used throughout this session's own area comparisons
    (postprocess_tile_summary.py's own _pixel_area_km2_by_row)."""
    assert str(crs).upper() == "EPSG:4326", crs
    px_w_deg = abs(transform.a)
    px_h_deg = abs(transform.e)
    rows = np.arange(height)
    lat_top = transform.f
    lat_center = lat_top + transform.e * (rows + 0.5)
    w_m = px_w_deg * 111320.0 * np.cos(np.radians(lat_center))
    h_m = px_h_deg * 110540.0
    return (w_m * h_m) / 1e6


def _flooded_area_km2(path: Path) -> float:
    with rasterio.open(path) as src:
        raw = src.read(1)
        nodata = src.nodata if src.nodata is not None else WATERDEPTH_NODATA
        row_area = _pixel_area_km2_by_row(src.transform, src.height, src.crs)
    depth_m = raw.astype(np.float32) / WATERDEPTH_SCALE
    wet = (raw != nodata) & (depth_m > WET_THRESHOLD_M)
    return float((wet.sum(axis=1) * row_area).sum())


def _tile_ids(model_outputs_dir: Path) -> set[str]:
    if not model_outputs_dir.is_dir():
        return set()
    return {p.name for p in model_outputs_dir.iterdir() if p.is_dir()}


def collect_sensitivity(sweep_root: Path) -> pd.DataFrame:
    records = []
    for group, rp in GROUPS:
        for param, (ref_suffix, targets) in PARAM_RUNS.items():
            ref_dir = sweep_root / f"{group}__{ref_suffix}" / "model_outputs"
            ref_tiles = _tile_ids(ref_dir)
            for value, target_suffix in targets:
                target_dir = sweep_root / f"{group}__{target_suffix}" / "model_outputs"
                target_tiles = _tile_ids(target_dir)
                common = sorted(ref_tiles & target_tiles)
                if not common:
                    print(f"  {group}/{param}={value}: no tiles in common with reference ({ref_suffix}) - skipping")
                    continue
                n_ok = 0
                for tile_id in common:
                    ref_path = ref_dir / tile_id / "results" / f"waterdepth_{rp}_SLR_0.tif"
                    target_path = target_dir / tile_id / "results" / f"waterdepth_{rp}_SLR_0.tif"
                    if not (ref_path.is_file() and target_path.is_file()):
                        continue
                    ref_km2 = _flooded_area_km2(ref_path)
                    target_km2 = _flooded_area_km2(target_path)
                    abs_change = target_km2 - ref_km2
                    pct_change = (abs_change / ref_km2 * 100.0) if ref_km2 > 0 else np.nan
                    records.append({
                        "group": group, "param": param, "value": value, "tile_id": tile_id,
                        "ref_km2": ref_km2, "target_km2": target_km2,
                        "abs_change_km2": abs_change, "pct_change": pct_change,
                    })
                    n_ok += 1
                print(f"  {group}/{param}={value}: {n_ok} of {len(common)} common tile(s) read OK")
    return pd.DataFrame.from_records(records)


def plot_boxwhisker(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, (param, (ref_suffix, targets)) in zip(axes, PARAM_RUNS.items()):
        values = [v for v, _ in targets]
        sub = df[df["param"] == param]
        data = [sub.loc[sub["value"] == v, "pct_change"].dropna().to_numpy() for v in values]
        ns = [len(d) for d in data]
        bp = ax.boxplot(data, labels=[f"{v}\n(n={n})" for v, n in zip(values, ns)], showfliers=True, whis=1.5)
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(f"{param}\n(reference: {ref_suffix})", fontsize=11)
        ax.set_xlabel(param.replace("_", " "))
        ax.set_ylabel("% change in flooded area vs reference")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Eikonal flooded-area sensitivity to max_rounds and obstacle_coupling.max_outer_iterations\n"
        "(esp_fra_rp100 + nor_rp250 tiles pooled, per-tile %% change vs each parameter's own reference run)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    config_path = _REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"
    cfg = load_config(str(config_path))
    sweep_root = Path(cfg["paths"]["root"]) / "calibration_esp_fra_nor"

    print("Collecting per-tile flooded area for each sweep point...")
    df = collect_sensitivity(sweep_root)
    if df.empty:
        print("No data collected - nothing to plot.")
        return

    csv_path = sweep_root / "calibration_tile_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} row(s))")

    for param in PARAM_RUNS:
        sub = df[df["param"] == param]
        print(f"\n{param}:")
        for value in sub["value"].unique():
            s = sub.loc[sub["value"] == value, "pct_change"].dropna()
            if s.empty:
                continue
            print(f"  {value}: n={len(s)} median={s.median():.2f}% IQR=[{s.quantile(0.25):.2f}, {s.quantile(0.75):.2f}]% "
                  f"min={s.min():.2f}% max={s.max():.2f}%")

    fig_path = sweep_root / "calibration_tile_sensitivity_boxplot.png"
    plot_boxwhisker(df, fig_path)


if __name__ == "__main__":
    main()
