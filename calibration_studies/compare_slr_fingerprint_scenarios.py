"""Compare SLR fingerprint shape between two scenarios across COAST-RP stations.

Tests the assumption that the fingerprint shape is scenario-invariant, so one
reference scenario can be rescaled to any target global-mean SLR. Rescaling is
base / global_mean * T, so Pearson r does not depend on the target T.

Reads, and on a cache miss writes, the same COAST-RP / SLR caches as
preparation/prepare_boundary_conditions.py in `paths.processed_inputs_dir`.
The figure goes to --out (default: the current directory).

Usage:
    python calibration_studies/compare_slr_fingerprint_scenarios.py
    python calibration_studies/compare_slr_fingerprint_scenarios.py --scenario-a ssp126 --scenario-b ssp245
    python calibration_studies/compare_slr_fingerprint_scenarios.py --scenario-a ssp245 --scenario-b ssp585
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "preparation"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from prepare_boundary_conditions import compute_slr_fingerprints, preprocess_coastrp  # noqa: E402

TARGET_SLR_M = 1.0  # plot units only; r is invariant to it


def _load_scenario_fingerprints(
    ds_coastrp: xr.Dataset, slr_dir: Path, scenario: str, confidence: str, cache_dir: Path,
) -> np.ndarray:
    slr_base_path = cache_dir / f"SLR_base_{scenario}_{confidence}_2100.nc"
    fingerprints_path = cache_dir / f"SLR_fingerprints_{scenario}_{confidence}_all.nc"
    ds_fp = compute_slr_fingerprints(
        ds_coastrp, slr_dir, scenario, confidence, [TARGET_SLR_M], slr_base_path, fingerprints_path,
    )
    key = f"SLR_{int(TARGET_SLR_M * 1000)}mm"
    return ds_fp[key].values.astype(np.float64)


def main() -> None:
    _default_cfg = str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--scenario-a", default="ssp126", help="options: ssp119/ssp126/ssp245/ssp370/ssp585")
    parser.add_argument("--scenario-b", default="ssp245")
    parser.add_argument("--confidence", default=None, help="default: boundary_conditions.confidence_level from config")
    parser.add_argument("--out", default="slr_fingerprint_scenario_comparison.png")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    wl_cfg = cfg["boundary_conditions"]
    confidence = args.confidence or wl_cfg.get("confidence_level", "medium")

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    slr_dir = Path(catalog.get_source("ipcc_ar6_slr_projections").path) / f"{confidence}_confidence"

    proc_dir = Path(cfg["paths"]["processed_inputs_dir"])
    preprocessed_path = proc_dir / "COAST-RP_preprocessed.nc"
    ds_coastrp = preprocess_coastrp(
        Path(catalog.get_source("coast_rp").path), preprocessed_path, wl_cfg["coastrp_min_lat"],
    )
    n_stations = int(ds_coastrp.dims["stations"])
    print(f"{n_stations} COAST-RP stations (Antarctic already excluded, coastrp_min_lat={wl_cfg['coastrp_min_lat']})")

    print(f"\nComputing fingerprints for {args.scenario_a} ({confidence} confidence)...")
    offsets_a = _load_scenario_fingerprints(ds_coastrp, slr_dir, args.scenario_a, confidence, proc_dir)
    print(f"Computing fingerprints for {args.scenario_b} ({confidence} confidence)...")
    offsets_b = _load_scenario_fingerprints(ds_coastrp, slr_dir, args.scenario_b, confidence, proc_dir)

    valid = np.isfinite(offsets_a) & np.isfinite(offsets_b)
    n_valid = int(valid.sum())
    x, y = offsets_a[valid], offsets_b[valid]

    r = float(np.corrcoef(x, y)[0, 1])
    bias = float(np.mean(y - x))
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    print(f"\n{n_valid}/{n_stations} stations with finite values in both scenarios")
    print(f"  Pearson r ({args.scenario_a} vs {args.scenario_b}): {r:.4f}")
    print(f"  mean bias (b - a): {bias:+.4f} m   RMSE: {rmse:.4f} m")
    print(f"  (this r holds for any target global-mean SLR, not just {TARGET_SLR_M}m - see module docstring)")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=10, alpha=0.4, color="#2a78d6", edgecolors="none")
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, color="black", linestyle=":", linewidth=1, label="1:1")
    ax.set_xlabel(f"{args.scenario_a} local SLR offset at global-mean SLR={TARGET_SLR_M}m (m)")
    ax.set_ylabel(f"{args.scenario_b} local SLR offset at global-mean SLR={TARGET_SLR_M}m (m)")
    ax.set_title(
        f"SLR fingerprint shape: {args.scenario_a} vs {args.scenario_b}\n"
        f"(n={n_valid} COAST-RP stations, r={r:.3f}, bias={bias:+.3f}m, RMSE={rmse:.3f}m)",
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
