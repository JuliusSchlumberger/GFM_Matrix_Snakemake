"""Pick a candidate pool of wave-0 tiles spanning both the full size range
AND a wide geographic spread, for the obstacle-coupling/sweep-budget
calibration studies (2026-08 - 400-tile scale-up from the earlier 40-tile
study; see `C:\\Users\\Schlu005\\.claude\\plans\\smooth-wandering-map.md`
for that study's original methodology; 2026-09-24 - repointed at this
session's own machine/paths and the 260-tile study, ~10% of all hop=0
tiles, replacing the never-actually-completed 100-tile study whose
D:\\GFM\\model_outputs paths belonged to a different machine).

Candidates are restricted to tiles whose model_outputs/<id>/inputs/dem.tif
already exists (2026-09-24) - the 100-tile study's real failure mode was
NOT dry tiles, it was candidates with no preprocessed inputs at all (the
Snakemake rebuild was still failing on most tiles when that study ran, so
only 6 of 143 candidates were usable) - checking this up front means every
candidate this script hands out is actually runnable right now.

Bbox area in deg2 is an exact-up-to-clipping proxy for pixel count here
(DeltaDTM tiles are native EPSG:4326 at ~1 arcsecond, so pixel count =
area_deg2 * 3600**2 regardless of latitude - no cos(lat) correction needed
since what matters is compute cost, not physical area). Restricted to
hop_distance == 0 (wave-0) tiles - obstacle_coupling isn't usable with
explicit seed cells (hop>=1) yet.

Geographic stratification: tiles are first binned into a coarse lon/lat grid
(GEO_BIN_DEG per side), then each occupied bin gets a candidate quota
proportional to sqrt(bin tile count) - dampens a few dense archipelago-heavy
bins from crowding out sparser coastlines, while still giving genuinely
tile-rich regions more representation than a one-tile bin. Within each bin,
candidates are picked at evenly-spaced percentiles of THAT bin's own area
distribution (same percentile-grid logic as the original single-axis
version), so both size and location vary together rather than one being a
free side-effect of the other.

Picks N candidates total - more than the final tile count actually needed,
since some candidates will turn out to have zero flooding for this scenario
and get dropped by the calibration scripts themselves (see
test_sweep_budget_calibration.py's dry-tile handling) - this is deliberately
just a candidate POOL, not the final list.

Writes candidate_tiles.txt (one tile_id per line, in bin then percentile-rank
order) to the given output directory, and prints a summary table.

Usage:
    python select_calibration_tiles.py <output_dir> [n_candidates]
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N_CANDIDATES = 300  # ~1.15x the 260 wanted (10% of ~2445 hop=0 tiles) - matches the
# margin ratio the earlier 400-tile study used, for replacing dry-at-sweep-1 tiles
GEO_BIN_DEG = 20.0  # lon/lat grid cell size for geographic stratification


def select_candidates(n_candidates: int, domain_tiles_path: Path, model_outputs_root: Path) -> list[tuple[int, str, float, float]]:
    """Returns [(tile_id, geo_bin, percentile_within_bin, area_deg2), ...]."""
    gdf = gpd.read_file(domain_tiles_path)
    wave0 = gdf[gdf["hop_distance"] == 0].copy()
    n_before_ready = len(wave0)
    ready = wave0["tile_id"].apply(lambda t: (model_outputs_root / str(int(t)) / "inputs" / "dem.tif").is_file())
    wave0 = wave0[ready].copy()
    print(f"{len(wave0)} of {n_before_ready} hop=0 tile(s) already have model_outputs/inputs/dem.tif built "
          f"- restricting candidates to these")
    bounds = wave0.geometry.bounds
    wave0["area_deg2"] = (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"])
    cx = (bounds["minx"] + bounds["maxx"]) / 2.0
    cy = (bounds["miny"] + bounds["maxy"]) / 2.0
    wave0["geo_bin"] = [
        f"{int(math.floor(x / GEO_BIN_DEG))},{int(math.floor(y / GEO_BIN_DEG))}"
        for x, y in zip(cx, cy)
    ]

    bins = defaultdict(list)
    for _, row in wave0.iterrows():
        bins[row["geo_bin"]].append(row)

    # Quota per bin proportional to sqrt(bin size), rounded, at least 1 per bin.
    weights = {b: math.sqrt(len(rows)) for b, rows in bins.items()}
    total_weight = sum(weights.values())
    quotas = {b: max(1, round(n_candidates * w / total_weight)) for b, w in weights.items()}

    result = []
    for geo_bin, rows in sorted(bins.items()):
        quota = min(quotas[geo_bin], len(rows))
        bin_sorted = sorted(rows, key=lambda r: r["area_deg2"])
        n = len(bin_sorted)
        percentiles = np.linspace(0.5, 99.5, quota) if quota > 1 else np.array([50.0])
        seen_idx: set[int] = set()
        for pct in percentiles:
            idx = min(int(round(pct / 100.0 * (n - 1))), n - 1)
            while idx in seen_idx and idx < n - 1:
                idx += 1
            seen_idx.add(idx)
            row = bin_sorted[idx]
            result.append((int(row["tile_id"]), geo_bin, float(pct), float(row["area_deg2"])))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir")
    parser.add_argument("n_candidates", type=int, nargs="?", default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"])
    domain_tiles_path = root / "processed_inputs" / "mask" / "domain_tiles_global.gpkg"
    model_outputs_root = root / "model_outputs"

    candidates = select_candidates(args.n_candidates, domain_tiles_path, model_outputs_root)
    n_bins = len({c[1] for c in candidates})

    print(f"{'tile_id':>10}  {'geo_bin':>12}  {'pct_in_bin':>10}  {'area_deg2':>12}  {'approx_px':>15}")
    for tile_id, geo_bin, pct, area in candidates:
        approx_px = area * 3600 * 3600
        print(f"{tile_id:>10}  {geo_bin:>12}  {pct:>10.1f}  {area:>12.6f}  {approx_px:>15,.0f}")

    tiles_file = out_dir / "candidate_tiles.txt"
    with open(tiles_file, "w") as f:
        for tile_id, _geo_bin, _pct, _area in candidates:
            f.write(f"{tile_id}\n")
    print(f"\n{len(candidates)} candidate tile_ids across {n_bins} geographic bins written to {tiles_file}")


if __name__ == "__main__":
    main()
