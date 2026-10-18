"""Pick calibration candidate tiles stratified by size and geography.

Wave-0 (hop_distance == 0) only: obstacle_coupling does not support explicit
seed cells. Existing model_outputs are not required. Pixel count = bbox
area_deg2 * 3600**2 (EPSG:4326, ~1 arcsecond; no cos(lat), compute cost not
area). Quota per GEO_BIN_DEG lon/lat bin ~ sqrt(bin tile count); within a bin,
evenly spaced area percentiles. Over-provisioned for dry tiles.

Usage:
    python select_study_candidate_tiles.py --out-dir <output_dir> [--n-candidates N]

The positional form `<output_dir> [n_candidates]` is still accepted.
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cli_args import add_option_with_positional, require  # noqa: E402
from config_utils import load_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N_CANDIDATES = 575  # ~1.15x the 500 wanted, for tiles dry at sweep 1
GEO_BIN_DEG = 20.0


def select_candidates(n_candidates: int, domain_tiles_path: Path, model_outputs_root: Path) -> list[tuple[int, str, float, float]]:
    """Return [(tile_id, geo_bin, percentile_within_bin, area_deg2), ...]."""
    gdf = gpd.read_file(domain_tiles_path)
    wave0 = gdf[gdf["hop_distance"] == 0].copy()
    n_ready = int(wave0["tile_id"].apply(lambda t: (model_outputs_root / str(int(t)) / "inputs" / "dem.tif").is_file()).sum())
    print(f"{len(wave0)} hop=0 tile(s) total ({n_ready} already have model_outputs/inputs/dem.tif built - "
          f"informational only, NOT a selection filter, see module docstring)")
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

    # sqrt damps dense archipelago bins crowding out sparse coastlines.
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
    add_option_with_positional(parser, "--out-dir", "out_dir")
    add_option_with_positional(parser, "--n-candidates", "n_candidates", type=int,
                               help=f"default: {DEFAULT_N_CANDIDATES}")
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    args = parser.parse_args()
    require(parser, args, "out_dir", "--out-dir")
    if args.n_candidates is None:
        args.n_candidates = DEFAULT_N_CANDIDATES

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
