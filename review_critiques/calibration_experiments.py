"""Score eikonal variants against the existing SFINCS RP100 calibration results, per tile.

Variants: land-cover friction (fsf sweep), uniform attenuation (critique 2), basin volume
cap A and duration-scaled friction B (critique 3). Solves on SFINCS's 30 m subgrid with the
production solver settings, scored as compute_calibration_result (land & SFINCS-active,
wet > 0.10 m). Reads the calibration tiles read-only; appends records to --out-dir/results.jsonl.

Usage:
    python review_critiques/calibration_experiments.py --out-dir <dir> [--n-random 60] [--workers 2]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))

import run_calibration_models as rcm  # noqa: E402
from event_volume import basin_volume_cap, duration_friction_multiplier, spill_level  # noqa: E402
from flood_agreement import WET_THRESHOLD_M, confusion_counts  # noqa: E402
from flood_model import flood_depth_dense  # noqa: E402
from sfincs_grid import active_on_subgrid  # noqa: E402

DATA_ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
BASE = DATA_ROOT / "calibration_roughness_SFINCS"
LU_FSF = [6.0, 9.0, 12.0, 15.0, 18.0]
UNIFORM_M_PER_KM = [0.1, 0.15, 0.2, 0.3, 0.45, 0.6]
B_FSF = [9.0, 12.0]
B_ALPHA = [0.5, 1.0]
B_T_REF_S = 12 * 3600.0
EXT_B_FSF = [6.0, 7.5]
EXT_T_REF_H = [6.0, 24.0]
EXT_B_UNIFORM_M_PER_KM = [0.1, 0.15]
HYDRO_WINDOW_HR = (40.0, 110.0)  # build_sfincs_tile.TRUNCATE_WINDOW_HR_DEFAULT
SOLVER = dict(ocean_code=1, river_code=3, obstacle_coupling=True, max_outer_iterations=4,
              outer_convergence_pct=0.01, max_rounds=40, waterlevel_epsilon_m=0.03)
MAP_TILES_DOWNSAMPLE = 3


def load_tile(tile_id: str) -> dict:
    tile_dir = BASE / tile_id
    sfincs_dir = tile_dir / "sfincs_model"
    dem, mask, base_friction, transform, crs = rcm.build_inputs_from_sfincs_subgrid(
        sfincs_dir, tile_dir / "inputs" / "mask.tif", friction_scale_factor=1.0, default_friction=0.002,
    )
    boundaries = gpd.read_file(tile_dir / "inputs" / "boundaries_RP100_SLR_0.gpkg")
    seeds = rcm.production_seeds(mask, transform, crs, boundaries, knn=15, river_code=3)
    solver_dem, solver_mask = rcm.solver_inputs(dem, mask)
    with rasterio.open(sfincs_dir / "hmax_subgrid.tif") as src:
        sfincs = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            sfincs[sfincs == src.nodata] = np.nan
    score = (mask == rcm.LAND_CODE) & active_on_subgrid(sfincs_dir, dem.shape, transform)
    cell_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    hydro = pd.read_csv(sfincs_dir / "corrected_hydrographs.csv")
    hydro = hydro[(hydro.elapsed_hr >= HYDRO_WINDOW_HR[0]) & (hydro.elapsed_hr <= HYDRO_WINDOW_HR[1])]
    points = gpd.read_file(sfincs_dir / "matched_boundary_points.gpkg").to_crs("EPSG:4326")
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    px, py = to_utm.transform(points.geometry.x.to_numpy(), points.geometry.y.to_numpy())
    prow, pcol = rasterio.transform.rowcol(transform, px, py)
    series = [hydro[str(i)].to_numpy() for i in range(len(points))]
    return dict(tile_id=tile_id, dem=solver_dem, mask=solver_mask, friction=base_friction, transform=transform,
                seeds=seeds, sfincs=sfincs, score=score, cell_m2=cell_m2,
                cell_m=float(np.hypot(transform.a, transform.d)),
                weight=np.full(dem.shape, cell_m2 / 1e6), series=series,
                point_tree=cKDTree(np.column_stack([prow, pcol])),
                dt_s=float(np.median(np.diff(hydro.elapsed_hr))) * 3600.0)


def solve(t: dict, friction: np.ndarray) -> np.ndarray:
    rows, cols, values = t["seeds"]
    depth, _ = flood_depth_dense(t["dem"], t["mask"], friction.astype(np.float32), t["transform"],
                                 seed_rows=rows, seed_cols=cols, seed_values=values, **SOLVER)
    return depth


def score(t: dict, depth: np.ndarray) -> dict:
    sfincs_wet = t["score"] & np.isfinite(t["sfincs"]) & (t["sfincs"] > WET_THRESHOLD_M)
    # Stored calibration rasters are int16 cm; round alike so the threshold matches theirs.
    wet = np.round(depth * 100.0) / 100.0 > WET_THRESHOLD_M
    m, mo, so = confusion_counts(wet, sfincs_wet, t["score"], t["weight"])
    return {"matched_km2": m, "model_only_km2": mo, "sfincs_only_km2": so}


def run_tile(tile_id: str, out_dir: str, save_maps: bool) -> list[dict]:
    t0 = time.perf_counter()
    t = load_tile(tile_id)
    if t["seeds"] is None:
        return [{"tile_id": tile_id, "variant": "no_seeds"}]
    sources = np.zeros(t["dem"].shape, dtype=bool)
    sources[t["seeds"][0], t["seeds"][1]] = True
    fill = spill_level(t["dem"], sources | (t["mask"] == rcm.OCEAN_CODE))

    def hydro_cell(r, c):
        return t["series"][int(t["point_tree"].query([r, c])[1])]

    def hydro_cells(rows, cols):
        return t["series"], t["point_tree"].query(np.column_stack([rows, cols]))[1]

    land = t["score"]
    records = []
    maps = {"sfincs": t["sfincs"]}

    def add(variant: str, param: float, depth: np.ndarray, **extra) -> None:
        records.append({"tile_id": tile_id, "variant": variant, "param": param, **score(t, depth), **extra})

    for fsf in LU_FSF:
        base = solve(t, t["friction"] * fsf)
        add("landcover", fsf, base, mean_m_per_km=float((t["friction"][land] * fsf).mean() / t["cell_m"] * 1000)
            if land.any() else np.nan)
        capped, basins = basin_volume_cap(base, t["dem"], fill, hydro_cell, t["dt_s"], t["cell_m2"], t["cell_m"])
        add("A_volume_cap", fsf, capped, n_basins=len(basins), n_capped=sum(b["capped"] for b in basins),
            n_edge=sum(b["touches_edge"] for b in basins))
        if fsf == 12.0:
            maps["landcover_fsf12"], maps["A_fsf12"] = base, capped
            mean_cost = float(t["friction"][land].mean() * fsf) if land.any() else float(t["friction"].mean() * fsf)
            add("uniform_equal_mean", mean_cost / t["cell_m"] * 1000, solve(t, np.full(t["dem"].shape, mean_cost)))
            with open(Path(out_dir) / "basins.jsonl", "a", encoding="utf-8") as f:
                for b in basins:
                    f.write(json.dumps({"tile_id": tile_id, **b}) + "\n")
        if fsf in B_FSF:
            for alpha in B_ALPHA:
                mult = duration_friction_multiplier(base, t["dem"], fill, hydro_cells, t["dt_s"], B_T_REF_S, alpha)
                b_depth = solve(t, t["friction"] * fsf * mult)
                add(f"B_duration_a{alpha:g}", fsf, b_depth)
                if fsf == 12.0 and alpha == 1.0:
                    maps["B_fsf12_a1"] = b_depth
    for rate in UNIFORM_M_PER_KM:
        depth = solve(t, np.full(t["dem"].shape, rate * t["cell_m"] / 1000))
        add("uniform", rate, depth)
        if rate == 0.2:
            maps["uniform_0.2"] = depth
    if save_maps:
        k = MAP_TILES_DOWNSAMPLE
        np.savez_compressed(Path(out_dir) / "maps" / f"{tile_id}.npz",
                            dem=t["dem"][::k, ::k], score=land[::k, ::k], fill=fill[::k, ::k],
                            **{n: a[::k, ::k].astype(np.float16) for n, a in maps.items()})
    for r in records:
        r["elapsed_s_tile"] = round(time.perf_counter() - t0, 1)
    return records


def run_tile_extended(tile_id: str, out_dir: str, save_maps: bool) -> list[dict]:
    """Second stage: B at lower fsf, T_ref sensitivity, and B on uniform friction."""
    t0 = time.perf_counter()
    t = load_tile(tile_id)
    if t["seeds"] is None:
        return [{"tile_id": tile_id, "variant": "no_seeds"}]
    sources = np.zeros(t["dem"].shape, dtype=bool)
    sources[t["seeds"][0], t["seeds"][1]] = True
    fill = spill_level(t["dem"], sources | (t["mask"] == rcm.OCEAN_CODE))

    def hydro_cells(rows, cols):
        return t["series"], t["point_tree"].query(np.column_stack([rows, cols]))[1]

    records = []

    def add(variant: str, param: float, depth: np.ndarray) -> None:
        records.append({"tile_id": tile_id, "variant": variant, "param": param, **score(t, depth)})

    def with_b(friction: np.ndarray, alpha: float, t_ref_s: float = B_T_REF_S) -> np.ndarray:
        base = solve(t, friction)
        mult = duration_friction_multiplier(base, t["dem"], fill, hydro_cells, t["dt_s"], t_ref_s, alpha)
        return base, solve(t, friction * mult)

    for fsf in EXT_B_FSF:
        for alpha in B_ALPHA:
            base, b_depth = with_b(t["friction"] * fsf, alpha)
            add(f"B_duration_a{alpha:g}", fsf, b_depth)
        if fsf not in LU_FSF:
            add("landcover", fsf, base)
    for t_ref_h in EXT_T_REF_H:
        _, b_depth = with_b(t["friction"] * 9.0, 0.5, t_ref_h * 3600.0)
        add(f"B_duration_a0.5_tref{t_ref_h:g}h", 9.0, b_depth)
    for rate in EXT_B_UNIFORM_M_PER_KM:
        for alpha in B_ALPHA:
            _, b_depth = with_b(np.full(t["dem"].shape, rate * t["cell_m"] / 1000), alpha)
            add(f"B_uniform_a{alpha:g}", rate, b_depth)
    for r in records:
        r["elapsed_s_tile"] = round(time.perf_counter() - t0, 1)
    return records


def run_tile_validation(tile_id: str, out_dir: str, save_maps: bool) -> list[dict]:
    """Out-of-sample stage: the candidate configurations (VALIDATION_CONFIGS) on fresh tiles."""
    t0 = time.perf_counter()
    t = load_tile(tile_id)
    if t["seeds"] is None:
        return [{"tile_id": tile_id, "variant": "no_seeds"}]
    sources = np.zeros(t["dem"].shape, dtype=bool)
    sources[t["seeds"][0], t["seeds"][1]] = True
    fill = spill_level(t["dem"], sources | (t["mask"] == rcm.OCEAN_CODE))

    def hydro_cells(rows, cols):
        return t["series"], t["point_tree"].query(np.column_stack([rows, cols]))[1]

    records = []
    base_depths = {}
    for variant, param, alpha in VALIDATION_CONFIGS:
        is_uniform = variant.startswith("uniform") or variant.startswith("B_uniform")
        if is_uniform:
            friction = np.full(t["dem"].shape, param * t["cell_m"] / 1000)
        else:
            friction = t["friction"] * param
        if (is_uniform, param) not in base_depths:
            base_depths[(is_uniform, param)] = solve(t, friction)
        depth = base_depths[(is_uniform, param)]
        if alpha is not None:
            mult = duration_friction_multiplier(depth, t["dem"], fill, hydro_cells, t["dt_s"], B_T_REF_S, alpha)
            depth = solve(t, friction * mult)
        records.append({"tile_id": tile_id, "variant": variant, "param": param, **score(t, depth),
                        "elapsed_s_tile": round(time.perf_counter() - t0, 1)})
    return records


# (variant, fsf or m/km, B alpha or None), chosen from the main and extended stages.
VALIDATION_CONFIGS = [
    ("landcover", 9.0, None),
    ("landcover", 12.0, None),
    ("uniform", 0.1, None),
    ("uniform", 0.15, None),
    ("B_duration_a0.5", 9.0, 0.5),
    ("B_duration_a1", 7.5, 1.0),
    ("B_uniform_a0.5", 0.1, 0.5),
    ("B_uniform_a1", 0.1, 1.0),
]

STAGES = {"main": (run_tile, "results.jsonl"), "extended": (run_tile_extended, "results_extended.jsonl"),
          "validation": (run_tile_validation, "results_validation.jsonl")}


def select_tiles(n_random: int, seed: int = 0) -> tuple[list[str], set[str]]:
    worst = pd.read_csv(BASE / "figures" / "worst_tiles_selection_fsf12.csv")
    id_col = "tile_id" if "tile_id" in worst.columns else worst.columns[0]
    worst_ids = {str(int(i)) for i in worst[id_col]}
    per_tile = pd.read_csv(BASE / "friction_sweep_per_tile_metrics.csv")
    pool = per_tile[(per_tile.friction_scale_factor == 12.0) & per_tile.CSI.notna()]
    pool = sorted({str(int(i)) for i in pool.tile_id} - worst_ids)
    rng = np.random.default_rng(seed)
    sample = list(rng.choice(pool, size=min(n_random, len(pool)), replace=False))
    return sorted(worst_ids) + sample, worst_ids


def select_validation_tiles(n: int, exclude: set[str], seed: int = 1) -> list[str]:
    per_tile = pd.read_csv(BASE / "friction_sweep_per_tile_metrics.csv")
    pool = per_tile[(per_tile.friction_scale_factor == 12.0) & per_tile.CSI.notna()]
    pool = sorted({str(int(i)) for i in pool.tile_id} - exclude)
    return list(np.random.default_rng(seed).choice(pool, size=min(n, len(pool)), replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-random", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tile-ids", nargs="+", default=None)
    parser.add_argument("--stage", choices=list(STAGES), default="main")
    args = parser.parse_args()
    run_fn, results_name = STAGES[args.stage]
    out_dir = Path(args.out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    tiles, worst = select_tiles(args.n_random)
    if args.stage == "validation":
        tiles, worst = select_validation_tiles(args.n_random, set(tiles)), set()
    if args.tile_ids:
        tiles = args.tile_ids
    results = out_dir / results_name
    done = set()
    if results.exists():
        done = {json.loads(line)["tile_id"] for line in results.read_text().splitlines() if line.strip()}
    todo = [t for t in tiles if t not in done]
    sample_name = "sample_validation.json" if args.stage == "validation" else "sample.json"
    (out_dir / sample_name).write_text(json.dumps({"tiles": tiles, "worst": sorted(worst)}, indent=1))
    print(f"{len(todo)} of {len(tiles)} tiles to run, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_fn, t, str(out_dir), t in worst): t for t in todo}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                recs = future.result()
            except Exception as exc:  # one bad tile must not stop the batch; recorded for the report
                recs = [{"tile_id": tile, "variant": "error", "error": repr(exc)}]
            with open(results, "a", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"done {tile}: {recs[-1].get('elapsed_s_tile', '?')} s", flush=True)


if __name__ == "__main__":
    main()
