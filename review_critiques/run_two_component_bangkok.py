"""Re-solve the Bangkok / Chao Phraya tiles with two-component SLR propagation.

Reads the scenario's tile inputs (read-only) and writes waterdepth rasters under --out-dir,
hop-0 tiles before hop-1 per scenario; the attenuated reference is the scenario's own output.

Usage:
    python review_critiques/run_two_component_bangkok.py --out-dir <dir>
        [--return-periods RP2 RP100 RP1000] [--waterlevel-names SLR_0 ...] [--workers 2]
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from boundaries import static_station_levels  # noqa: E402
from config_utils import load_config, merged_waterlevel_names  # noqa: E402
from eikonal_runner import run_tile_scenario  # noqa: E402
from tiles import load_tile_grid  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml"


def _solve_scenario(config_path: str, out_dir: str, return_period: str, waterlevel_name: str) -> list[dict]:
    cfg = load_config(config_path)
    sim, bc = cfg["simulation"], cfg["boundary_conditions"]
    prep = sim.get("preprocessing_inputs_dir") or sim["model_outputs"]
    mdt_path = Path(cfg["paths"]["processed_inputs_dir"]) / "MDT_mapped_on_coastal_points.nc"
    stations = static_station_levels(
        bc["waterlevel_nc_dir"], bc["nc_filename_template"], bc["nc_variable_template"], mdt_path,
        return_period, waterlevel_name,
    )
    grid = load_tile_grid(cfg["tile_grid"]["path"]).sort_values("hop_distance")
    records = []
    for tile_id in grid["tile_id"].astype(int).astype(str):
        inputs = Path(prep) / tile_id / "inputs"
        scen = f"{return_period}_{waterlevel_name}"
        output_path = Path(out_dir) / tile_id / "results" / f"waterdepth_{scen}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        result = run_tile_scenario(
            tile_id, return_period, waterlevel_name,
            dem_path=str(inputs / "dem.tif"), mask_path=str(inputs / "mask.tif"),
            friction_path=str(inputs / "friction.tif"),
            boundaries_path=str(inputs / f"boundaries_{scen}.gpkg"),
            output_path=str(output_path), model_outputs=out_dir,
            # Neighbour dem/mask come from the scenario's inputs, neighbour depths from out_dir.
            preprocessing_inputs_dir=prep, tile_grid_path=cfg["tile_grid"]["path"],
            raster_config=cfg["raster_format"], flooding_config=sim["flooding"],
            ocean_code=cfg["tile_generation"]["ocean_code"], river_code=cfg["tile_generation"]["river_code"],
            static_stations=stations,
        )
        records.append({"tile_id": tile_id, "scenario": scen, "outcome": result["outcome"],
                        "elapsed_s": round(time.perf_counter() - start, 1),
                        "n_static_wet_cells": result.get("diagnostics", {}).get("n_static_wet_cells")})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--return-periods", nargs="+", default=["RP2", "RP100", "RP1000"])
    parser.add_argument("--waterlevel-names", nargs="+", default=None, help="default: all of the scenario")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    names = args.waterlevel_names or merged_waterlevel_names(cfg["boundary_conditions"], cfg["adaptation"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "solve_log.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(line)["scenario"] for line in log_path.read_text().splitlines() if line.strip()}
    jobs = [(rp, wl) for rp in args.return_periods for wl in names if f"{rp}_{wl}" not in done]
    print(f"{len(jobs)} scenarios x 9 tiles, {args.workers} workers -> {out_dir}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_solve_scenario, args.config, str(out_dir), rp, wl): (rp, wl) for rp, wl in jobs}
        for future in as_completed(futures):
            records = future.result()
            with open(log_path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")
            print(f"done {futures[future]}: {sum(r['elapsed_s'] for r in records):.0f} s", flush=True)


if __name__ == "__main__":
    main()
