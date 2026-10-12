"""Pre-flight check for the final SFINCS calibration: build only the SFINCS
grid + mask from a tile's boundary line (build_station_boundary_lines.py) -
the step that failed for 22 of the 573 bline runs - without elevation,
subgrid or forcing, so a tile is known to build before it is selected.

Runs under hydromt-sfincs-dev (it calls build_sfincs_tile._create_mask).
Writes {base}/boundary_lines/preflight/{tile_id}.json:
    {"tile_id", "ok", "n_active", "n_boundary", "error"}
ok = the mask builds, has active cells and >= MIN_BOUNDARY_CELLS waterlevel
boundary cells. Only tiles whose line status says suitable are checked
(the rest get ok=false, error="not suitable") unless --all.

Usage:
    python preflight_sfincs_mask.py --config <config.yml> --base-dir-name calibration_roughness_SFINCS \\
        --tiles-dir-name model_outputs --tile-ids-file tiles_000.txt [--skip-done]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import geopandas as gpd
from hydromt_sfincs import SfincsModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_sfincs_tile import MAIN_RES_M_DEFAULT, _create_mask  # noqa: E402
from gfm_config import read_root  # noqa: E402
from retry_io import retry_transient_io  # noqa: E402

MIN_BOUNDARY_CELLS = 1


def preflight_tile(tile_id: str, tiles_root: Path, lines_dir: Path, resolution_m: float) -> dict:
    row = {"tile_id": int(tile_id), "ok": False, "n_active": 0, "n_boundary": 0, "error": None}
    gpkg = lines_dir / f"{tile_id}.gpkg"
    tile_gdf = retry_transient_io(gpd.read_file, tiles_root / tile_id / "inputs" / "tile_geometry.gpkg")
    tmp = Path(tempfile.mkdtemp(prefix=f"preflight_{tile_id}_"))
    try:
        sf = SfincsModel(root=str(tmp), mode="w+")
        sf.grid.create_from_region(region={"geom": tile_gdf}, res=resolution_m, crs="utm", rotated=True)
        _create_mask(sf, tile_gdf, None, gpkg, resolution_m)
        msk = sf.grid.data["mask"]
        row["n_active"] = int((msk > 0).sum())
        row["n_boundary"] = int((msk == 2).sum())
        if row["n_active"] == 0:
            row["error"] = "no active cell"
        elif row["n_boundary"] < MIN_BOUNDARY_CELLS:
            row["error"] = "no waterlevel boundary cell"
        else:
            row["ok"] = True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", required=True)
    parser.add_argument("--tiles-dir-name", default=None, help="where the tile dirs live (default: --base-dir-name)")
    parser.add_argument("--tile-ids", nargs="*", default=[])
    parser.add_argument("--tile-ids-file", default=None)
    parser.add_argument("--resolution-m", type=float, default=MAIN_RES_M_DEFAULT)
    parser.add_argument("--all", action="store_true", help="also check tiles whose line is not suitable")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()

    root = read_root(Path(args.config))
    lines_dir = root / args.base_dir_name / "boundary_lines"
    tiles_root = root / (args.tiles_dir_name or args.base_dir_name)
    out_dir = lines_dir / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(args.tile_ids)
    if args.tile_ids_file:
        ids += [t.strip() for t in Path(args.tile_ids_file).read_text().split() if t.strip()]

    n_ok = 0
    for k, tile_id in enumerate(ids, 1):
        out = out_dir / f"{tile_id}.json"
        if args.skip_done and out.exists():
            n_ok += json.loads(out.read_text())["ok"]
            continue
        status_path = lines_dir / "status" / f"{tile_id}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if not args.all and not status.get("suitable", False):
            row = {"tile_id": int(tile_id), "ok": False, "n_active": 0, "n_boundary": 0,
                   "error": "not suitable" if status else "no line status"}
        else:
            try:
                row = preflight_tile(tile_id, tiles_root, lines_dir, args.resolution_m)
            except Exception as e:  # noqa: BLE001 - recorded per tile, the job goes on
                row = {"tile_id": int(tile_id), "ok": False, "n_active": 0, "n_boundary": 0,
                       "error": f"{type(e).__name__}: {e}"[:500]}
                traceback.print_exc()
        out.write_text(json.dumps(row))
        n_ok += row["ok"]
        print(f"  [{k}/{len(ids)}] tile {tile_id}: ok={row['ok']} active={row['n_active']} "
              f"bnd={row['n_boundary']}{'' if row['ok'] else ' - ' + str(row['error'])}", flush=True)
    print(f"preflight ok: {n_ok}/{len(ids)} -> {out_dir}")


if __name__ == "__main__":
    main()
