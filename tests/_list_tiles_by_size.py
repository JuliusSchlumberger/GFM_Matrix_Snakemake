"""One-off helper: enumerate every tile under MODEL_OUTPUTS that has the
RP100_SLR_0 scenario inputs, read its exact dem.tif dimensions (header only,
no full read), and print tile,width,height,n_cells sorted ascending by
n_cells - used to build the calibration run's tile order (smallest first,
largest last).
"""
import sys
from pathlib import Path

import rasterio

MODEL_OUTPUTS = Path("D:/GFM/model_outputs")
RETURN_PERIOD = "RP100"
WATERLEVEL_NAME = "SLR_0"


def main():
    scenario = f"{RETURN_PERIOD}_{WATERLEVEL_NAME}"
    rows = []
    for tile_dir in sorted(MODEL_OUTPUTS.iterdir()):
        if not tile_dir.is_dir():
            continue
        inputs = tile_dir / "inputs"
        dem = inputs / "dem.tif"
        mask = inputs / "mask.tif"
        friction = inputs / "friction.tif"
        toml_f = inputs / f"aqueduct_{scenario}.toml"
        gpkg_f = inputs / f"boundaries_{scenario}.gpkg"
        if not (dem.exists() and mask.exists() and friction.exists() and toml_f.exists() and gpkg_f.exists()):
            continue
        try:
            with rasterio.open(dem) as src:
                w, h = src.width, src.height
        except Exception as exc:
            print(f"SKIP {tile_dir.name}: {exc}", file=sys.stderr)
            continue
        rows.append((tile_dir.name, w, h, w * h))

    rows.sort(key=lambda r: r[3])
    for name, w, h, n in rows:
        print(f"{name},{w},{h},{n}")
    print(f"\nTOTAL={len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
