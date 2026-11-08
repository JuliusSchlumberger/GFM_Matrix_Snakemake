#!/usr/bin/env python3
"""
Convert England's EA LIDAR Composite DTM (ODN via OSGM15) to GOCO06s, tile by tile.

This is the local-input validation plan's Section 3 output-product deliverable
(TODO.md): a GOCO06s-referenced DTM, a coarsened difference raster
(H_GOCO - H_ODN, for visual/numeric review only - see _DIFF_COARSEN_FACTOR) and
a README recording the formula and grid versions used - produced by converting
each of the EA 5km OS-grid tiles in place rather than materializing the whole
coastal strip as one in-memory raster.

Usage (repo pipeline):
  python run_preparation.py england_dtm_goco06s --config snakemake_workflow/config/config.yml

Usage (standalone):
  python preparation/convert_england_dtm_to_goco06s.py --config snakemake_workflow/config/config.yml

Requires: the England lidar_dem VRT (ea_coastal_dtm.py), the OSGM15 hybrid grid
alongside it, and the cached GOCO06s absolute raster
(vertical_datum_correction.goco06s_absolute_raster_path, rule
compute_goco06s_absolute_raster) to already exist.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from osgeo import gdal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402
from vertical_datum import compute_vertical_datum_offset  # noqa: E402

_FORMULA = (
    "h_ETRS89 = H_ODN + N_OSGM15 ; H_GOCO06s = h_ETRS89 - N_GOCO06s\n"
    "  => H_GOCO06s = H_ODN + (N_OSGM15 - N_GOCO06s)   (ETRS89 ~ ITRF vertically, no shift needed)"
)

# The offset is smooth at GOCO06s's own ~67km half-wavelength (d/o 300), so a
# full-resolution diff raster is pure storage cost with no extra reviewable
# signal - matches the land_use catalog source's own 100m grid.
_DIFF_COARSEN_FACTOR = 100


def _coarsen_mean(values: np.ndarray, valid: np.ndarray, factor: int) -> tuple[np.ndarray, np.ndarray]:
    """Block-mean `values` over valid cells only, factor x factor per output cell."""
    h, w = values.shape
    bh, bw = h // factor, w // factor
    v = values[: bh * factor, : bw * factor].reshape(bh, factor, bw, factor)
    m = valid[: bh * factor, : bw * factor].reshape(bh, factor, bw, factor)
    count = m.sum(axis=(1, 3))
    total = np.where(m, v, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return mean, count > 0


def _source_tiles(vrt_path: Path) -> list[str]:
    """Return the VRT's own per-tile source files (post no-coverage-tile fix, ea_coastal_dtm.py)."""
    ds = gdal.Open(str(vrt_path))
    tiles = sorted(fl for fl in ds.GetFileList() if fl is not None and Path(fl).suffix.lower() == ".tif")
    ds = None
    return tiles


def _convert_tile(
    tile_path: str, dtm_out: Path, diff_out: Path,
    geoid_offset_raster: Path, hybrid_grid_raster: Path, goco06s_absolute_raster: Path,
) -> tuple[float, float, float, int]:
    """Write the full-res corrected tile and a coarsened diff tile; return (min, max, sum, n) over valid cells."""
    with retry_transient_io(rasterio.open, tile_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
        offset = compute_vertical_datum_offset(
            "odn_osgm15", src.transform, src.crs, src.height, src.width,
            geoid_offset_raster=geoid_offset_raster,
            hybrid_grid_raster=hybrid_grid_raster,
            goco06s_absolute_raster=goco06s_absolute_raster,
        )

    valid = data != nodata
    corrected = np.where(valid, data + offset, nodata).astype(np.float32)

    profile.update(dtype="float32", compress="lzw", tiled=True, predictor=3)
    atomic_write(dtm_out, lambda f: f.write(corrected, 1), mode="w", opener=rasterio.open, **profile)

    coarse_diff, coarse_valid = _coarsen_mean(offset, valid, _DIFF_COARSEN_FACTOR)
    coarse_diff = np.where(coarse_valid, coarse_diff, nodata).astype(np.float32)
    diff_profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "width": coarse_diff.shape[1], "height": coarse_diff.shape[0],
        "crs": profile["crs"], "transform": profile["transform"] * Affine.scale(_DIFF_COARSEN_FACTOR),
        "nodata": nodata, "compress": "lzw",
    }
    atomic_write(diff_out, lambda f: f.write(coarse_diff, 1), mode="w", opener=rasterio.open, **diff_profile)

    if not valid.any():
        return (np.inf, -np.inf, 0.0, 0)
    valid_offset = offset[valid]
    return (float(valid_offset.min()), float(valid_offset.max()), float(valid_offset.sum()), int(valid.sum()))


def _write_readme(out_dir: Path, vrt_path: Path, hybrid_grid_raster: Path, goco06s_absolute_raster: Path,
                   n_tiles: int, n_skipped: int, diff_min: float, diff_max: float, diff_mean: float) -> None:
    readme = f"""# England DTM, GOCO06s-referenced

Converts `{vrt_path}` (ODN via OSGM15) to the GOCO06s geoid, per EA 5km OS-grid tile.

## Formula

{_FORMULA}

## Grids

- Source DTM: EA LIDAR Composite DTM 1m (2022), `dtm_1m_2022_coastal.vrt`
  ({n_tiles} tiles converted, {n_skipped} no-coverage tiles excluded - see
  `ea_coastal_dtm.py::mosaic`).
- Hybrid grid (N_OSGM15): `{hybrid_grid_raster.name}` (PROJ, `cdn.proj.org`).
- Target geoid (N_GOCO06s): `{goco06s_absolute_raster.name}`, computed from the
  GOCO06s `.gfc` coefficients cached in `inputs/ICGM/` (rule
  `compute_goco06s_absolute_raster`).

## Outputs

- `dtm/*.tif`: GOCO06s-referenced DTM, same grid/nodata as the source tile (1m).
- `diff/*.tif`: H_GOCO06s - H_ODN (the offset actually added), block-mean
  coarsened to {_DIFF_COARSEN_FACTOR}m - the offset is smooth at GOCO06s's own
  resolution, so this is for visual/numeric review only, not an input to
  anything downstream.
- `england_dtm_goco06s.vrt`, `england_dtm_goco06s_diff.vrt`: mosaics of the above.

## Sanity check

Diff raster range over all valid cells (computed at native 1m resolution,
before coarsening): min={diff_min:.3f} m, max={diff_max:.3f} m,
mean={diff_mean:.3f} m. Expected decimetre-to-~1m order with a smooth spatial
pattern (GOCO06s is d/o 300, ~67km half-wavelength - it lacks OSGM15's
short-wavelength signal; this is an expected property of the target geoid, not
a bug). Sign convention cross-checked against PROJ's own registered
EPSG:4937->5701 transform at 3 points (London, Liverpool, Portsmouth) - see
TODO.md.
"""
    atomic_write(out_dir / "README.md", lambda f: f.write(readme), encoding="utf-8")


def run(config: dict) -> None:
    """Convert every real-coverage England LIDAR tile to GOCO06s; build mosaics + README.

    Resumable: a tile whose dtm/diff output already exists is skipped.
    """
    root = Path(config["paths"]["root"])
    lidar_dir = root / "inputs" / "validation" / "England" / "new_inputs" / "lidar_dem"
    vrt_path = lidar_dir / "dtm_1m_2022_coastal.vrt"
    hybrid_grid_raster = lidar_dir / "uk_os_OSGM15_GB.tif"

    vc_cfg = config["vertical_datum_correction"]  # load_config already expanded {root} in these paths
    geoid_offset_raster = Path(vc_cfg["offset_raster_path"])
    goco06s_absolute_raster = Path(vc_cfg["goco06s_absolute_raster_path"])

    out_dir = lidar_dir.parent / "lidar_dem_goco06s"
    dtm_dir, diff_dir = out_dir / "dtm", out_dir / "diff"
    retry_transient_io(dtm_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io(diff_dir.mkdir, parents=True, exist_ok=True)

    tiles = _source_tiles(vrt_path)
    n_dropped = len(sorted(lidar_dir.glob("*.tif"))) - len(tiles)

    diff_min, diff_max, diff_sum, diff_n = np.inf, -np.inf, 0.0, 0
    n_converted, n_skipped = 0, 0
    for tile_path in tiles:
        name = Path(tile_path).name
        dtm_out, diff_out = dtm_dir / name, diff_dir / name
        if dtm_out.exists() and diff_out.exists():
            n_skipped += 1
            continue
        t_min, t_max, t_sum, t_n = _convert_tile(
            tile_path, dtm_out, diff_out,
            geoid_offset_raster, hybrid_grid_raster, goco06s_absolute_raster,
        )
        diff_min, diff_max = min(diff_min, t_min), max(diff_max, t_max)
        diff_sum += t_sum
        diff_n += t_n
        n_converted += 1

    print(f"{n_converted} tiles converted, {n_skipped} already done -> {out_dir}")

    gdal.BuildVRT(str(out_dir / "england_dtm_goco06s.vrt"), sorted(str(p) for p in dtm_dir.glob("*.tif")))
    gdal.BuildVRT(str(out_dir / "england_dtm_goco06s_diff.vrt"), sorted(str(p) for p in diff_dir.glob("*.tif")))

    diff_mean = diff_sum / diff_n if diff_n else float("nan")
    _write_readme(out_dir, vrt_path, hybrid_grid_raster, goco06s_absolute_raster,
                  len(tiles), n_dropped, diff_min, diff_max, diff_mean)
    print(f"diff range: min={diff_min:.3f} m, max={diff_max:.3f} m, mean={diff_mean:.3f} m")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="snakemake_workflow/config/config.yml")
    args = ap.parse_args()
    run(load_config(Path(args.config).resolve()))


if __name__ == "__main__":
    main()
