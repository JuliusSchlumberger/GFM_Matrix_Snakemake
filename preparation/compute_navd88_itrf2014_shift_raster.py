"""Cache the NAD83(2011)+NAVD88-height -> ITRF2014 shift as a GeoTIFF (one-time).

This is `navd88_geoid18`'s "hybrid grid" (src/vertical_datum.py): GEOID18 plus the
mandatory NAD83(2011)->ITRF2014 Helmert shift, applied by pyproj in a single
compound-CRS transform (EPSG:6318+5703 -> EPSG:7912) rather than downloaded as a
national grid file the way OSGM15 was for England/Wales. Verified empirically
(TODO.md): the shift is a pure additive function of (lon, lat) - independent of
the input height - so it is safe to evaluate at H=0 and cache as an ordinary
raster, sampled later exactly like OSGM15's via sample_geoid_offset(_at_points).

pyproj defaults to network OFF, under which PROJ silently falls back to a
"ballpark" operation that OMITS the GEOID18 term entirely (wrong by ~30+ m in
California) with only a UserWarning, no exception - this script hard-enables
the network and hard-verifies the high-accuracy (grid-based) operation was
actually selected before writing anything, never trusting that fallback silently.

Usage:
    python preparation/compute_navd88_itrf2014_shift_raster.py \\
        --bbox -122.64 37.40 -121.88 38.28 --out <path/to/navd88_itrf2014_shift.tif>
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vertical_datum import write_grid_raster  # noqa: E402

# EPSG:6318+5703 = NAD83(2011) geographic 2D + NAVD88 height (compound 3D source).
# EPSG:7912 = ITRF2014 geographic 3D (target). PROJ's best-accuracy registered
# operation between these chains GEOID18 (height shift) and the NAD83(2011)->
# ITRF2014 Helmert shift in one pipeline.
_SRC_CRS = "EPSG:6318+5703"
_DST_CRS = "EPSG:7912"
# The grid-based (GEOID18) operation reports ~0.015 m accuracy; every ballpark/
# NADCON5-chained fallback reports -1 (unknown/ballpark) or >=1 m - a wide,
# deliberate margin, not a tight threshold tuned to one observed value.
_MAX_TRUSTED_ACCURACY_M = 0.1


def _verify_high_accuracy_operation_available() -> None:
    from pyproj.transformer import TransformerGroup

    tg = TransformerGroup(_SRC_CRS, _DST_CRS)
    best = tg.transformers[0]
    if best.accuracy < 0 or best.accuracy > _MAX_TRUSTED_ACCURACY_M:
        raise RuntimeError(
            f"Best available {_SRC_CRS}->{_DST_CRS} operation has accuracy "
            f"{best.accuracy} m (description: {best.description!r}) - this is the "
            "'ballpark'/NADCON5-chained fallback that OMITS the real GEOID18 grid, "
            "not the real transform. Enable pyproj's network grid fetch "
            "(pyproj.network.set_network_enabled(True)) and ensure outbound access "
            "to PROJ's CDN, or install the proj-data conda package, before retrying."
        )
    print(f"Verified: best operation accuracy={best.accuracy} m ({best.description})")


def compute_shift_grid(
    bbox: tuple[float, float, float, float], resolution_deg: float, pad_deg: float,
) -> tuple[np.ndarray, Affine]:
    """Evaluate the NAD83(2011)+NAVD88(H=0) -> ITRF2014 height shift on a regular grid.

    Returns (values, transform): north-up float32 grid, EPSG:4326, covering
    bbox padded by pad_deg on every side.
    """
    import pyproj

    pyproj.network.set_network_enabled(True)
    _verify_high_accuracy_operation_available()
    from pyproj import Transformer

    transformer = Transformer.from_crs(_SRC_CRS, _DST_CRS, always_xy=True)

    lon_min, lat_min, lon_max, lat_max = bbox
    lon_min, lat_min, lon_max, lat_max = lon_min - pad_deg, lat_min - pad_deg, lon_max + pad_deg, lat_max + pad_deg
    lons = np.arange(lon_min, lon_max + resolution_deg, resolution_deg)
    lats = np.arange(lat_max, lat_min - resolution_deg, -resolution_deg)  # north-up: decreasing lat
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    _, _, z = transformer.transform(lon_grid.ravel(), lat_grid.ravel(), np.zeros(lon_grid.size))
    values = np.asarray(z, dtype=np.float32).reshape(lon_grid.shape)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite value(s) in the computed shift grid - investigate before caching")

    transform = Affine(resolution_deg, 0.0, lon_min, 0.0, -resolution_deg, lat_max)
    return values, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution-deg", type=float, default=0.005, help="~550m at this latitude")
    parser.add_argument("--pad-deg", type=float, default=0.1, help="coverage margin beyond --bbox")
    args = parser.parse_args()

    values, transform = compute_shift_grid(tuple(args.bbox), args.resolution_deg, args.pad_deg)
    print(f"shift range: [{values.min():.4f}, {values.max():.4f}] m, mean {values.mean():.4f} m, shape {values.shape}")
    write_grid_raster(values, transform, "EPSG:4326", args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
