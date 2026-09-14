"""Do the storm-surge polygons include the sea?  Geometric check on two AOIs.

National sum-of-areas for every scenario came out at ~101-104 thousand km2
(01/05 scripts), i.e. barely different between mean-high-water and the
1000-year surge.  That strongly suggests the polygons are "everything below
level X" INCLUDING the permanently-wet sea, not just the flooded land.

This script proves it properly: clip each layer to an exact AOI box, dissolve
(so overlaps don't double-count), and compute

    land_inundation = stormfloNNNar  MINUS  middelhoyvann (today's coastline)

Writes the derived land-only strips to ``norway_land_inundation_<aoi>.gpkg``.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import box

HERE = Path(__file__).parent
_T = Transformer.from_crs(4326, 25833, always_xy=True)

AOIS = {
    "oslofjord": (10.55, 59.80, 10.90, 59.98),
    "bergen": (5.20, 60.33, 5.42, 60.45),
}
SEA_LAYER = "middelhoyvann_klimaarna"
SURGE_LAYERS = ["stormflo20ar_klimaarna", "stormflo200ar_klimaarna", "stormflo1000ar_klimaarna"]


def clipbox(bb):
    x0, y0 = _T.transform(bb[0], bb[1])
    x1, y1 = _T.transform(bb[2], bb[3])
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def main() -> None:
    for aoi, bb in AOIS.items():
        gpkg = HERE / f"norway_aoi_{aoi}.gpkg"
        clip = clipbox(bb)
        aoi_km2 = clip.area / 1e6
        print("=" * 84)
        print(f"AOI {aoi}   lon/lat {bb}   box area {aoi_km2:,.1f} km2")
        print("=" * 84)

        sea = gpd.read_file(gpkg, layer=SEA_LAYER).clip(clip)
        sea_u = sea.union_all() if hasattr(sea, "union_all") else sea.unary_union
        print(f"  {SEA_LAYER:<28} sum={sea.area.sum() / 1e6:8.2f} km2   "
              f"dissolved={sea_u.area / 1e6:8.2f} km2  "
              f"({sea_u.area / 1e6 / aoi_km2 * 100:5.1f}% of AOI)")

        out = HERE / f"norway_land_inundation_{aoi}.gpkg"
        recs = [{"layer": "sea_middelhoyvann", "kind": "sea", "geometry": sea_u}]

        for layer in SURGE_LAYERS:
            g = gpd.read_file(gpkg, layer=layer).clip(clip)
            u = g.union_all() if hasattr(g, "union_all") else g.unary_union
            land = u.difference(sea_u)
            print(
                f"  {layer:<28} sum={g.area.sum() / 1e6:8.2f} km2   "
                f"dissolved={u.area / 1e6:8.2f} km2  "
                f"-> LAND ONLY {land.area / 1e6:7.3f} km2  "
                f"({land.area / u.area * 100:5.2f}% of the polygon, "
                f"{land.area / 1e6 / aoi_km2 * 100:5.2f}% of AOI)"
            )
            recs.append({"layer": layer, "kind": "land_only", "geometry": land})
            recs.append({"layer": layer, "kind": "full_polygon_incl_sea", "geometry": u})

        if out.exists():
            out.unlink()
        gpd.GeoDataFrame(recs, geometry="geometry", crs=25833).to_file(
            out, layer="comparison", driver="GPKG"
        )
        print(f"  -> wrote {out.name}\n")


if __name__ == "__main__":
    main()
