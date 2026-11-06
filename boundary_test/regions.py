"""IPCC AR6 reference regions (Iturbide et al., 2020) and their continent for coastal sites.

A site takes the nearest AR6 land region within LAND_KM (gauges sit on the coast, often just
outside the land polygons), otherwise the AR6 ocean region that contains it (remote islands).
The region file is `IPCC-WGI-reference-regions-v4.geojson` from
https://github.com/IPCC-WG1/Atlas/tree/main/reference-regions, unpacked into AR6_FILE.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
AR6_FILE = REPO / ".cleanup_baseline" / "ipcc_ar6_regions" / "IPCC-WGI-reference-regions-v4.geojson"
LAND_KM = 200.0
EQUAL_AREA_CRS = 8857  # Equal Earth, metres


def ar6_regions(lon, lat) -> pd.DataFrame:
    """AR6 acronym, name and continent per site, in the input order."""
    regions = gpd.read_file(AR6_FILE)
    pts = gpd.GeoDataFrame({"i": np.arange(len(lon))}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
    land = regions[regions.Type.str.contains("Land")][["Acronym", "Name", "Continent", "geometry"]].to_crs(EQUAL_AREA_CRS)
    near = gpd.sjoin_nearest(pts.to_crs(EQUAL_AREA_CRS), land, distance_col="dist_m")
    near = near[~near.i.duplicated()].set_index("i")
    ocean = regions[regions.Type == "Ocean"][["Acronym", "Name", "Continent", "geometry"]]
    inside = gpd.sjoin(pts, ocean, predicate="within")
    inside = inside[~inside.i.duplicated()].set_index("i")
    cols = ["region", "region_name", "continent"]
    out = pd.DataFrame(index=np.arange(len(lon)), columns=cols, dtype=object)
    use_land = near.dist_m <= LAND_KM * 1e3
    out.loc[near.index[use_land], cols] = near.loc[use_land, ["Acronym", "Name", "Continent"]].to_numpy()
    rest = out.region.isna() & out.index.isin(inside.index)
    out.loc[rest, cols] = inside.loc[out.index[rest], ["Acronym", "Name", "Continent"]].to_numpy()
    rest = out.region.isna()
    out.loc[rest, cols] = near.loc[out.index[rest], ["Acronym", "Name", "Continent"]].to_numpy()
    return out
