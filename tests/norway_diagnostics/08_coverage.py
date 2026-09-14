"""Profile `dekningsomrade` - the dataset's own declared coverage/survey area.

Analogous to Spain's "only surveyed coastal stretches" and France's TRI-zones-only
caveat: this layer states where Kartverket actually mapped, which is what the
validation eval-domain should be restricted to.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely

from nor_dump import block_for, iter_rows, load_index

HERE = Path(__file__).parent
IDX = load_index()
DUMP = Path(IDX["dump"])

rows = list(iter_rows(DUMP, block_for(IDX, "dekningsomrade")))
df = pd.DataFrame(rows)
gdf = gpd.GeoDataFrame(
    df.drop(columns=["omrade"]),
    geometry=shapely.from_wkb([bytes.fromhex(h) for h in df["omrade"]]),
    crs="EPSG:25833",
)

print(f"dekningsomrade: {len(gdf):,} rows")
print(f"  geom types : {sorted(gdf.geom_type.unique())}")
print(f"  bbox 4326  : {[round(v, 3) for v in gdf.to_crs(4326).total_bounds]}")
print(f"  total area : {gdf.area.sum() / 1e6:,.0f} km2")
_u = gdf.union_all() if hasattr(gdf, "union_all") else gdf.unary_union
print(f"  dissolved  : {_u.area / 1e6:,.0f} km2")
print("\n  dekningstatus value counts (area km2):")
print(
    gdf.assign(km2=gdf.area / 1e6)
    .groupby("dekningstatus")
    .agg(features=("km2", "size"), km2=("km2", "sum"))
    .to_string()
)

out = HERE / "norway_dekningsomrade.gpkg"
gdf.to_file(out, layer="dekningsomrade", driver="GPKG")
print(f"\nwrote {out}  ({out.stat().st_size / 1024**2:.1f} MB)")
