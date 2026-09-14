"""Step 1: locate Martinique's simulation tile(s) in the global tile manifest,
and resolve the "three islands" question for postprocessing merge chunk N10W065.

Outputs (written next to this script):
  tile_manifest_martinique.csv   - manifest rows intersecting Martinique / the FRJ benchmark
  merge_chunk_N10W065_tiles.csv  - every simulation tile intersecting the 5x5 deg merge chunk
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

OUT = Path(__file__).resolve().parent
ROOT = Path("P:/11212688-004-global-floodmaps/modelling")
MANIFEST = ROOT / "processed_inputs/mask/domain_tiles_global.gpkg"
BENCH = ROOT / "inputs/validation/FRA/n_inondable_03_02moy_s.shp"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

tiles = gpd.read_file(MANIFEST)
print(f"manifest: {MANIFEST}")
print(f"  {len(tiles)} features, crs={tiles.crs}")
print(f"  columns: {list(tiles.columns)}")

# --- benchmark: Martinique rows (id_tri starting FRJ) -------------------------
bench = gpd.read_file(BENCH)
print(f"\nbenchmark: {BENCH}\n  {len(bench)} features, crs={bench.crs}")
print(f"  columns: {list(bench.columns)}")
tri_col = [c for c in bench.columns if c.lower() == "id_tri"]
tri_col = tri_col[0] if tri_col else None
if tri_col:
    frj = bench[bench[tri_col].astype(str).str.startswith("FRJ")].copy()
else:
    frj = bench.iloc[0:0].copy()
print(f"  FRJ rows: {len(frj)}")
frj_ll = frj.to_crs(4326)
print(f"  FRJ bounds (EPSG:4326): {frj_ll.total_bounds}")
if tri_col:
    print(f"  distinct id_tri: {sorted(frj[tri_col].unique().tolist())}")
frj_ll.to_file(OUT / "benchmark_FRJ_martinique.gpkg", driver="GPKG")

# --- which simulation tile(s) cover Martinique --------------------------------
mart_box = box(-61.30, 14.35, -60.75, 14.95)   # generous Martinique island envelope
hits = tiles[tiles.intersects(mart_box)].copy()
hits["bounds"] = hits.geometry.apply(lambda g: tuple(round(v, 4) for v in g.bounds))
print("\n=== simulation tiles intersecting the Martinique island envelope ===")
print(hits.drop(columns="geometry").to_string())

bench_hits = tiles[tiles.intersects(frj_ll.geometry.buffer(0).unary_union)].copy()
print("\n=== simulation tiles intersecting the FRJ benchmark polygons ===")
print(bench_hits.drop(columns="geometry").to_string())

hits.drop(columns="geometry").to_csv(OUT / "tile_manifest_martinique.csv", index=False)
hits.drop(columns="bounds").to_file(OUT / "tiles_martinique.gpkg", driver="GPKG")

# --- merge chunk N10W065: which simulation tiles feed it ----------------------
# postprocessing.chunk_size_deg = 5; chunk id N10W065 -> lat 10..15 N, lon -65..-60
chunk = box(-65.0, 10.0, -60.0, 15.0)
chunk_tiles = tiles[tiles.intersects(chunk)].copy()
chunk_tiles["bounds"] = chunk_tiles.geometry.apply(lambda g: tuple(round(v, 4) for v in g.bounds))
print(f"\n=== merge chunk N10W065 (lon -65..-60, lat 10..15): "
      f"{len(chunk_tiles)} distinct simulation tiles intersect ===")
print(chunk_tiles.drop(columns="geometry").to_string())
chunk_tiles.drop(columns="geometry").to_csv(OUT / "merge_chunk_N10W065_tiles.csv", index=False)
chunk_tiles.drop(columns="bounds").to_file(OUT / "merge_chunk_N10W065_tiles.gpkg", driver="GPKG")

# --- Guadeloupe baseline tile -------------------------------------------------
gp_box = box(-61.85, 15.80, -61.00, 16.55)
gp_hits = tiles[tiles.intersects(gp_box)].copy()
print("\n=== simulation tiles intersecting Guadeloupe envelope (baseline) ===")
print(gp_hits.drop(columns="geometry").to_string())
gp_hits.drop(columns="geometry").to_csv(OUT / "tile_manifest_guadeloupe.csv", index=False)
