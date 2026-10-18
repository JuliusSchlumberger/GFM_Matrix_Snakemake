"""Regression checks for src/tiles.py::build_chunk_grid.

The chunk grid is the one definition of chunk IDs shared by the Snakefile (local DAG),
snakemake_workflow/scripts/generate_hpc_postprocess_job.py (HPC) and
check_postprocess_progress.py - a change here silently renames every merged chunk, so the
naming, alignment and empty-cell rules are pinned down on synthetic tiles.

Plain assert-based script (this repo's tests/ convention): runs standalone with
`python tests/chunk_grid_validation/validate_chunk_grid.py`, and pytest collects it too.
"""

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tiles import build_chunk_grid  # noqa: E402


def _tiles(*boxes):
    return gpd.GeoDataFrame({"tile_id": list(range(len(boxes)))}, geometry=[box(*b) for b in boxes], crs="EPSG:4326")


def test_ids_name_south_west_corner_in_all_hemispheres():
    # One tile per quadrant around (0, 0), each inside a single 5-degree cell.
    grid = build_chunk_grid(_tiles((1, 1, 2, 2), (-4, 1, -3, 2), (1, -4, 2, -3), (-4, -4, -3, -3)), 5)
    assert sorted(grid["chunk_id"]) == ["N00E000", "N00W005", "S05E000", "S05W005"], grid["chunk_id"].tolist()
    print("PASS: chunk IDs name the south-west corner (N/S lat 2 digits, E/W lon 3 digits)")


def test_cells_align_to_chunk_size_and_bounds_match_geometry():
    grid = build_chunk_grid(_tiles((31.2, -9.7, 33.9, -8.1)), 5)
    assert grid["chunk_id"].tolist() == ["S10E030"]
    assert [float(v) for v in grid["bounds"].iloc[0]] == [30.0, -10.0, 35.0, -5.0]
    assert grid.geometry.iloc[0].equals(box(30, -10, 35, -5))
    assert grid.crs == "EPSG:4326"
    print("PASS: cells snap to multiples of chunk_size_deg; bounds == geometry box")


def test_cells_without_tiles_are_skipped():
    # Two tiles 20 degrees apart: the bbox spans 5 cells along x, only the 2 occupied ones are kept.
    grid = build_chunk_grid(_tiles((0.5, 0.5, 1.0, 1.0), (20.5, 0.5, 21.0, 1.0)), 5)
    assert grid["chunk_id"].tolist() == ["N00E000", "N00E020"], grid["chunk_id"].tolist()
    print("PASS: cells that intersect no tile are dropped")


def test_tile_spanning_cells_yields_every_cell_in_x_then_y_order():
    grid = build_chunk_grid(_tiles((4, 4, 6, 6)), 5)
    assert grid["chunk_id"].tolist() == ["N00E000", "N05E000", "N00E005", "N05E005"], grid["chunk_id"].tolist()
    print("PASS: a tile crossing cell edges appears in every touched cell, ordered x then y")


def main():
    test_ids_name_south_west_corner_in_all_hemispheres()
    test_cells_align_to_chunk_size_and_bounds_match_geometry()
    test_cells_without_tiles_are_skipped()
    test_tile_spanning_cells_yields_every_cell_in_x_then_y_order()
    print("\nAll chunk grid checks passed.")


if __name__ == "__main__":
    main()
