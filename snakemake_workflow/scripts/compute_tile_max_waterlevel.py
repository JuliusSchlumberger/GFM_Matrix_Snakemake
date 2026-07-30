"""Compute the highest boundary water level across every scenario for a single tile.

Used by crop_flood_extent to size ONE shared flood-candidate crop window
that stays valid for every (return_period, waterlevel_name) combination this
tile will be run for - see src/flood_extent.py for why a single, shared
maximum is safe to use across all of them.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import retry_transient_io  # noqa: E402

boundary_paths = snakemake.input.boundaries  # noqa: F821
# Parallel list, same length/order as boundary_paths (see the rule: both are
# built with expand() over the identical (return_period, waterlevel_name)
# combinations), giving the boundary value column name for each file - see
# extract_boundaries.py, which writes each file with column_name=waterlevel_name.
waterlevel_names = snakemake.params.waterlevel_names  # noqa: F821

max_waterlevel = None
for path, column in zip(boundary_paths, waterlevel_names):
    stations = retry_transient_io(gpd.read_file, path)
    if stations.empty:
        continue
    scenario_max = float(stations[column].max())
    max_waterlevel = scenario_max if max_waterlevel is None else max(max_waterlevel, scenario_max)

with open(snakemake.output.max_waterlevel, "w", encoding="utf-8") as f:  # noqa: F821
    json.dump({"max_waterlevel": max_waterlevel}, f)
