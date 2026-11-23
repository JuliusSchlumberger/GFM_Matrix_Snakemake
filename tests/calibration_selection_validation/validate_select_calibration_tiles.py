"""Test the calibration tile selection's antimeridian flag and the failed-tile replacement."""

import sys
from argparse import Namespace
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sfincs_tiles"))
from select_calibration_tiles import (  # noqa: E402
    GRID_DEG, TILE_GRID, cmd_replace, stratified_sample_exact, touches_antimeridian,
)


def test_antimeridian_flag_catches_tiles_stored_past_180_and_wrapped_ones() -> None:
    bounds = pd.DataFrame({"minx": [170.0, 178.3, -180.0, -179.0, -179.5],
                           "maxx": [172.0, 180.2, -178.0, 179.0, -178.0]})
    assert touches_antimeridian(bounds).tolist() == [False, True, True, True, False]


def _selection(tmp_path: Path) -> tuple[Path, Path, list[int]]:
    # 40 eligible tiles in two 15 deg bins; the bin at 165-180 E holds tiles reaching 180.
    lon = [150.5 + i * 0.3 for i in range(20)] + [165.5 + i * 0.75 for i in range(20)]
    tiles = gpd.GeoDataFrame({"tile_id": range(100, 140), "hop_distance": 0},
                             geometry=[box(x - 0.5, 60.0, x + 0.5, 61.0) for x in lon], crs="EPSG:4326")
    root, base = tmp_path / "root", tmp_path / "root" / "cal"
    (root / Path(TILE_GRID).parent).mkdir(parents=True)
    (base / "candidates").mkdir(parents=True)
    tiles.to_file(root / TILE_GRID, driver="GPKG")
    funnel = pd.DataFrame({"tile_id": tiles["tile_id"], "lon": lon, "lat": 60.5, "category": "eligible"})
    selected = stratified_sample_exact(funnel, 10, GRID_DEG, 0)
    funnel.loc[funnel["tile_id"].isin(selected["tile_id"]), "category"] = "selected"
    funnel.to_csv(base / "candidates" / "candidate_funnel.csv", index=False)
    selected.to_csv(base / "tile_selection_metadata.csv", index=False)
    (base / "tile_ids.txt").write_text("".join(f"{t}\n" for t in selected["tile_id"]))
    tiles.merge(funnel[["tile_id", "category"]]).to_file(base / "tile_selection.gpkg", layer="tiles", driver="GPKG")
    return root, base, [int(t) for t in selected["tile_id"]]


def test_replace_swaps_a_failed_tile_within_its_bin_and_keeps_the_rest(tmp_path) -> None:
    root, base, ids = _selection(tmp_path)
    failed = next(t for t in ids if t >= 120)
    cmd_replace(Namespace(tile_ids=[str(failed)], seed=0), root, base)
    new = [int(t) for t in (base / "tile_ids.txt").read_text().split()]
    (added,) = set(new) - set(ids)
    assert set(ids) - set(new) == {failed} and len(new) == len(ids)
    assert 120 <= added < 140                                 # same bin
    assert 165.5 + (added - 120) * 0.75 + 0.5 < 180.0        # tile 139 reaches 180.25 E
    funnel = pd.read_csv(base / "candidates" / "candidate_funnel.csv").set_index("tile_id")["category"]
    assert funnel[failed] == "run_failed" and funnel[added] == "selected"
    assert sorted(pd.read_csv(base / "tile_selection_metadata.csv")["tile_id"]) == sorted(new)

    cmd_replace(Namespace(tile_ids=[str(added)], seed=0), root, base)   # a second round still reproduces
    again = {int(t) for t in (base / "tile_ids.txt").read_text().split()}
    assert failed not in again and added not in again and len(again) == len(ids)
