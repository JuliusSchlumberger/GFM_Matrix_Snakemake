"""Test the depth-tolerance pooling and figure on synthetic per-tile depth-difference histograms."""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))
from calibration_paths import DEPTH_ACCURACY_NAME  # noqa: E402
from calibration_results import RESULT_NAME  # noqa: E402
from flood_agreement import CELL_KINDS, DIFF_EDGES, SFINCS_CLASS_LABELS  # noqa: E402
from plot_depth_accuracy import (  # noqa: E402
    ALIGNMENT_BANDS_M, _hist_quantiles, alignment_table, load_tolerance_hists, plot_depth_alignment, plot_depth_tolerance,
    tolerance_table,
)

RUNS = {"eikonal_uniform_cap": "eikonal_uniform_cap_rate0.1", "bathtub_connected": "bathtub_connected"}


def _hist(error_m: float, area_km2: float, kind: int = 0, cls: int = 2) -> np.ndarray:
    h = np.zeros((len(CELL_KINDS), len(SFINCS_CLASS_LABELS), len(DIFF_EDGES) + 1))
    h[kind, cls, np.searchsorted(DIFF_EDGES, error_m, side="right")] = area_km2
    return h


def _tiles(base: Path, finished: bool = True) -> None:
    # Tile 1: the eikonal is 5 cm off on 9 km2 and misses 1 km2 by 0.8 m; the bathtub is 0.6 m too deep.
    for tile, scored in ((1, True), (2, finished)):
        out = base / str(tile) / "outputs"
        out.mkdir(parents=True)
        eik = _hist(0.05, 9.0) + _hist(-0.8, 1.0, kind=1)
        bt = _hist(0.6, 10.0)
        np.savez(out / DEPTH_ACCURACY_NAME, **{f"{RUNS['eikonal_uniform_cap']}__area": eik,
                                               f"{RUNS['eikonal_uniform_cap']}__pop": eik * 100,
                                               f"{RUNS['bathtub_connected']}__area": bt,
                                               f"{RUNS['bathtub_connected']}__pop": bt * 100})
        if scored:
            (out / RESULT_NAME).write_text(json.dumps({"tile_id": str(tile)}))


def test_tolerance_shares_count_missed_cells_and_skip_unscored_tiles(tmp_path) -> None:
    _tiles(tmp_path, finished=False)
    pooled, per_tile, _ = load_tolerance_hists(tmp_path, RUNS)
    assert per_tile["tile_id"].nunique() == 1
    table = tolerance_table(pooled).set_index(["variant", "weight", "sfincs_class_m"])
    eik = table.loc[("eikonal_uniform_cap", "area", "all")]
    assert np.isclose(eik["within_0.1m"], 0.9) and np.isclose(eik["within_0.5m"], 0.9)
    assert np.isclose(table.loc[("bathtub_connected", "area", "all"), "within_0.5m"], 0.0)
    h = (pooled["eikonal_uniform_cap"]["area"][0] + pooled["eikonal_uniform_cap"]["area"][1]).sum(axis=0)
    q5, q50 = _hist_quantiles(h, (0.05, 0.5))
    assert np.isclose(q5, -0.795) and np.isclose(q50, 0.055)


def test_depth_tolerance_figure_renders(tmp_path) -> None:
    _tiles(tmp_path / "base")
    pooled, per_tile, per_class = load_tolerance_hists(tmp_path / "base", RUNS)
    labels = {"eikonal_uniform_cap": "Eik. uniform + cap (rate 0.1)", "bathtub_connected": "Connected bathtub"}
    plot_depth_tolerance(pooled, per_tile, labels, tmp_path / "depth_tolerance.png")
    assert (tmp_path / "depth_tolerance.png").exists() and (tmp_path / "depth_tolerance.txt").exists()
    table = alignment_table(pooled, per_class)
    for _, g in table.dropna(subset=["pooled_share"]).groupby(["variant", "sfincs_class_m"]):
        assert (np.diff(g.sort_values("band_m")["pooled_share"].to_numpy()) >= 0).all()  # wider band, more area
    assert set(table["band_m"]) == set(ALIGNMENT_BANDS_M)
    q = table.dropna(subset=["tile_p50"])
    assert len(q) and (q["tile_p5"] <= q["tile_p50"]).all() and (q["tile_p50"] <= q["tile_p95"]).all()
    plot_depth_alignment(table, labels, tmp_path / "depth_alignment.png")
    assert (tmp_path / "depth_alignment.txt").exists()
