"""Synthetic-tile tests for sfincs_tiles/boundary_forcing_smoothed.py - each
builds a tiny fake SFINCS tile directory (sfincs.inp/.ind/.msk, matched
stations, COAST-HG hydrographs, tile bbox) with a known geometry and checks
the smoothed boundary forcing against the expected answer.

Covers: a uniform field stays exactly uniform; a single high spike is damped
but never exceeded; low/negative station peaks pass through unclipped; every
forcing value stays inside the station envelope; forcing points every <=25 km;
two separate boundary stretches each follow their own stations; a stretch
without own stations falls back to the nearest stations; stations outside
the bbox are ignored unless none is inside; a rotated grid's boundary cells
sit on the rotated edge; a one-cell boundary works; a run-window mismatch
raises; a gapped offshore line (hydromt's ~1 km gaps) stays one stretch; an
L-shaped, two-cell-wide boundary gets evenly spaced, non-duplicated points;
station_selection="all" uses stations outside the bbox.

Usage:
    python validate_boundary_forcing_smoothed.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "sfincs_tiles"))
import boundary_forcing_smoothed as bfs  # noqa: E402

EPSG = 32631
X0, Y0 = 500_000.0, 5_000_000.0
DX = 120.0
ELAPSED_HR = np.arange(0.0, 148.84, 1.0 / 6.0)
SHAPE = np.exp(-0.5 * ((ELAPSED_HR - 74.5) / 8.0) ** 2)  # unit storm peak at 74.5 h


def make_tile(root: Path, nmax: int, mmax: int, boundary_nm: list[tuple[int, int]],
              stations_xy: list[tuple[float, float]], peaks: list[float], rotation: float = 0.0,
              bbox_xy: tuple[float, float, float, float] | None = None) -> Path:
    """Fake tile: an all-active nmax x mmax grid with msk=2 at boundary_nm
    (n=row from y0, m=col), stations at model-CRS xy with hydrograph
    0.2 + (peak - 0.2) * SHAPE, and a tile bbox (model CRS; default = grid)."""
    tile = root / "9999"
    sm = tile / "sfincs_model"
    (sm / "gis").mkdir(parents=True)
    (tile / "inputs").mkdir()
    (sm / "sfincs.inp").write_text(
        f"mmax = {mmax}\nnmax = {nmax}\ndx = {DX}\ndy = {DX}\nx0 = {X0}\ny0 = {Y0}\n"
        f"rotation = {rotation}\nepsg = {EPSG}\n")
    flat = np.ones(mmax * nmax, dtype=np.uint8)
    for n, m in boundary_nm:
        flat[m * nmax + n] = 2  # column-major (mmax x nmax), as read_boundary_cells expects
    ind = np.concatenate([[mmax * nmax], np.arange(1, mmax * nmax + 1)]).astype("<u4")
    ind.tofile(sm / "sfincs.ind")
    flat.tofile(sm / "sfincs.msk")
    gpd.GeoDataFrame({"SLR_0": np.round(np.array(peaks) * 100).astype(int)},
                     geometry=gpd.points_from_xy(*zip(*stations_xy)), crs=EPSG).to_file(sm / "matched_boundary_points.gpkg")
    hydro = pd.DataFrame({"elapsed_hr": ELAPSED_HR})
    for k, pk in enumerate(peaks):
        hydro[str(k)] = 0.2 + (pk - 0.2) * SHAPE
    hydro.to_csv(sm / "corrected_hydrographs.csv", index=False)
    bb = bbox_xy or (X0, Y0, X0 + mmax * DX, Y0 + nmax * DX)
    gpd.GeoDataFrame(geometry=[box(*bb)], crs=EPSG).to_crs(4326).to_file(tile / "inputs" / "tile_geometry.gpkg")
    return tile


def west_line(nmax: int) -> list[tuple[int, int]]:
    return [(n, 0) for n in range(nmax)]


def peaks_of(f: dict) -> np.ndarray:
    return f["bzs"][:, 1:].max(axis=0)


def check_envelope(f: dict) -> None:
    H, S = f["station_hydrographs"], f["bzs"][:, 1:]
    assert np.isfinite(S).all(), "non-finite forcing value"
    assert (S <= H.max(axis=1, keepdims=True) + 1e-9).all(), "forcing above the highest station at some time"
    assert (S >= H.min(axis=1, keepdims=True) - 1e-9).all(), "forcing below the lowest station at some time"


def check_spacing(f: dict, spacing_km: float = 25.0) -> float:
    """Every point's nearest neighbour on its stretch <= spacing_km (network
    distance), every boundary cell within spacing_km / 2 of a point, no
    duplicate points. Returns the largest nearest-neighbour distance."""
    d = f["point_dist_km"]
    nn = np.where(np.isfinite(d), d, np.nan)
    multi = np.isfinite(d).any(axis=1)
    worst = float(np.nanmin(nn[multi], axis=1).max()) if multi.any() else 0.0
    assert worst <= spacing_km + 1.0, f"neighbouring forcing points {worst:.1f} km apart"
    assert f["network_coverage_km"].max() <= spacing_km / 2 + 1.0, f["network_coverage_km"].max()
    assert len(np.unique(np.round(f["bnd_xy"], 1), axis=0)) == len(f["bnd_xy"]), "duplicate forcing points"
    return worst


def run(name, fn) -> None:
    root = Path(tempfile.mkdtemp())
    try:
        print(f"=== {name} ===")
        fn(root)
        print("PASS\n")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_uniform_field_stays_uniform(root: Path) -> None:
    nmax = 1250  # 150 km line
    st = [(X0 + 3000, Y0 + y) for y in np.linspace(5_000, 145_000, 8)]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, 60, west_line(nmax), st, [1.3] * 8))
    check_envelope(f)
    assert np.allclose(f["bzs"][:, 1:], (0.2 + 1.1 * SHAPE)[(ELAPSED_HR >= 40) & (ELAPSED_HR <= 110)][:, None])
    worst = check_spacing(f)
    print(f"  {f['info']['n_points']} points on a 150 km line, all exactly 1.30 m, neighbours <= {worst:.1f} km apart")


def test_high_spike_damped_never_exceeded(root: Path) -> None:
    nmax = 1250
    ys = np.linspace(5_000, 145_000, 8)
    pk = [0.5] * 8
    pk[3] = 3.0
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, 60, west_line(nmax), [(X0 + 3000, Y0 + y) for y in ys], pk))
    check_envelope(f)
    p = peaks_of(f)
    assert 0.5 < p.max() < 3.0, p
    far = np.abs(f["bnd_xy"][:, 1] - (Y0 + ys[3])) / 1000 > 80  # real position: arc may start at either end
    assert np.allclose(p[far], 0.5, atol=0.02), p[far]
    print(f"  3.0 m spike among 0.5 m stations -> peaks {p.min():.2f}-{p.max():.2f} m, back to 0.50 m >80 km away")


def test_low_and_negative_peaks_pass_through(root: Path) -> None:
    nmax = 1250
    ys = np.linspace(5_000, 145_000, 6)
    pk = [-0.24, 0.06, 0.1, 0.5, 6.2, 5.8]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, 60, west_line(nmax), [(X0 + 3000, Y0 + y) for y in ys], pk))
    check_envelope(f)
    p = peaks_of(f)
    order = np.argsort(f["bnd_xy"][:, 1])  # south -> north, like the stations
    assert p[order][0] < 0.25 and p[order][-1] > 5.0, p[order]
    print(f"  stations -0.24..6.2 m -> forcing peaks {p.min():.2f}..{p.max():.2f} m along the line, within the envelope")


def test_two_separate_boundaries_follow_own_stations(root: Path) -> None:
    nmax, mmax = 834, 100  # 100 km x 12 km: west and east edges 12 km apart (> 5 km link)
    bnd = west_line(nmax) + [(n, mmax - 1) for n in range(nmax)]
    west = [(X0 + 2000, Y0 + y) for y in (20_000, 50_000, 80_000)]
    east = [(X0 + mmax * DX - 2000, Y0 + y) for y in (20_000, 50_000, 80_000)]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, mmax, bnd, west + east, [0.5] * 3 + [2.0] * 3))
    check_envelope(f)
    assert f["info"]["n_stretches"] == 2, f["info"]
    xw = f["bnd_xy"][:, 0] < X0 + mmax * DX / 2
    p = peaks_of(f)
    assert np.allclose(p[xw], 0.5) and np.allclose(p[~xw], 2.0), (p[xw], p[~xw])
    print(f"  2 stretches 12 km apart: west points {p[xw].min():.2f} m, east points {p[~xw].min():.2f} m (no mixing)")


def test_stretch_without_own_station_uses_nearest(root: Path) -> None:
    nmax, mmax = 834, 400  # west edge (stations) + a short south-east stretch 48 km away (no station)
    bnd = west_line(nmax) + [(0, m) for m in range(mmax - 40, mmax)]
    st = [(X0 + 2000, Y0 + 5_000), (X0 + 2000, Y0 + 95_000)]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, mmax, bnd, st, [0.4, 1.6]))
    check_envelope(f)
    se = f["bnd_xy"][:, 0] > X0 + mmax * DX / 2
    p = peaks_of(f)
    assert f["info"]["n_stretches"] == 2 and se.any()
    assert (p[se] < 1.0).all(), p[se]  # nearer to the 0.4 m station (south) than the 1.6 m one
    print(f"  stationless stretch takes nearest stations: {p[se].round(2).tolist()} m (south station 0.4, north 1.6)")


def test_outside_bbox_ignored_unless_none_inside(root: Path) -> None:
    nmax = 834
    inside = [(X0 + 2000, Y0 + y) for y in (20_000, 50_000, 80_000)]
    outside = [(X0 - 30_000, Y0 + 50_000)]  # west of the bbox, 5.0 m
    tile = make_tile(root, nmax, 60, west_line(nmax), inside + outside, [0.6, 0.6, 0.6, 5.0])
    f = bfs.build_smoothed_forcing(tile, station_selection="bbox")
    check_envelope(f)
    assert not f["info"]["used_all_matched_stations"] and f["info"]["n_stations_used"] == 3
    assert np.allclose(peaks_of(f), 0.6)
    shutil.rmtree(tile)
    tile = make_tile(root, nmax, 60, west_line(nmax), [(X0 - 30_000, Y0 + 20_000), (X0 - 30_000, Y0 + 80_000)], [0.8, 1.2])
    g = bfs.build_smoothed_forcing(tile, station_selection="bbox")
    check_envelope(g)
    assert g["info"]["used_all_matched_stations"] and g["info"]["n_stations_used"] == 2
    print(f"  5.0 m station outside bbox ignored (all points 0.60 m); none inside -> all {g['info']['n_stations_used']} "
          f"matched used, peaks {peaks_of(g).min():.2f}-{peaks_of(g).max():.2f} m")


def test_rotated_grid_boundary_on_rotated_edge(root: Path) -> None:
    nmax, rot = 834, 10.0
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, 60, west_line(nmax),
                                             [(X0 - 5000, Y0 + 50_000)], [0.9], rotation=rot,
                                             bbox_xy=(X0 - 20_000, Y0 - 1000, X0 + 10_000, Y0 + 101_000)))
    th = np.deg2rad(rot)
    # west edge cell centres: u = 0.5 dx, v = (n + 0.5) dy -> distance from the rotated edge line = 0.5 dx
    rel = f["network_xy"] - [X0, Y0]
    u = rel[:, 0] * np.cos(th) + rel[:, 1] * np.sin(th)
    assert np.allclose(u, 0.5 * DX), (u.min(), u.max())
    assert f["info"]["n_stretches"] == 1 and np.allclose(peaks_of(f), 0.9)
    print(f"  10 deg rotated grid: all {f['info']['n_boundary_cells']} boundary cells 60 m inside the rotated edge")


def test_gapped_line_is_one_stretch(root: Path) -> None:
    nmax = 1250  # 150 km west edge as 77-cell runs with ~1 km gaps (hydromt's offshore-edge pattern)
    bnd = [(n, 0) for n in range(nmax) if (n % 85) < 77]
    st = [(X0 + 3000, Y0 + y) for y in (10_000, 75_000, 140_000)]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, 60, bnd, st, [0.4, 0.8, 1.2]))
    check_envelope(f)
    worst = check_spacing(f)
    assert f["info"]["n_stretches"] == 1, f["info"]
    p, y = peaks_of(f), f["bnd_xy"][:, 1]
    assert np.all(np.diff(p[np.argsort(y)]) >= -1e-9), "peaks should rise monotonically south -> north"
    print(f"  gapped line -> 1 stretch, {f['info']['n_points']} points <= {worst:.1f} km apart, peaks rise "
          f"{p.min():.2f} -> {p.max():.2f} m monotonically")


def test_corner_and_wide_band(root: Path) -> None:
    nmax, mmax = 834, 834  # 100 x 100 km; boundary = west edge (2 cells wide) + south edge -> one L-shaped stretch
    bnd = [(n, 0) for n in range(nmax)] + [(n, 1) for n in range(nmax)] + [(0, m) for m in range(2, mmax)]
    st = [(X0 + 3000, Y0 + 90_000), (X0 + 90_000, Y0 + 3000)]
    f = bfs.build_smoothed_forcing(make_tile(root, nmax, mmax, bnd, st, [0.5, 1.5]))
    check_envelope(f)
    worst = check_spacing(f)
    assert f["info"]["n_stretches"] == 1, f["info"]
    p = peaks_of(f)
    north = f["bnd_xy"][:, 1] > Y0 + 80_000
    east = f["bnd_xy"][:, 0] > X0 + 80_000
    assert p[north].max() < 0.7 and p[east].min() > 1.3, (p[north], p[east])
    print(f"  L-shaped, 2-cell-wide boundary -> 1 stretch, {f['info']['n_points']} points <= {worst:.1f} km apart; "
          f"north end {p[north].max():.2f} m, east end {p[east].min():.2f} m (stations 0.5 / 1.5)")


def test_all_mode_uses_outside_station(root: Path) -> None:
    nmax = 834
    st = [(X0 + 2000, Y0 + 20_000), (X0 - 3000, Y0 + 95_000)]  # second one just outside the bbox, near the north end
    tile = make_tile(root, nmax, 60, west_line(nmax), st, [0.5, 1.5])
    fb = bfs.build_smoothed_forcing(tile, station_selection="bbox")
    fa = bfs.build_smoothed_forcing(tile, station_selection="all")
    check_envelope(fa)
    north_a = peaks_of(fa)[np.argmax(fa["bnd_xy"][:, 1])]
    assert np.allclose(peaks_of(fb), 0.5) and north_a > 1.3, (peaks_of(fb), north_a)
    print(f"  'bbox': all points 0.50 m (outside 1.5 m station ignored); 'all': north end {north_a:.2f} m")


def test_single_cell_boundary(root: Path) -> None:
    f = bfs.build_smoothed_forcing(make_tile(root, 50, 50, [(25, 0)], [(X0 + 1000, Y0 + 3000)], [1.1]))
    check_envelope(f)
    assert f["info"]["n_points"] == 1 and np.allclose(peaks_of(f), 1.1)
    print("  one boundary cell -> one forcing point, 1.10 m")


def test_run_window_mismatch_raises(root: Path) -> None:
    tile = make_tile(root, 200, 20, west_line(200), [(X0 + 1000, Y0 + 10_000)], [1.0])
    bad = tile / "sfincs_model" / "ref.bzs"
    np.savetxt(bad, np.column_stack([np.arange(10) * 600.0, np.zeros(10)]))
    try:
        bfs.build_smoothed_forcing(tile, reference_bzs=bad)
    except RuntimeError as e:
        print(f"  raised as expected: {e}")
        return
    raise AssertionError("no error for a reference bzs with a different time axis")


def main() -> None:
    for name, fn in [
        ("uniform field stays exactly uniform, spacing <= 25 km", test_uniform_field_stays_uniform),
        ("single high spike is damped, never exceeded", test_high_spike_damped_never_exceeded),
        ("low and negative peaks pass through unclipped", test_low_and_negative_peaks_pass_through),
        ("two separate boundaries follow their own stations", test_two_separate_boundaries_follow_own_stations),
        ("stretch without own station uses the nearest ones", test_stretch_without_own_station_uses_nearest),
        ("'bbox' mode: stations outside the bbox ignored unless none inside", test_outside_bbox_ignored_unless_none_inside),
        ("rotated grid: boundary cells on the rotated edge", test_rotated_grid_boundary_on_rotated_edge),
        ("gapped offshore line stays one stretch", test_gapped_line_is_one_stretch),
        ("L-shaped, two-cell-wide boundary", test_corner_and_wide_band),
        ("'all' mode uses stations outside the bbox", test_all_mode_uses_outside_station),
        ("single-cell boundary", test_single_cell_boundary),
        ("run-window mismatch raises", test_run_window_mismatch_raises),
    ]:
        run(name, fn)
    print("All boundary_forcing_smoothed validation checks passed.")


if __name__ == "__main__":
    main()
