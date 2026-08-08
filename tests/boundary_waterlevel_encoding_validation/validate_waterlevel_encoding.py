"""Validate the int16-centimetre encoding for boundary-forcing (COAST-RP)
water levels (src/rasters.py's encode_waterlevel_cm/decode_waterlevel_cm,
src/boundaries.py's save_boundary_points, 2026-08) - added so
boundaries.gpkg matches the same on-disk precision convention as
DEM/friction/waterdepth, since flood_model._idw_seed_values compares
station values directly against the (already int16-cm-encoded-then-
decoded) DEM.

Two levels: pure encode/decode round-trip logic (no I/O), and a real
save_boundary_points -> read-back-raw -> decode round trip through an
actual GeoPackage file (to catch anything the OGR/GeoPackage int field
write path itself might do differently than a plain numpy round-trip).

Usage:
    python validate_waterlevel_encoding.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

REPO_ROOT = Path(r"C:/Users/Schlu005/GFM")
sys.path.insert(0, str(REPO_ROOT / "snakemake_workflow" / "src"))

import boundaries as bnd  # noqa: E402
import rasters  # noqa: E402


def test_encode_decode_roundtrip() -> None:
    print("=== encode_waterlevel_cm / decode_waterlevel_cm: pure round-trip ===")
    values_m = np.array([0.0, 2.35, -1.20, 9.99, 0.001, -0.001], dtype=np.float64)
    encoded = rasters.encode_waterlevel_cm(values_m)
    assert encoded.dtype == np.int16, encoded.dtype
    assert list(encoded) == [0, 235, -120, 999, 0, 0], list(encoded)  # rounds to nearest cm
    decoded = rasters.decode_waterlevel_cm(encoded)
    assert np.allclose(decoded, np.round(values_m, 2), atol=1e-6), decoded
    print(f"PASS: {values_m} -> encoded {list(encoded)} (int16 cm) -> decoded {decoded} (matches, rounded to nearest cm)")
    print()


def test_encode_empty_array() -> None:
    print("=== encode_waterlevel_cm: empty array (dry/no-station tile placeholder) ===")
    empty = np.array([], dtype=np.float64)
    encoded = rasters.encode_waterlevel_cm(empty)
    assert encoded.dtype == np.int16, encoded.dtype
    assert len(encoded) == 0, encoded
    print("PASS: empty input -> empty int16 output, no crash (the min/max check that would "
          "raise on an empty array is correctly bypassed)")
    print()


def test_save_boundary_points_writes_int16_and_roundtrips() -> None:
    print("=== save_boundary_points: real GeoPackage write, int16 on disk, decodes correctly ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="waterlevel_encoding_test_"))
    try:
        stations = gpd.GeoDataFrame(
            {"SLR_0": [2.35, -1.20, 9.99]},
            geometry=[Point(10.0, 50.0), Point(11.0, 51.0), Point(12.0, 52.0)],
            crs="EPSG:4326",
        )
        out_path = tmpdir / "boundaries_test.gpkg"
        bnd.save_boundary_points(stations, out_path, column_name="SLR_0")

        raw = gpd.read_file(out_path)
        raw_values = raw["SLR_0"].to_numpy()
        assert np.issubdtype(raw_values.dtype, np.integer), raw_values.dtype
        assert list(raw_values) == [235, -120, 999], list(raw_values)
        print(f"PASS: on-disk raw values are integers in cm: {list(raw_values)}")

        decoded = rasters.decode_waterlevel_cm(raw_values.astype(np.int16))
        assert np.allclose(decoded, [2.35, -1.20, 9.99], atol=1e-6), decoded
        print(f"PASS: decoded back to metres: {decoded} (matches original, rounded to nearest cm)")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def test_save_boundary_points_empty_gdf() -> None:
    print("=== save_boundary_points: empty (dry-tile placeholder) GeoDataFrame ===")
    tmpdir = Path(tempfile.mkdtemp(prefix="waterlevel_encoding_test_empty_"))
    try:
        empty_stations = gpd.GeoDataFrame({"SLR_0": []}, geometry=[], crs="EPSG:4326")
        out_path = tmpdir / "boundaries_empty_test.gpkg"
        bnd.save_boundary_points(empty_stations, out_path, column_name="SLR_0")

        raw = gpd.read_file(out_path)
        assert len(raw) == 0, raw
        assert "SLR_0" in raw.columns, raw.columns
        print("PASS: empty placeholder writes without error, schema (SLR_0 column) survives the round trip")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    print()


def main() -> None:
    test_encode_decode_roundtrip()
    test_encode_empty_array()
    test_save_boundary_points_writes_int16_and_roundtrips()
    test_save_boundary_points_empty_gdf()
    print("All waterlevel-encoding validation checks passed.")


if __name__ == "__main__":
    main()
