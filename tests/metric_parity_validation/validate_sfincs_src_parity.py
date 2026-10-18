"""Pin sfincs_tiles/ copies of src/ and preparation/ helpers to their originals (run in `gfm`).

sfincs_tiles/ (hydromt 1.x) cannot import src/ (needs hydromt 0.9). Known
differences: hit rate `HT` vs `HR`, no tn/EB; nodata NaN vs EIKONAL_NODATA;
retry catches plain OSError vs a tuple including it.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from affine import Affine

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sfincs_tiles"))
sys.path.insert(0, str(REPO / "preparation"))
import calibration_paths as sf_paths  # noqa: E402
import config_utils  # noqa: E402
import flood_agreement as sf_agree  # noqa: E402
import mdt as sf_mdt  # noqa: E402
import prepare_boundary_conditions as src_bc  # noqa: E402
import rasters  # noqa: E402
import retry_io as sf_retry  # noqa: E402
import validation as src_val  # noqa: E402
from merge import EIKONAL_NODATA  # noqa: E402


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def test_confusion_counts_and_metrics_match() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        shape = (40, 50)
        model, bench, dom = (rng.random(shape) < p for p in (0.3, 0.4, 0.8))
        w = rng.random(shape)
        tp, fp, fn, _tn = src_val.confusion_counts(model, bench, dom, w)
        assert sf_agree.confusion_counts(model, bench, dom, w) == (tp, fp, fn)
        s, f = src_val.metrics_from_counts(tp, fp, fn), sf_agree.metrics_from_counts(tp, fp, fn)
        for src_key, sf_key in (("HR", "HT"), ("FAR", "FAR"), ("CSI", "CSI"), ("bias", "bias")):
            assert _same(s[src_key], f[sf_key]), (src_key, s[src_key], f[sf_key])
    for counts in ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 3.0)):
        s, f = src_val.metrics_from_counts(*counts), sf_agree.metrics_from_counts(*counts)
        assert all(_same(s[a], f[b]) for a, b in (("HR", "HT"), ("FAR", "FAR"), ("CSI", "CSI"), ("bias", "bias")))
    print("PASS: confusion counts and HR(HT)/FAR/CSI/bias identical, incl. zero-denominator NaNs")


def test_waterdepth_decoding_matches() -> None:
    assert sf_paths.WATERDEPTH_NODATA_INT16 == rasters.WATERDEPTH_NODATA_INT16
    assert sf_paths.WATERDEPTH_SCALE == rasters.WATERDEPTH_SCALE
    depth_m = np.array([[0.0, 0.01, 1.23], [2.5, 0.5, 0.0]])
    enc = rasters.encode_waterdepth_cm(depth_m)
    enc[1, 2] = rasters.WATERDEPTH_NODATA_INT16
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "wd.tif"
        with rasterio.open(p, "w", driver="GTiff", dtype="int16", count=1, height=2, width=3,
                           transform=Affine(0.001, 0, 0, 0, -0.001, 0), crs="EPSG:4326",
                           nodata=rasters.WATERDEPTH_NODATA_INT16) as dst:
            dst.write(enc, 1)
        sf_depth = sf_paths.decode_waterdepth_cm(p)[0]
    src_depth = rasters.decode_waterdepth_cm(enc)
    nodata = enc == rasters.WATERDEPTH_NODATA_INT16
    assert np.isnan(sf_depth[nodata]).all() and (src_depth[nodata] == EIKONAL_NODATA).all()
    assert np.array_equal(sf_depth[~nodata], src_depth[~nodata])
    print("PASS: waterdepth decoding identical (nodata: NaN vs EIKONAL_NODATA, both non-wet)")


def test_retry_semantics_match() -> None:
    for impl in (sf_retry.retry_transient_io, config_utils.retry_transient_io):
        calls = []
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("transient")
            return "ok"
        assert impl(flaky, retries=4, delay_s=0) == "ok" and len(calls) == 3
        def always():
            raise FileNotFoundError("gone")
        try:
            impl(always, retries=2, delay_s=0)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("must re-raise after the last retry")
    assert OSError in config_utils._TRANSIENT_IO_EXCEPTIONS or any(
        issubclass(OSError, e) for e in config_utils._TRANSIENT_IO_EXCEPTIONS)
    print("PASS: both retry transient OSErrors and re-raise after the last attempt")


def test_mdt_lookup_matches() -> None:
    """sfincs_tiles/mdt.py vs preparation/prepare_boundary_conditions MDT load + lookup."""
    rng = np.random.default_rng(2)
    lat = np.arange(60.0, -61.0, -2.5)            # descending: _load_mdt must sort
    lon = np.arange(1.25, 360.0, 2.5)             # 0..360: _load_mdt must wrap to -180..180
    values = rng.normal(0.0, 0.5, (2, lat.size, lon.size))
    values[:, 10:20, 30:50] = np.nan              # land patch: nearest NaN -> fallback window
    values[:, 30:, 100:] = np.nan                 # large patch: no valid cell within fallback
    da = xr.DataArray(values, dims=("time", "latitude", "longitude"),
                      coords={"time": [0, 1], "latitude": lat, "longitude": lon}, name="mdt")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mdt.nc"
        da.to_dataset().to_netcdf(p)
        src_da = src_bc._load_mdt(p, "mdt")
        sf_da = sf_mdt._load_mdt(p, "mdt")
        xr.testing.assert_identical(src_da, sf_da)
        sf_lookup = sf_mdt.mdt_lookup_fn(p, "mdt", fallback_deg=3.0)
    pts = [(float(x), float(y)) for x, y in zip(rng.uniform(-180, 180, 400), rng.uniform(-62, 62, 400))]
    pts += [(float(src_da.longitude[c]), float(src_da.latitude[r])) for r, c in ((15, 40), (35, 120), (0, 0))]
    n_nan = n_fallback = 0
    for x, y in pts:
        s = src_bc._nearest_valid_grid(src_da, "longitude", x, "latitude", y, 3.0)
        assert _same(s, sf_mdt._nearest_valid_grid(sf_da, "longitude", x, "latitude", y, 3.0)), (x, y)
        assert _same(s, sf_lookup(x, y)), (x, y)
        n_nan += bool(np.isnan(s))
        n_fallback += bool(np.isnan(float(src_da.sel(longitude=x, latitude=y, method="nearest"))) and not np.isnan(s))
    assert n_nan and n_fallback, (n_nan, n_fallback)
    print(f"PASS: MDT load and nearest-valid lookup identical on {len(pts)} points "
          f"({n_fallback} via the fallback window, {n_nan} NaN)")


def main() -> None:
    test_confusion_counts_and_metrics_match()
    test_waterdepth_decoding_matches()
    test_retry_semantics_match()
    test_mdt_lookup_matches()
    print("\nAll sfincs_tiles/src parity checks passed.")


if __name__ == "__main__":
    main()
