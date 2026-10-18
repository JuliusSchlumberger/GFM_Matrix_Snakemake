"""Parity between sfincs_tiles/ copies and their src/ originals.

sfincs_tiles/ runs in the hydromt-sfincs-dev env (hydromt 1.x), which cannot import src/
(src/config_utils.py needs hydromt 0.9's `hydromt.log.setuplog`), so it keeps its own small
copies of a few src/ helpers. This test pins them to the originals so they cannot drift:

- `sfincs_tiles/flood_agreement.confusion_counts` / `metrics_from_counts` vs
  `src/validation.confusion_counts` / `metrics_from_counts` (the SFINCS copy reports the hit
  rate as `HT`, src as `HR`; it omits `tn`/`EB`/`EB_ratio`).
- `sfincs_tiles/calibration_paths.decode_waterdepth_cm` (path -> NaN for nodata) vs
  `src/rasters.decode_waterdepth_cm` (array -> `EIKONAL_NODATA` for nodata).
- `sfincs_tiles/retry_io.retry_transient_io` vs `src/config_utils.retry_transient_io` (the
  SFINCS copy catches plain `OSError`; src catches a curated tuple that includes it).

Runs in `gfm` (both sides import there). Plain assert-based script; pytest collects it.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sfincs_tiles"))
import calibration_paths as sf_paths  # noqa: E402
import config_utils  # noqa: E402
import flood_agreement as sf_agree  # noqa: E402
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
    for counts in ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 3.0)):  # zero denominators -> NaN in both
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


def main() -> None:
    test_confusion_counts_and_metrics_match()
    test_waterdepth_decoding_matches()
    test_retry_semantics_match()
    print("\nAll sfincs_tiles/src parity checks passed.")


if __name__ == "__main__":
    main()
