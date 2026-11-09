"""Population- and area-weighted soft counts of the national-map scorers on synthetic rasters."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "validation"))
import validation as v  # noqa: E402
from exposure_validation_report import local_vs_global  # noqa: E402
from population_grid import cell_area_km2, population_per_pixel  # noqa: E402
from validate_sf_bay import _exposure_counts  # noqa: E402

RES = 30 / 3600
SHAPE = (200, 200)
BNG = from_origin(450_000, 310_000, 50, 50)  # 10 km x 10 km in the English Midlands


def _population_raster(path: Path, count: np.ndarray) -> Path:
    with rasterio.open(path, "w", driver="GTiff", height=180, width=240, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(-2.0, 53.5, RES, RES), nodata=-99999) as dst:
        dst.write(count.astype(np.float32), 1)
    return path


@pytest.fixture
def people(tmp_path) -> np.ndarray:
    count = np.random.default_rng(1).integers(0, 500, size=(180, 240))
    return population_per_pixel(_population_raster(tmp_path / "pop.tif", count), SHAPE, BNG, "EPSG:27700")[0]


def _synthetic_maps():
    rng = np.random.default_rng(0)
    depth = rng.gamma(0.5, 0.2, SHAPE)
    fraction = np.clip(rng.normal(0.5, 0.4, SHAPE), 0, 1)
    domain = rng.random(SHAPE) > 0.1
    return depth, fraction, domain


def test_uniform_density_gives_density_times_projected_pixel_area(tmp_path) -> None:
    lat_top = 53.5 - RES * np.arange(180)
    cell_km2 = np.array([cell_area_km2(t, t - RES, RES) for t in lat_top])[:, None] * np.ones((1, 240))
    path = _population_raster(tmp_path / "pop.tif", 1000.0 * cell_km2)
    people = population_per_pixel(path, SHAPE, BNG, "EPSG:27700")[0]
    assert people == pytest.approx(np.full(SHAPE, 1000.0 * 0.0025), rel=1e-4)


def test_exposure_counts_partition_the_weights(people) -> None:
    depth, fraction, domain = _synthetic_maps()
    area_km2 = abs(BNG.a * BNG.e) / 1e6
    model_wet, exposure_wet = depth > 0.01, depth > 0.10
    c = v.exposure_counts(model_wet, exposure_wet, fraction, domain, area_km2, people)

    bench_pop = float((fraction * people)[domain].sum())
    assert c["tp_pop"] + c["fn_pop"] == pytest.approx(bench_pop)
    assert c["tp_exp_pop"] + c["fn_exp_pop"] == pytest.approx(bench_pop)
    assert c["tp_pop"] + c["fp_pop"] == pytest.approx(float(people[domain & model_wet].sum()))
    assert c["tp_exp_pop"] + c["fp_exp_pop"] == pytest.approx(float(people[domain & exposure_wet].sum()))
    assert sum(c[f"{k}_pop"] for k in ("tp", "fp", "fn", "tn")) == pytest.approx(float(people[domain].sum()))
    assert sum(c[f"{k}_exp_km2"] for k in ("tp", "fp", "fn", "tn")) == pytest.approx(domain.sum() * area_km2)
    # The deeper threshold can only lose model-wet people.
    assert c["tp_exp_pop"] <= c["tp_pop"] and c["fp_exp_pop"] <= c["fp_pop"]

    cols = v.exposure_columns(c)
    tp, fp, fn = c["tp_pop"], c["fp_pop"], c["fn_pop"]
    assert cols["CSI_pop"] == pytest.approx(tp / (tp + fp + fn))
    assert cols["bias_pop"] == pytest.approx((tp + fp) / (tp + fn))
    assert set(cols) == {f"{m}_{w}" for w in v.EXPOSURE_WEIGHTS
                         for m in ("tp", "fp", "fn", "tn", "HR", "FAR", "CSI", "bias")}


def test_uniform_population_reproduces_cell_counts() -> None:
    depth, fraction, domain = _synthetic_maps()
    c = v.exposure_counts(depth > 0.01, depth > 0.10, fraction, domain, 0.0025, np.ones(SHAPE))
    cells = v.confusion_counts_soft(depth > 0.01, fraction, domain, np.ones(SHAPE))
    assert [c[f"{k}_pop"] for k in ("tp", "fp", "fn", "tn")] == pytest.approx(cells)


def test_depth_benchmark_is_thresholded_at_both_levels(people) -> None:
    rng = np.random.default_rng(2)
    model, bench = rng.gamma(0.5, 0.2, SHAPE), rng.gamma(0.5, 0.2, SHAPE)
    domain = np.ones(SHAPE, bool)
    c = _exposure_counts(model, bench, model > 0.01, bench > 0.01, domain, 0.0025, people, 0.10)
    assert c["tp_pop"] + c["fn_pop"] == pytest.approx(float(people[bench > 0.01].sum()))
    assert c["tp_exp_pop"] + c["fn_exp_pop"] == pytest.approx(float(people[bench > 0.10].sum()))
    assert c["tp_exp_pop"] == pytest.approx(float(people[(bench > 0.10) & (model > 0.10)].sum()))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_local_vs_global_table(tmp_path) -> None:
    key = {"benchmark_key": "fz3", "region": "england", "return_period": "RP100"}
    scores = {"CSI": 0.7, "CSI_pop": 0.6, "bias_pop": 0.9, "CSI_exp_km2": 0.65, "CSI_exp_pop": 0.55,
              "bias_exp_pop": 0.8}
    _write(tmp_path / "local" / "GBR" / "metrics_GBR_fz3_RP100_SLR_0.csv", [{**key, **scores}])
    _write(tmp_path / "global" / "GBR" / "metrics_GBR_RP100_SLR_0.csv",
           [{**key, **{k: 0.1 for k in scores}}, {**key, "region": "wales", **scores}])
    _write(tmp_path / "global" / "GBR" / "metrics_GBR_fz3_RP100_SLR_0.csv", [{**key, **{k: 0.5 for k in scores}}])
    t = local_vs_global([("england", tmp_path / "local", tmp_path / "global"),
                         ("solo", tmp_path / "local", None)])
    assert len(t) == 4  # one local row x two thresholds x two scenarios; wales has no local row
    eng = t[t.scenario == "england"].set_index("threshold")
    assert eng.loc["primary", "CSI_area_local"] == 0.7 and eng.loc["exposure", "CSI_pop_local"] == 0.55
    assert (eng["CSI_pop_global"] == 0.5).all()  # the standalone scorer's row wins
    assert t[t.scenario == "solo"]["CSI_pop_global"].isna().all()
