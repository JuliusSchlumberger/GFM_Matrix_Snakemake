#!/usr/bin/env python3
"""Build the storm-tide product (RP levels + per-RP hydrographs at the COAST-RP stations) from raw inputs.

Entry point: run(config), via `python run_preparation.py storm_tide_product`; read when
boundary_conditions.storm_tide_source is gfm_product. Method: src/storm_tide_product.py and
boundary_test/INPUT_ANALYSIS.md §8.4-8.5. Raw inputs: GESLA-4.1, GTSM-ERA5 surge residual,
FES2022 and FES2014, COAST-RP (+ _ETC, _TC) (config `storm_tide_product`).

Writes into storm_tide_product.cache_dir (~30 GB; may live off the data root) copies of raw
inputs and what depends on them alone: FES2022 decompressed, tide bases, GTSM series at the
stations and gauges, GESLA records, FES constants and MHWS at the stations, the coastline nodes
and their FES constants; deterministic, kept across runs. Into
boundary_conditions.storm_tide_product_dir (~3 GB):
  gauges/       gauge table (ratio, class, tidal MHWS), TC donor shapes and the tide-correction
                gauges (tide_gauges.parquet: short harmonic fits of every coastal GESLA site)
  stations.csv, levels.parquet, hydrographs.npy, hydrograph_info.parquet, surge_models.npy
  (stp.model_vector per station), node_levels.parquet, and the two product files
  (boundaries.PRODUCT_*_FILENAME; the levels file holds the stations, then the coastline nodes).
force=True (default) recomputes everything but the cache; ~6 h on 4 workers, most of it the GTSM
extraction (552 monthly files read in full) and the per-station build.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from boundaries import PRODUCT_HYDROGRAPHS_FILENAME, PRODUCT_LEVELS_FILENAME  # noqa: E402
from coastline_nodes import coastline_nodes  # noqa: E402
from config_utils import get_data_catalog  # noqa: E402
from gesla import coastal_sites  # noqa: E402
from gtsm_series import extract_series, gtsm_stations  # noqa: E402
import storm_tide_product as stp  # noqa: E402
import tide_correction as tc  # noqa: E402
import tide_prediction as tp  # noqa: E402

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _Paths:
    def __init__(self, root: Path, cache: Path):
        self.root = root
        self.cache = cache
        self.fes2022 = self.cache / "fes2022"
        self.bases = {
            "fes2022": (self.cache / "fes2022_hourly_re.npy", self.cache / "fes2022_hourly_im.npy",
                        self.cache / "fes2022_nodal_re.npy", self.cache / "fes2022_nodal_im.npy"),
            "fes2014": (self.cache / "fes2014_hourly_re.npy", self.cache / "fes2014_hourly_im.npy",
                        self.cache / "fes2014_nodal_re.npy", self.cache / "fes2014_nodal_im.npy"),
        }
        self.gtsm_stations = self.cache / "gtsm_surge_stations"
        self.gtsm_gauges = self.cache / "gtsm_surge_gauges"
        self.records = self.cache / "gesla"
        self.gauges = root / "gauges"
        self.tide_gauges = self.gauges / "tide_gauges.parquet"
        self.stations = root / "stations.csv"
        self.constants = self.cache / "fes_constants.npz"
        self.mhws = self.cache / "fes_mhws.npz"
        self.levels = root / "levels.parquet"
        self.hydrographs = root / "hydrographs.npy"
        self.info = root / "hydrograph_info.parquet"
        self.coastline_nodes = self.cache / "coastline_nodes.parquet"
        self.node_constants = self.cache / "node_fes_constants.npz"
        self.surge_models = root / "surge_models.npy"
        self.node_levels = root / "node_levels.parquet"


def _bases(models: dict, paths: _Paths, atlases) -> None:
    for atlas in atlases:
        h_re, h_im, n_re, n_im = paths.bases[atlas]
        tp.cached_basis(models[atlas], tp.hourly_times(), h_re, h_im)
        # FES2014's 10-min basis only feeds MHWS and the high-water peaks: float32 is ample
        tp.cached_basis(models[atlas], tp.nodal_times(), n_re, n_im, np.float64 if atlas == "fes2022" else np.float32)


def gauge_table(cfg: dict, paths: _Paths, coast_rp: Path, gtsm_lonlat: np.ndarray, model22, workers: int) -> pd.DataFrame:
    """Gauge ratios, station class and tidal MHWS of the GESLA gauges (gauges/gauges.parquet)."""
    out = paths.gauges / "gauges.parquet"
    if out.exists():
        return pd.read_parquet(out)
    paths.gauges.mkdir(parents=True, exist_ok=True)
    logger.info("GESLA sites and records")
    pairs = stp.pair_gauges(coastal_sites(Path(cfg["gesla_dir"])), gtsm_lonlat, coast_rp)
    records = stp.extract_gauge_records(Path(cfg["gesla_dir"]), pairs, paths.records, workers)
    pairs["good_years"] = records.good_years.reindex(pairs.file_name).to_numpy()
    pairs["station_class"] = stp.gauge_station_class(pairs, coast_rp.parent).to_numpy()
    pairs["calibrate"] = pairs.station_class != "tc_dominated"
    if not (paths.gtsm_gauges / "series.npy").exists():
        logger.info("GTSM surge at %d gauges", pairs.gtsm_index.nunique())
        extract_series(Path(cfg["gtsm_dir"]), pairs.gtsm_index.to_numpy(), paths.gtsm_gauges, workers)
    logger.info("Gauge tides: harmonic fits and FES2022 at the gauges")
    h_re, h_im, n_re, n_im = paths.bases["fes2022"]
    hc_gauge = stp.fit_gauge_tides(pairs, paths.records, (h_re, h_im), workers)
    hc_fes = tp.grouped_constants(model22, pairs[["lon", "lat"]].to_numpy(dtype=float))
    np.savez(paths.gauges / "gauge_constants.npz", file_name=pairs.file_name.to_numpy(dtype=str),
             gauge=hc_gauge, fes=hc_fes)
    seasonal = tp.seasonal_columns(model22)
    hg, hf = hc_gauge.copy(), hc_fes.copy()
    hg[:, seasonal] = hf[:, seasonal] = 0.0
    pairs["mhws_gauge_m"] = stp.nodal_mhws((n_re, n_im), hg, workers)
    pairs["mhws_fes2022_m"] = stp.nodal_mhws((n_re, n_im), hf, workers)
    logger.info("Gauge surge ratios")
    extremes = ((pairs.good_years >= stp.GAUGE_GOOD_YEARS) & (pairs.gtsm_dist_km <= stp.GAUGE_GTSM_KM)
                & np.isfinite(hg).all(axis=1) & np.isfinite(hf).all(axis=1)).to_numpy()
    sites = [{"file_name": f, "gtsm_index": int(g), "hc_gauge": hg[k], "hc_fes": hf[k]}
             for k, (f, g) in enumerate(zip(pairs.file_name, pairs.gtsm_index)) if extremes[k]]
    ratios = stp.gauge_ratios(sites, paths.records, paths.gtsm_gauges, (h_re, h_im), workers)
    table = pairs.set_index("file_name").join(ratios)
    logger.info("TC donor shapes")
    donors, shapes = stp.tc_donors(pairs, hc_gauge, paths.records, (h_re, h_im), workers)
    donors.to_parquet(paths.gauges / "tc_donors.parquet")
    np.save(paths.gauges / "tc_donor_shapes.npy", shapes)
    table.to_parquet(out)
    logger.info("%d gauges, %d with a ratio (%d calibrate), %d TC donors", len(table),
                int(table.ratio_fes.notna().sum()), int((table.calibrate & table.ratio_fes.notna()).sum()), len(donors))
    return table


def tide_gauge_table(cfg: dict, paths: _Paths, models: dict, workers: int) -> pd.DataFrame:
    """Short harmonic fits of the coastal GESLA sites and their tide ratio to both FES atlases."""
    if paths.tide_gauges.exists():
        return pd.read_parquet(paths.tide_gauges)
    paths.gauges.mkdir(parents=True, exist_ok=True)
    logger.info("Tide-correction gauges: harmonic fits of the coastal GESLA sites, FES at them")
    gt = tc.gauge_table(Path(cfg["gesla_dir"]), models, workers)
    gt.to_parquet(paths.tide_gauges)
    logger.info("%d tide gauges, %d usable", len(gt), int(gt.usable.sum()))
    return gt


def station_atlas(paths: _Paths, stations: pd.DataFrame, models: dict, gauges: pd.DataFrame,
                  tide_gauges: pd.DataFrame, workers: int) -> dict:
    """FES constants (Sa/Ssa zeroed), MHWS of both atlases, the atlas choice and the tide factor per station.

    The factor (tide_correction) scales the chosen atlas's constants; MHWS scales with it.
    """
    lonlat = stations[["lon", "lat"]].to_numpy(dtype=float)
    if not paths.constants.exists():
        logger.info("FES2022 and FES2014 constants at %d stations", len(stations))
        np.savez(paths.constants, fes2022=tp.grouped_constants(models["fes2022"], lonlat),
                 fes2014=tp.grouped_constants(models["fes2014"], lonlat))
    c = np.load(paths.constants)
    hc = {a: tp.fill_masked(c[a], lonlat) for a in ("fes2022", "fes2014")}
    for a in hc:
        hc[a][:, tp.seasonal_columns(models[a])] = 0.0
    if not paths.mhws.exists():
        logger.info("MHWS of both atlases")
        np.savez(paths.mhws, **{a: stp.nodal_mhws(paths.bases[a][2:], hc[a], workers) for a in hc})
    mh = np.load(paths.mhws)
    ok = stp.fes2022_ok_gauges(gauges.reset_index(), gauges.mhws_gauge_m.to_numpy(), gauges.mhws_fes2022_m.to_numpy())
    use14 = tp.consistent_version_flag(mh["fes2022"], mh["fes2014"], lonlat, ok)
    logger.info("FES2014 at %d stations", int(use14.sum()))
    glonlat = tide_gauges[["lon", "lat"]].to_numpy(dtype=float)
    factor, count = np.ones(len(stations)), np.zeros(len(stations), dtype=int)
    for a, sel in (("fes2014", use14), ("fes2022", ~use14)):
        factor[sel], count[sel] = tc.correction_factors(lonlat[sel], glonlat, tide_gauges[f"log_ratio_{a}"].to_numpy())
    logger.info("Tide corrected at %d stations (factor p10/p50/p90 %s)", int((count > 0).sum()),
                np.round(np.quantile(factor[count > 0], [0.1, 0.5, 0.9]), 2) if (count > 0).any() else "-")
    return {"hc": hc, "mhws": {a: mh[a] for a in hc}, "use14": use14, "tide_factor": factor, "tide_gauges": count}


def node_levels(config: dict, paths: _Paths, models: dict, todo: list[dict], lev: pd.DataFrame,
                tide_gauges: pd.DataFrame, workers: int) -> pd.DataFrame:
    """RP levels at the coastline nodes from the stations' surge models (`stp.build_nodes`).

    A node takes the atlas of its nearest station and the surge models of the
    simulation.flooding knn nearest product stations, weights distance^-idw_power, as the seeds
    weight the stations. Its TC levels are the same weighting over those of the neighbours with a
    TC part, if the nearest has one. Its tide is corrected by the gauges around the node itself.
    """
    if not paths.coastline_nodes.exists():
        catalog = get_data_catalog(_REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"])
        mask_dir = Path(catalog.get_source("deltadtm_mask").path).parent
        logger.info("Coastline nodes from the DeltaDTM masks in %s", mask_dir)
        coastline_nodes(mask_dir, config["tile_generation"]["ocean_code"], config["tile_generation"]["river_code"],
                        workers).to_parquet(paths.coastline_nodes)
    nodes = pd.read_parquet(paths.coastline_nodes)
    lonlat = nodes[["lon", "lat"]].to_numpy(dtype=float)
    if not paths.node_constants.exists():
        logger.info("FES2022 and FES2014 constants at %d coastline nodes", len(nodes))
        np.savez(paths.node_constants, fes2022=tp.grouped_constants(models["fes2022"], lonlat),
                 fes2014=tp.grouped_constants(models["fes2014"], lonlat))
    c = np.load(paths.node_constants)
    st_lonlat = lev[["lon", "lat"]].to_numpy(dtype=float)
    nearest_station, _ = stp.nearest(st_lonlat, lonlat)
    atlas = np.array([todo[k]["atlas"] for k in nearest_station])
    vectors = np.load(paths.surge_models, mmap_mode="r")
    usable = np.flatnonzero((lev.level_source == "product").to_numpy() & np.isfinite(vectors[:, 0]))
    flooding = config["simulation"]["flooding"]
    d, i = BallTree(np.radians(st_lonlat[usable][:, ::-1]), metric="haversine").query(
        np.radians(lonlat[:, ::-1]), k=int(flooding["knn"]))
    neighbours = usable[i]
    w = np.maximum(d * stp.EARTH_RADIUS_KM, 1e-3) ** -float(flooding["idw_power"])
    w /= w.sum(axis=1, keepdims=True)
    tc_used = np.full((len(todo), len(stp.RPS)), np.nan)
    for k, st in enumerate(todo):
        if st["tc"] is not None:
            tc_used[k] = st["tc"] * st.get("tc_scale", 1.0) + st.get("tc_shift_m", 0.0)
    hc = np.empty((len(nodes), c["fes2022"].shape[1]), dtype=complex)
    factor, count = np.ones(len(nodes)), np.zeros(len(nodes), dtype=int)
    glonlat = tide_gauges[["lon", "lat"]].to_numpy(dtype=float)
    for a in ("fes2022", "fes2014"):
        sel = atlas == a
        if not sel.any():
            continue
        h = tp.fill_masked(c[a], lonlat)
        h[:, tp.seasonal_columns(models[a])] = 0.0
        factor[sel], count[sel] = tc.correction_factors(lonlat[sel], glonlat, tide_gauges[f"log_ratio_{a}"].to_numpy())
        hc[sel] = h[sel] * factor[sel, None]
    todo_nodes = []
    for j in range(len(nodes)):
        tc_nb = tc_used[neighbours[j]]
        has_tc = np.isfinite(tc_nb[:, 0])
        tc_node = (w[j, has_tc] @ tc_nb[has_tc]) / w[j, has_tc].sum() if has_tc[0] else None
        todo_nodes.append({"node": j, "atlas": atlas[j], "hc": hc[j], "neighbours": neighbours[j], "weights": w[j],
                           "tc": tc_node})
    logger.info("Levels at %d coastline nodes (%d tide-corrected)", len(nodes), int((count > 0).sum()))
    out = nodes.join(stp.build_nodes(todo_nodes, paths.surge_models, paths.bases, workers))
    out["atlas"], out["tide_factor"], out["tide_gauges"] = atlas, factor, count
    out["nearest_station_km"] = d[:, 0] * stp.EARTH_RADIUS_KM
    return out


def run(config: dict, force: bool = True) -> None:
    """Write the storm-tide product files (and intermediates) into storm_tide_product_dir."""
    cfg = config["storm_tide_product"]
    workers = int(cfg["workers"])
    paths = _Paths(Path(config["boundary_conditions"]["storm_tide_product_dir"]), Path(cfg["cache_dir"]))
    paths.cache.mkdir(parents=True, exist_ok=True)
    catalog = get_data_catalog(_REPO_ROOT / config["paths"]["hydromt_data_catalog"], root=config["paths"]["root"])
    coast_rp = Path(catalog.get_source("coast_rp").path)
    if force:
        for p in [paths.gauges / n for n in ("gauges.parquet", "gauge_constants.npz", "tc_donors.parquet",
                                             "tc_donor_shapes.npy", "tide_gauges.parquet")] + [
                paths.stations, paths.levels, paths.hydrographs, paths.info, paths.surge_models, paths.node_levels,
                paths.root / PRODUCT_LEVELS_FILENAME, paths.root / PRODUCT_HYDROGRAPHS_FILENAME]:
            if p.exists():
                p.unlink()
                logger.info("  Deleted %s (force=True)", p.name)

    tp.ensure_local_fes(Path(cfg["fes2022_dir"]), paths.fes2022)
    models = {"fes2022": tp.fes2022_model(paths.fes2022),
              "fes2014": tp.fes2014_model(Path(catalog.get_source("fes2014_ocean_tide").path))}
    _bases(models, paths, ("fes2022", "fes2014"))
    gtsm_lonlat, _ = gtsm_stations(Path(cfg["gtsm_dir"]))

    gauges = gauge_table(cfg, paths, coast_rp, gtsm_lonlat, models["fes2022"], workers)

    if not paths.stations.exists():
        stp.match_stations(coast_rp, gtsm_lonlat).to_csv(paths.stations)
    stations = pd.read_csv(paths.stations, index_col="station")
    if not (paths.gtsm_stations / "series.npy").exists():
        logger.info("GTSM surge at %d stations", stations.gtsm_index.nunique())
        extract_series(Path(cfg["gtsm_dir"]), stations.gtsm_index.to_numpy(), paths.gtsm_stations, workers)

    tide_gauges = tide_gauge_table(cfg, paths, models, workers)
    atlas = station_atlas(paths, stations, models, gauges, tide_gauges, workers)
    ratios = stp.station_ratios(stations, gauges)
    tc_levels = stp.coast_rp_levels(coast_rp.parent / "COAST-RP_TC.nc")
    tc_levels = tc_levels[np.isfinite(tc_levels).all(axis=1)]
    donors = pd.read_parquet(paths.gauges / "tc_donors.parquet")
    tc_shapes = stp.station_tc_shapes(stations, donors, np.load(paths.gauges / "tc_donor_shapes.npy"))
    coast = stp.coast_rp_levels(coast_rp).to_numpy()
    todo = []
    for k, (st, s) in enumerate(stations.iterrows()):
        a = "fes2014" if atlas["use14"][k] else "fes2022"
        f = atlas["tide_factor"][k]
        todo.append({"station": st, "gtsm_index": int(s.gtsm_index), "atlas": a, "coast_rp": coast[k],
                     "hc": atlas["hc"][a][k] * f, "mhws_m": float(atlas["mhws"][a][k] * f),
                     "r": float(ratios.r.iloc[k]),
                     "tc": tc_levels.loc[s.station_id].to_numpy() if s.station_id in tc_levels.index else None,
                     "tc_shape": tc_shapes.get(st)})

    logger.info("Levels and hydrographs at %d stations", len(todo))
    hyd = np.lib.format.open_memmap(paths.hydrographs, mode="w+", dtype=np.float32,
                                    shape=(len(stations), len(stp.RPS), stp.N_STEPS))
    hyd[:] = np.nan
    vectors = np.lib.format.open_memmap(paths.surge_models, mode="w+", dtype=np.float32,
                                        shape=(len(stations), len(stp.BLEND_QUANTILES) + len(stp.MODEL_FIELDS)))
    vectors[:] = np.nan
    rows, info = stp.build_stations(todo, paths.gtsm_stations, paths.bases, hyd, workers, vectors)
    hyd.flush()
    vectors.flush()
    del vectors
    pd.DataFrame(info).to_parquet(paths.info)
    # station_id is not unique in COAST-RP (offshore "currents_*" points), so its rows join by position
    etc = stp.coast_rp_levels(coast_rp.parent / "COAST-RP_ETC.nc")
    lev = stations.join(ratios).join(pd.DataFrame(rows).set_index("station"))
    lev["mhws_fes2022_m"], lev["mhws_fes2014_m"] = atlas["mhws"]["fes2022"], atlas["mhws"]["fes2014"]
    lev["tide_factor"], lev["tide_gauges"] = atlas["tide_factor"], atlas["tide_gauges"]
    for j, rp in enumerate(stp.RPS):
        lev[f"coast_rp_rp{rp}_m"] = coast[:, j]
        lev[f"coast_rp_etc_rp{rp}_m"] = etc[rp].to_numpy()
        lev[f"coast_rp_tc_rp{rp}_m"] = tc_levels[rp].reindex(lev.station_id).to_numpy()
    lev.to_parquet(paths.levels)
    nodes = node_levels(config, paths, models, todo, lev, tide_gauges, workers)
    nodes.to_parquet(paths.node_levels)
    stp.export_levels(lev, coast_rp, paths.root / PRODUCT_LEVELS_FILENAME, nodes)
    stp.export_hydrographs(lev, np.load(paths.hydrographs, mmap_mode="r"), paths.root / PRODUCT_HYDROGRAPHS_FILENAME)
    logger.info("Storm-tide product written to %s", paths.root)


if __name__ == "__main__":
    sys.exit(
        "prepare_storm_tide_product.py is not a standalone entry point.\n"
        "Run it via: python run_preparation.py storm_tide_product"
    )
