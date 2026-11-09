"""National-map exposure scores of the exposure/depth assessment (stage 3.1): tables and figures.

Reads validate_country.py's metrics CSVs with the exposure columns (area at the exposure
threshold, population at both thresholds), checks the area counts against a reference run's
CSVs, and draws X8 (area vs population scores per region) and X9 (one region's agreement map,
shaded by population; agreement at the primary threshold as validate_country writes it).
--local-vs-global instead tabulates local- against global-input runs (T4: area CSI, population
CSI and population bias at both thresholds), from any scorer's metrics CSVs with those columns.

Usage:
    python validation/exposure_validation_report.py --config <config.yml> --metrics-dir <dir> \\
        --reference-dir <saved validation dir> --out-dir <dir> [--map-region GBR:firth_of_forth:RP100]
    python validation/exposure_validation_report.py --out-dir <dir> \\
        --local-vs-global england <local validation dir> <global validation dir> [--local-vs-global ...]
"""

from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatchcase as fnmatch
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from config_utils import get_data_catalog, load_config  # noqa: E402
from population_grid import population_per_pixel  # noqa: E402
from map_style import AGREE_COLOR, A_ONLY_COLOR, B_ONLY_COLOR, LAND_COLOR, draw_panel_letter  # noqa: E402

SCORES = (("primary, area", "CSI", "bias"), ("exposure, area", "CSI_exp_km2", "bias_exp_km2"),
          ("primary, population", "CSI_pop", "bias_pop"), ("exposure, population", "CSI_exp_pop", "bias_exp_pop"))


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(metrics_dir.glob("*/metrics_???_RP*_SLR_0.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def replication(metrics: pd.DataFrame, reference_dir: Path) -> pd.DataFrame:
    """Area counts of this run against the reference run, per region (relative difference)."""
    ref = load_metrics(reference_dir)
    keys = ["iso", "benchmark_key", "region", "return_period"]
    m = metrics.merge(ref, on=keys, suffixes=("", "_ref"))
    out = m[keys].copy()
    for c in ("tp_km2", "fp_km2", "fn_km2"):
        out[f"{c}_rel_diff"] = (m[c] - m[f"{c}_ref"]) / m[f"{c}_ref"].where(m[f"{c}_ref"] != 0)
    return out


# (threshold, area CSI, population CSI, population bias) columns of a metrics CSV.
LOCAL_VS_GLOBAL = (("primary", "CSI", "CSI_pop", "bias_pop"),
                   ("exposure", "CSI_exp_km2", "CSI_exp_pop", "bias_exp_pop"))
_KEYS = ["benchmark_key", "region", "return_period"]


def _read_scored_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "HT" in df.columns:  # validate_sf_bay.py: one row per grid, indexed by grid name
        df = df.rename(columns={df.columns[0]: "grid"})
        df = df[df.grid == "A_coarsen_to_model_grid"].assign(
            benchmark_key="usgs_cosmos_depth", region="model_grid", return_period="RP100")
    return df


def load_scenario_metrics(validation_dir: Path) -> pd.DataFrame:
    """Every scored row under a validation dir; a standalone scorer's row wins over validate_country's."""
    paths = [*validation_dir.glob("metrics_*.csv"), *validation_dir.glob("*/metrics_*.csv")]
    paths.sort(key=lambda p: (not fnmatch(p.name, "metrics_???_RP*"), p.as_posix()))
    frames = [_read_scored_rows(p) for p in paths]
    if not frames:
        return pd.DataFrame(columns=_KEYS)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(_KEYS, keep="last")


def local_vs_global(pairs: list[tuple[str, Path, Path | None]]) -> pd.DataFrame:
    """Area CSI, population CSI and population bias of local vs global input, per threshold.

    One row per scenario, local-input (benchmark, region, RP) and threshold; the global
    twin's columns are NaN where it has no matching row or no twin exists.
    """
    rows = []
    for scenario, local_dir, global_dir in pairs:
        local = load_scenario_metrics(local_dir).set_index(_KEYS).add_suffix("_local")
        glob_ = load_scenario_metrics(global_dir) if global_dir is not None else pd.DataFrame(columns=_KEYS)
        m = local.join(glob_.set_index(_KEYS).add_suffix("_global"), how="left").reset_index()
        for threshold, csi_area, csi_pop, bias_pop in LOCAL_VS_GLOBAL:
            out = m[_KEYS].assign(scenario=scenario, threshold=threshold)
            for name, col in (("CSI_area", csi_area), ("CSI_pop", csi_pop), ("bias_pop", bias_pop)):
                for side in ("local", "global"):
                    out[f"{name}_{side}"] = m.get(f"{col}_{side}", np.nan)
            rows.append(out)
    cols = ["scenario", *_KEYS, "threshold"]
    df = pd.concat(rows, ignore_index=True)
    return df[cols + [c for c in df.columns if c not in cols]].sort_values(cols, ignore_index=True)


def plot_scores(metrics: pd.DataFrame, out: Path) -> None:
    m = metrics[metrics.region != "ALL"].copy()
    m["label"] = m.benchmark_key.str.split("_").str[0] + ": " + m.region + " " + m.return_period
    m = m.sort_values("label")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.38 * len(m) + 1.8), sharey=True)
    y = np.arange(len(m))
    colors = ("#c6dbef", "#6baed6", "#fdae6b", "#e6550d")
    for k, (name, csi, bias) in enumerate(SCORES):
        axes[0].barh(y + (k - 1.5) * 0.2, m[csi], height=0.19, color=colors[k], label=name)
        axes[1].barh(y + (k - 1.5) * 0.2, m[bias], height=0.19, color=colors[k])
    axes[0].set_yticks(y, m.label, fontsize=7)
    axes[0].set(xlabel="CSI", xlim=(0, 1))
    axes[1].axvline(1, color="#525252", lw=0.8)
    axes[1].set(xlabel="bias (model / benchmark wet)", xscale="log")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    for ax, letter in zip(axes, "ab"):
        draw_panel_letter(ax, letter)
    fig.suptitle("Eikonal vs national flood maps: area and population, at the primary (0.01 m) and exposure "
                 "(0.10 m) thresholds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "X8_regional_area_vs_population.png", dpi=200)
    plt.close(fig)


def plot_map(metrics_dir: Path, iso: str, region: str, rp: str, pop_path: str, out: Path) -> None:
    path = metrics_dir / iso / f"agreement_{iso}_{region}_{rp}_SLR_0.tif"
    with rasterio.open(path) as src:
        cat, transform, crs = src.read(1), src.transform, src.crs
    people = population_per_pixel(pop_path, cat.shape, transform, crs)[0]
    shade = np.clip(np.log10(1 + people / np.maximum(people.max(), 1e-9) * 1e3) / 3, 0.15, 1.0)
    rgb = np.ones(cat.shape + (3,))
    for code, color in ((0, LAND_COLOR), (1, AGREE_COLOR), (2, A_ONLY_COLOR), (3, B_ONLY_COLOR)):
        c = np.array(matplotlib.colors.to_rgb(color))
        sel = cat == code
        rgb[sel] = c if code == 0 else 1 - (1 - c) * shade[sel, None]
    rgb[cat == 255] = 1.0
    fig, ax = plt.subplots(figsize=(8, 8 * cat.shape[0] / cat.shape[1]))
    ax.imshow(rgb)
    ax.set_axis_off()
    handles = [matplotlib.patches.Patch(color=c, label=lab) for c, lab in
               ((AGREE_COLOR, "both wet"), (A_ONLY_COLOR, "benchmark only"), (B_ONLY_COLOR, "model only"))]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title(f"{iso} {region} {rp}: agreement, colour strength = population per pixel", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"X9_exposure_agreement_{iso}_{region}_{rp}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--metrics-dir", help="validation.output_dir of the exposure run")
    parser.add_argument("--reference-dir", help="validation.output_dir of the saved run")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--map-region", default="GBR:firth_of_forth:RP100", help="ISO:region:RP for X9")
    parser.add_argument("--local-vs-global", nargs="+", action="append", metavar="ARG",
                        help="SCENARIO LOCAL_DIR [GLOBAL_DIR]: validation dirs of a local-input run and its "
                             "global-input twin; repeatable. Writes T4 only.")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.local_vs_global:
        if any(len(p) not in (2, 3) for p in args.local_vs_global):
            parser.error("--local-vs-global takes SCENARIO LOCAL_DIR [GLOBAL_DIR]")
        pairs = [(p[0], Path(p[1]), Path(p[2]) if len(p) == 3 else None) for p in args.local_vs_global]
        table = local_vs_global(pairs)
        table.to_csv(out / "T4_local_vs_global_exposure.csv", index=False)
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 240):
            print(table.to_string(index=False))
        return
    if not (args.config and args.metrics_dir and args.reference_dir):
        parser.error("--config, --metrics-dir and --reference-dir are required without --local-vs-global")
    cfg = load_config(args.config)
    paths = cfg["paths"]["hydromt_data_catalog"]
    catalog = get_data_catalog([_REPO_ROOT / p for p in ([paths] if isinstance(paths, str) else paths)],
                               root=cfg["paths"]["root"])
    metrics = load_metrics(Path(args.metrics_dir))
    rep = replication(metrics, Path(args.reference_dir))
    rep.to_csv(out / "replication_vs_saved_run.csv", index=False)
    cols = ["iso", "benchmark_key", "region", "return_period"] + [c for s in SCORES for c in s[1:]] + \
           ["HR_exp_pop", "FAR_exp_pop", "tp_exp_pop", "fp_exp_pop", "fn_exp_pop"]
    metrics[cols].to_csv(out / "T3_regional_exposure_scores.csv", index=False)
    plot_scores(metrics, out)
    iso, region, rp = args.map_region.split(":")
    plot_map(Path(args.metrics_dir), iso, region, rp, catalog.get_source("population").path, out)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 240):
        print(rep.to_string(index=False))
        print(metrics[cols[:12]].to_string(index=False))


if __name__ == "__main__":
    main()
