"""Figures comparing attenuated and two-component SLR propagation on the Bangkok tiles.

H1: flooded area vs SLR per return period (all tiles, and per tile).
H2: depth maps (attenuated, two-component, difference, static extent) for chosen tiles.
H3: water-level profile along a coast-to-inland row of a tile.
Inputs: the scenario's own waterdepth rasters (attenuated) and run_two_component_bangkok.py output.

Usage:
    python review_critiques/plot_two_component_bangkok.py --two-component-dir <dir> --fig-dir <dir>
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from config_utils import load_config  # noqa: E402
from merge import EIKONAL_NODATA, decode_waterdepth_array  # noqa: E402
from pipeline_plots import pixel_area_km2_grid  # noqa: E402
from rasters import decode_dem_cm  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "snakemake_workflow" / "config" / "bangkok_chao_phraya_materialized.yml"
WET_THRESHOLD_M = 0.10  # exposure.exceedance_threshold_m
MAP_DOWNSAMPLE = 8


def _depth(path: Path, downsample: int = 1) -> tuple[np.ndarray, object]:
    with rasterio.open(path) as src:
        if downsample == 1:
            raw, transform = src.read(1), src.transform
        else:
            shape = (src.height // downsample, src.width // downsample)
            raw = src.read(1, out_shape=shape, resampling=Resampling.nearest)
            transform = src.transform * src.transform.scale(src.width / shape[1], src.height / shape[0])
    d = decode_waterdepth_array(raw)
    d[d == EIKONAL_NODATA] = np.nan
    return d, transform


def flooded_area_table(ref_dir: Path, new_dir: Path, tiles: list[str], scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for tile in tiles:
        for scen in scenarios:
            new_path = new_dir / tile / "results" / f"waterdepth_{scen}.tif"
            ref_path = ref_dir / tile / "results" / f"waterdepth_{scen}.tif"
            if not (new_path.exists() and ref_path.exists()):
                continue
            out = {"tile_id": tile, "scenario": scen}
            for name, path in (("attenuated", ref_path), ("two_component", new_path)):
                d, transform = _depth(path)
                area = pixel_area_km2_grid(transform, d.shape[1], d.shape[0])
                out[f"area_km2_{name}"] = float(area[np.nan_to_num(d) >= WET_THRESHOLD_M].sum())
            rows.append(out)
    df = pd.DataFrame(rows)
    df["return_period"] = df["scenario"].str.split("_").str[0]
    df["slr_m"] = df["scenario"].str.split("_").str[-1].astype(int) / 1000
    return df


def plot_h1(df: pd.DataFrame, fig_dir: Path) -> None:
    rps = sorted(df["return_period"].unique(), key=lambda r: int(r[2:]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tot = df.groupby(["return_period", "slr_m"])[["area_km2_attenuated", "area_km2_two_component"]].sum().reset_index()
    for i, rp in enumerate(rps):
        sub = tot[tot.return_period == rp].sort_values("slr_m")
        col = plt.cm.viridis(i / max(1, len(rps) - 1))
        axes[0].plot(sub.slr_m, sub.area_km2_attenuated, "--o", color=col, label=f"{rp} attenuated")
        axes[0].plot(sub.slr_m, sub.area_km2_two_component, "-s", color=col, label=f"{rp} two-component")
    axes[0].set_xlabel("SLR (m)")
    axes[0].set_ylabel(f"flooded area, depth >= {WET_THRESHOLD_M} m (km²), 9 tiles")
    axes[0].set_title("Flooded area vs SLR")
    axes[0].legend(fontsize=8)
    sub = df[df.return_period == "RP100"].copy()
    sub["ratio"] = sub.area_km2_two_component / sub.area_km2_attenuated.replace(0, np.nan)
    for tile, g in sub.groupby("tile_id"):
        g = g.sort_values("slr_m")
        axes[1].plot(g.slr_m, g.ratio, "-o", label=tile)
    axes[1].axhline(1.0, color="k", lw=0.8)
    axes[1].set_xlabel("SLR (m)")
    axes[1].set_ylabel("two-component / attenuated flooded area")
    axes[1].set_title("RP100, per tile")
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "H1_flooded_area_vs_slr.png", dpi=150)
    plt.close(fig)


def plot_h2(ref_dir: Path, new_dir: Path, prep_dir: Path, tile: str, rp: str, slrs: list[str], fig_dir: Path) -> None:
    fig, axes = plt.subplots(len(slrs), 4, figsize=(20, 4.6 * len(slrs)), squeeze=False)
    for r, slr in enumerate(slrs):
        scen = f"{rp}_{slr}"
        ref, _ = _depth(ref_dir / tile / "results" / f"waterdepth_{scen}.tif", MAP_DOWNSAMPLE)
        new, _ = _depth(new_dir / tile / "results" / f"waterdepth_{scen}.tif", MAP_DOWNSAMPLE)
        vmax = float(np.nanpercentile(np.where(new > 0, new, np.nan), 98)) if np.nanmax(new) > 0 else 1.0
        for c, (arr, title) in enumerate([(ref, "attenuated"), (new, "two-component")]):
            im = axes[r, c].imshow(np.where(arr > 0, arr, np.nan), cmap="Blues", vmin=0, vmax=vmax)
            axes[r, c].set_title(f"{scen} {title} depth (m)")
            fig.colorbar(im, ax=axes[r, c], fraction=0.04)
        diff = np.nan_to_num(new) - np.nan_to_num(ref)
        lim = float(np.nanpercentile(np.abs(diff[diff != 0]), 98)) if (diff != 0).any() else 1.0
        im = axes[r, 2].imshow(np.where(diff != 0, diff, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 2].set_title("two-component - attenuated (m)")
        fig.colorbar(im, ax=axes[r, 2], fraction=0.04)
        cls = np.zeros(new.shape)
        cls[(np.nan_to_num(ref) > 0)] = 1
        cls[(np.nan_to_num(new) > 0) & ~(np.nan_to_num(ref) > 0)] = 2
        axes[r, 3].imshow(cls, cmap=matplotlib.colors.ListedColormap(["#f7f7f7", "#6baed6", "#e6550d"]),
                          vmin=-0.5, vmax=2.5, interpolation="nearest")
        axes[r, 3].set_title("wet in both (blue) / only two-component (orange)")
        for ax in axes[r]:
            ax.set_axis_off()
    fig.suptitle(f"Tile {tile}, {rp}: attenuated vs two-component propagation", fontsize=14)
    fig.tight_layout()
    fig.savefig(fig_dir / f"H2_maps_tile{tile}_{rp}.png", dpi=110)
    plt.close(fig)


def plot_h3(ref_dir: Path, new_dir: Path, prep_dir: Path, tile: str, rp: str, slrs: list[str], fig_dir: Path) -> None:
    with rasterio.open(prep_dir / tile / "inputs" / "dem.tif") as src:
        dem = decode_dem_cm(src.read(1))
    with rasterio.open(prep_dir / tile / "inputs" / "mask.tif") as src:
        mask = src.read(1)
    ref0, _ = _depth(ref_dir / tile / "results" / f"waterdepth_{rp}_{slrs[-1]}.tif")
    row = int(np.argmax((np.nan_to_num(ref0) > 0).sum(axis=1)))
    x_km = np.arange(dem.shape[1]) * 0.03
    fig, ax = plt.subplots(figsize=(14, 5))
    land = mask[row] == 0
    ax.fill_between(x_km, -5, np.where(land, dem[row], np.nan), color="#bcaa8a", label="ground (land)")
    for i, slr in enumerate(slrs):
        col = plt.cm.plasma(i / max(1, len(slrs) - 1))
        for path_dir, ls, name in ((ref_dir, "--", "attenuated"), (new_dir, "-", "two-component")):
            d, _ = _depth(path_dir / tile / "results" / f"waterdepth_{rp}_{slr}.tif")
            level = np.where(np.nan_to_num(d[row]) > 0, dem[row] + d[row], np.nan)
            ax.plot(x_km, level, ls, color=col, lw=1.2, label=f"{slr} {name}")
    ax.set_ylim(-3, max(4.0, float(np.nanpercentile(dem[row][land], 60)) + 1))
    ax.set_xlabel("distance along tile row (km)")
    ax.set_ylabel("water level / ground (m, DEM datum)")
    ax.set_title(f"Tile {tile} {rp}, row {row}: water level (dashed attenuated, solid two-component)")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"H3_profile_tile{tile}_{rp}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--two-component-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    parser.add_argument("--map-tiles", nargs="+", default=None, help="default: the 2 tiles with most added area")
    args = parser.parse_args()
    cfg = load_config(args.config)
    sim = cfg["simulation"]
    ref_dir = Path(sim["model_outputs"])
    prep_dir = Path(sim.get("preprocessing_inputs_dir") or sim["model_outputs"])
    new_dir = Path(args.two_component_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tiles = sorted(p.name for p in new_dir.iterdir() if (p / "results").is_dir())
    scenarios = sorted({f.stem.replace("waterdepth_", "") for t in tiles for f in (new_dir / t / "results").glob("*.tif")})
    df = flooded_area_table(ref_dir, new_dir, tiles, scenarios)
    df.to_csv(fig_dir / "H1_flooded_area.csv", index=False)
    plot_h1(df, fig_dir)
    print("  wrote H1")

    slrs = sorted(df["scenario"].str.split("_", n=1).str[1].unique(), key=lambda s: int(s.split("_")[1]))
    rp100 = df[df.return_period == "RP100"]
    added = (rp100.area_km2_two_component - rp100.area_km2_attenuated).groupby(rp100.tile_id).sum()
    map_tiles = args.map_tiles or list(added.sort_values(ascending=False).index[:2])
    for tile in map_tiles:
        plot_h2(ref_dir, new_dir, prep_dir, tile, "RP100", [slrs[0], slrs[len(slrs) // 2], slrs[-1]], fig_dir)
        plot_h3(ref_dir, new_dir, prep_dir, tile, "RP100", slrs, fig_dir)
        print(f"  wrote H2/H3 tile {tile}")


if __name__ == "__main__":
    main()
