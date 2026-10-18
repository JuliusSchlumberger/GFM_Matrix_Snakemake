"""Plot validate_sf_bay.py's model-grid outputs: agreement map and model-minus-CoSMoS depth.

Same colours, land/water background and metrics box as plot_agreement_map.py.

Usage:
    python validation/plot_sf_bay.py [--config path/to/config.yml] [--out-dir DIR] [--diff-range-m 1.0]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_agreement_map import (  # noqa: E402
    _draw_annotation_box,
    _draw_background,
    plot_agreement_map,
)
from validate_sf_bay import OUTPUT_TAG  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRID_ROW = "A_coarsen_to_model_grid"


def plot_depth_difference(
    diff_path: Path, gfm_catalog, permanent_water_source, permanent_water_codes,
    output_path: Path, diff_range_m: float, figsize, dpi: int, metrics: dict | None = None,
) -> None:
    """Map model minus CoSMoS depth (m), symmetric colour range +-`diff_range_m`."""
    with retry_transient_io(rasterio.open, diff_path) as src:
        diff = src.read(1)
        bounds, transform = src.bounds, src.transform
    fig, ax = plt.subplots(figsize=figsize)
    _draw_background(
        ax, [bounds.left, bounds.bottom, bounds.right, bounds.top], transform, diff.shape,
        gfm_catalog, permanent_water_source, permanent_water_codes,
    )
    im = ax.imshow(
        np.ma.masked_invalid(diff), extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        cmap="RdBu_r", vmin=-diff_range_m, vmax=diff_range_m, origin="upper", zorder=1,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, extend="both", label="Model - CoSMoS depth (m)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if metrics:
        _draw_annotation_box(ax, [
            f"RMSE={metrics['RMSE_all_m']:.2f} m  MAE={metrics['MAE_all_m']:.2f} m (all)",
            f"RMSE={metrics['RMSE_flooded_m']:.2f} m  MAE={metrics['MAE_flooded_m']:.2f} m (CoSMoS wet)",
        ])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write agreement_*.png and depth_diff_*.png next to validate_sf_bay.py's rasters."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_REPO_ROOT / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--out-dir", default=None, help="default: {validation.output_dir}/USA")
    parser.add_argument("--diff-range-m", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    val_cfg, plots_cfg = cfg["validation"], cfg["validation"]["plots"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(val_cfg["output_dir"]) / "USA"

    gfm_catalog = None
    try:
        gfm_catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    except Exception as e:
        print(f"WARNING: no GFM data catalog for the land/water background ({e}) - plotting without it.")
    water = (val_cfg["permanent_water_source"], val_cfg["permanent_water_codes"])

    row = pd.read_csv(out_dir / f"metrics_{OUTPUT_TAG}.csv", index_col=0).loc[_GRID_ROW]
    extent_metrics = {"HR": row["HT"], "FAR": row["FAR"], "CSI": row["CSI"], "bias": row["bias"]}

    agreement_png = out_dir / f"agreement_{OUTPUT_TAG}.png"
    plot_agreement_map(
        {"RP100": out_dir / f"agreement_{OUTPUT_TAG}.tif"}, gfm_catalog, *water, agreement_png,
        colors=plots_cfg["agreement_colors"], figsize=tuple(plots_cfg["figsize"]), dpi=int(plots_cfg["dpi"]),
        rp_metrics={"RP100": extent_metrics},
    )
    print(f"Written: {agreement_png}")

    diff_png = out_dir / f"depth_diff_{OUTPUT_TAG}.png"
    plot_depth_difference(
        out_dir / f"depth_diff_{OUTPUT_TAG}.tif", gfm_catalog, *water, diff_png, args.diff_range_m,
        figsize=tuple(plots_cfg["figsize"]), dpi=int(plots_cfg["dpi"]), metrics=row.to_dict(),
    )
    print(f"Written: {diff_png}")


if __name__ == "__main__":
    main()
