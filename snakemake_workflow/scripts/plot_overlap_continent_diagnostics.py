"""Per-continent overlap-agreement diagnostics for one return period and SLR scenario.

Pools every chunk's reservoir-sampled per-cell (min, max) depth-across-
overlapping-tiles pairs (written by merge_chunk.py / merge.merge_tile_
rasters_chunk), groups chunks by continent (Natural Earth naturalearth_lowres,
point-in-polygon on each chunk's centroid), and writes one two-subplot PNG
per continent via plotting.plot_overlap_continent_diagnostics:
  - Left: hexbin of (min, max) depth with Pearson r and a y=x agreement line.
  - Right: pie chart of confirmed-flood / confirmed-no-flood / ambiguous
    cells, split by exposure.exceedance_threshold_m.

See merge.py's module docstring for why min/max across ALL overlapping
tiles is collected instead of a single tile-pair.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import retry_transient_io  # noqa: E402
from plotting import plot_overlap_continent_diagnostics  # noqa: E402

pp_cfg = snakemake.params.pp_cfg  # noqa: F821
plot_cfg = pp_cfg["plots"]
threshold_m = snakemake.params.threshold_m  # noqa: F821
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_label = f"{return_period}_{waterlevel_name}"

output_dir = Path(snakemake.output.diagnostics)  # noqa: F821
retry_transient_io(output_dir.mkdir, parents=True, exist_ok=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    continents = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))[["continent", "geometry"]]


def _continent_for_point(x: float, y: float) -> str:
    """Point-in-polygon lookup against the ~7-row continents layer (cheap enough
    to just iterate; not worth an sjoin for this few candidate rows)."""
    pt = Point(x, y)
    hit = continents[continents.contains(pt)]
    if hit.empty:
        # Coastal chunk centroids can legitimately fall just offshore of every
        # land polygon; fall back to nearest continent by centroid distance.
        hit = continents.iloc[[continents.distance(pt).idxmin()]]
    return str(hit.iloc[0]["continent"])


# ── Pool every chunk's samples by continent ─────────────────────────────────
by_continent: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
chunk_counts: dict[str, int] = {}
total_cells: dict[str, int] = {}

for npz_path in snakemake.input.overlap_files:  # noqa: F821
    data = np.load(npz_path)
    mins, maxs = data["mins"], data["maxs"]
    if len(mins) == 0:
        continue
    minx, miny, maxx, maxy = data["bounds"]
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    continent = _continent_for_point(cx, cy)
    by_continent.setdefault(continent, []).append((mins, maxs))
    chunk_counts[continent] = chunk_counts.get(continent, 0) + 1
    # overlap_minmax.npz files without a total_overlap_cells field fall back to
    # len(mins), i.e. assume that chunk never hit overlap_corr_max_samples -
    # a lower-bound approximation.
    chunk_total = int(data["total_overlap_cells"]) if "total_overlap_cells" in data.files else len(mins)
    total_cells[continent] = total_cells.get(continent, 0) + chunk_total

if not by_continent:
    # No overlap data anywhere for this scenario - still write a placeholder
    # so Snakemake's directory() output isn't empty.
    plot_overlap_continent_diagnostics(
        mins=np.empty(0), maxs=np.empty(0), threshold_m=threshold_m,
        output_path=output_dir / f"no_overlap_{scenario_label}.png",
        continent_name="(none)", waterlevel_name=scenario_label, n_chunks=0,
        total_overlap_cells=0, pie_colors=plot_cfg["overlap_pie_colors"],
        figsize=tuple(plot_cfg["overlap_continent_figsize"]), dpi=plot_cfg["dpi"],
    )
else:
    for continent, pairs in by_continent.items():
        mins = np.concatenate([m for m, _ in pairs])
        maxs = np.concatenate([m for _, m in pairs])
        safe_name = continent.replace(" ", "_").replace("/", "-")
        plot_overlap_continent_diagnostics(
            mins=mins, maxs=maxs, threshold_m=threshold_m,
            output_path=output_dir / f"{safe_name}_{scenario_label}.png",
            continent_name=continent, waterlevel_name=scenario_label,
            n_chunks=chunk_counts[continent], total_overlap_cells=total_cells[continent],
            pie_colors=plot_cfg["overlap_pie_colors"],
            figsize=tuple(plot_cfg["overlap_continent_figsize"]), dpi=plot_cfg["dpi"],
        )
