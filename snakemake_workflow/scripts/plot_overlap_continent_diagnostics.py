"""Per-continent overlap-agreement diagnostics for one return period and SLR scenario.

Pools every chunk's reservoir-sampled per-cell (min, max) depth-across-
overlapping-tiles pairs (written by merge_chunk.py / merge.merge_tile_
rasters_chunk), groups chunks by continent (Natural Earth naturalearth_lowres,
point-in-polygon on each chunk's centroid), and writes one two-subplot PNG
per continent via plotting.plot_overlap_continent_diagnostics:
  - Left: hexbin of (min, max) depth with Pearson r and a y=x agreement line.
  - Right: pie chart of confirmed-flood / confirmed-no-flood / ambiguous
    cells, split by postprocessing.flood_area_threshold_m.

Replaces the old per-chunk overlap_correlation_plot (single tile-pair,
single designated scenario) — see merge.py's module docstring for why
min/max across ALL overlapping tiles is collected instead of a single pair.
"""

import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plotting import plot_overlap_continent_diagnostics  # noqa: E402

pp_cfg = snakemake.params.pp_cfg  # noqa: F821
plot_cfg = pp_cfg["plots"]
return_period = snakemake.wildcards.return_period  # noqa: F821
waterlevel_name = snakemake.wildcards.waterlevel_name  # noqa: F821
scenario_label = f"{return_period}_{waterlevel_name}"

output_dir = Path(snakemake.output.diagnostics)  # noqa: F821
output_dir.mkdir(parents=True, exist_ok=True)

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

if not by_continent:
    # No overlap data anywhere for this scenario - still write a placeholder
    # so Snakemake's directory() output isn't empty.
    plot_overlap_continent_diagnostics(
        mins=np.empty(0), maxs=np.empty(0), threshold_m=pp_cfg["flood_area_threshold_m"],
        output_path=output_dir / f"no_overlap_{scenario_label}.png",
        continent_name="(none)", waterlevel_name=scenario_label, n_chunks=0,
        dpi=plot_cfg["dpi"],
    )
else:
    for continent, pairs in by_continent.items():
        mins = np.concatenate([m for m, _ in pairs])
        maxs = np.concatenate([m for _, m in pairs])
        safe_name = continent.replace(" ", "_").replace("/", "-")
        plot_overlap_continent_diagnostics(
            mins=mins, maxs=maxs, threshold_m=pp_cfg["flood_area_threshold_m"],
            output_path=output_dir / f"{safe_name}_{scenario_label}.png",
            continent_name=continent, waterlevel_name=scenario_label,
            n_chunks=chunk_counts[continent], dpi=plot_cfg["dpi"],
        )
