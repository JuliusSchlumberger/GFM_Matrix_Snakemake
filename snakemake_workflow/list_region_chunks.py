"""Print one line per region (see src/regions.py), for postprocess_region /
run_postprocess_regions.sh: name, number of "safe" chunks (fully contained
within that region - see src/chunks.py:safe_chunks_for_region), number of
"partial" chunks (also touch another region, not yet requestable for this
region alone), job count (safe chunks x return periods x SLR scenarios),
and the full marker-file path postprocess_region produces for it -
tab-separated.

Single source of truth for run_postprocess_regions.sh, mirroring
list_regions.py's role for run_simulate_regions.sh: it discovers regions
and their safe/partial chunk split by calling the same
src/chunks.py/src/regions.py functions the Snakefile itself uses, rather
than duplicating that logic, so the two can never drift out of sync with
what the Snakefile would compute for the current tile grid.  Not part of
the Snakemake DAG.

Usage:
    python snakemake_workflow/list_region_chunks.py [--config path/to/config.yml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chunks import build_chunk_grid, chunk_tile_lookup, safe_chunks_for_region  # noqa: E402
from config_utils import load_config  # noqa: E402
from regions import assign_regions  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def main() -> None:
    # See list_regions.py's own main() for why this is forced - Windows
    # Python's stdout \r\n translation otherwise corrupts the marker path
    # (this script's last tab-separated field) when read via process
    # substitution, the same confirmed-not-hypothetical bug fixed there.
    sys.stdout.reconfigure(newline="\n")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(Path(__file__).resolve().parent / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg, help=f"path to config.yml (default: {_default_cfg})")
    args = parser.parse_args()

    cfg = load_config(Path(args.config).resolve())
    tile_gdf = load_tile_grid(cfg["tile_grid"]["path"])

    bc_cfg = cfg["boundary_conditions"]
    n_slr = len(list(dict.fromkeys(bc_cfg["slr_scenarios"] + cfg["adaptation"]["slr_intensities"])))
    n_scenarios_per_tile = len(bc_cfg["return_periods"]) * n_slr

    tile_regions = assign_regions(tile_gdf, n_scenarios_per_tile)

    chunk_size_deg = cfg["postprocessing"]["chunk_size_deg"]
    chunk_grid = build_chunk_grid(tile_gdf, chunk_size_deg)
    lookup = chunk_tile_lookup(chunk_grid, tile_gdf)

    region_names = sorted(set(tile_regions.values()))
    merged_outputs = cfg["postprocessing"]["merged_outputs"]

    rows = []
    for name in region_names:
        safe, partial = safe_chunks_for_region(lookup, tile_regions, name)
        n_jobs = len(safe) * n_scenarios_per_tile
        rows.append((name, len(safe), len(partial), n_jobs))

    # Largest-job-count first, matching list_regions.py's own sort order.
    for name, n_safe, n_partial, n_jobs in sorted(rows, key=lambda r: -r[3]):
        # Forward slashes only, deliberately not os.path.join - must
        # textually match the Snakefile's own postprocess_region output
        # pattern byte-for-byte, since this string gets passed straight
        # through as a snakemake CLI target argument (see that rule's own
        # comment, and waterdepth_tiles_for_chunk's, for why).
        marker_path = f"{merged_outputs}/_postprocess_region_done/{name}.done"
        print(f"{name}\t{n_safe}\t{n_partial}\t{n_jobs}\t{marker_path}")

    if any(n_partial for _, _, n_partial, _ in rows):
        skipped = [(name, n_partial) for name, _, n_partial, _ in rows if n_partial]
        print(
            "# Note: some regions have chunks straddling a region boundary, "
            "not counted above and not requestable via postprocess_region "
            "until every region touching them is done - "
            + ", ".join(f"{name} ({n})" for name, n in skipped)
            + ". Run the plain `postprocess` target once all regions are "
            "simulated to pick these up.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
