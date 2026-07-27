"""Print one line per region (see src/regions.py): name, job count, and the
full marker-file path `simulate_region` produces for it - tab-separated.

Single source of truth for run_simulate_regions.sh: it discovers regions by
calling this script rather than hardcoding/duplicating the mapping logic,
so the two can never drift out of sync with what the Snakefile itself
would compute for the current tile grid. Not part of the Snakemake DAG.

Usage:
    python snakemake_workflow/list_regions.py [--config path/to/config.yml]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config_utils import load_config  # noqa: E402
from regions import assign_regions  # noqa: E402
from tiles import load_tile_grid  # noqa: E402


def main() -> None:
    # Windows Python's stdout is opened in text mode by default, translating
    # every "\n" this script prints to "\r\n". A plain file redirect
    # (`> out.txt`) can end up losing that trailing \r somewhere in Git
    # Bash's own translation layer, but run_simulate_regions.sh reads this
    # script's output via process substitution (`< <(python list_regions.py)`),
    # which preserves it verbatim - leaving every parsed field's LAST value
    # on each line (the marker path) with an invisible trailing \r. That
    # silently made every marker path textually different from the
    # Snakefile's own clean output pattern, which then reports
    # "MissingRuleException: No rule to produce ...done" for what looks
    # like (but isn't) the exact same path - confirmed by hexdump; not a
    # hypothetical. Force \n-only output so every consumer gets clean lines
    # regardless of platform default or how it captures this script's stdout.
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

    regions = assign_regions(tile_gdf, n_scenarios_per_tile)
    counts: dict[str, int] = {}
    for name in regions.values():
        counts[name] = counts.get(name, 0) + 1

    model_outputs = cfg["simulation"]["model_outputs"]
    for name, n_tiles in sorted(counts.items(), key=lambda kv: -kv[1]):
        n_jobs = n_tiles * n_scenarios_per_tile
        # Forward slashes only, deliberately not os.path.join - must
        # textually match the Snakefile's own simulate_region output
        # pattern byte-for-byte, since this string gets passed straight
        # through as a snakemake CLI target argument (see that rule's
        # own comment for why os.path.join's Windows backslashes break
        # this exact kind of manually-constructed path).
        marker_path = f"{model_outputs}/_region_done/{name}.done"
        print(f"{name}\t{n_jobs}\t{marker_path}")


if __name__ == "__main__":
    main()
