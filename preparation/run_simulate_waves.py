"""CLI wrapper: run one materialized scenario's `simulate` stage wave by wave.

Thin entry point around `run_delta_floodmaps.run_simulate_stage` - that function is
already fully config-driven (tile_grid's own hop_distance column, boundary_conditions'
own return_periods/waterlevel_names), with zero England/Wales-specific logic, so this
script adds no new wave-propagation code. It exists because `run_delta_floodmaps.py`'s
own CLI is scoped to the named deltas_ssp126/ssp245 scenarios - any materialized config
can use this instead (first exercised for england_hop_test's Wash cluster, then reused
for wales_local_input's Severn Estuary sample - TODO.md).

Usage:
    python preparation/run_simulate_waves.py --config <materialized.yml> --cores 4
    python preparation/run_simulate_waves.py --config <materialized.yml> --cores 4 \\
        --return-periods RP100 RP250 --waterlevel-names SLR_0
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
from run_delta_floodmaps import run_simulate_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="materialized scenario config path")
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--mem-mb", type=int, default=None)
    parser.add_argument("--return-periods", nargs="+", default=None,
                        help="e.g. RP100 RP250 - restricts the eikonal solve below the "
                             "config's own boundary_conditions.return_periods list "
                             "(unset = use the config's full list, the production default)")
    parser.add_argument("--waterlevel-names", nargs="+", default=None,
                        help="e.g. SLR_0 - restricts below the config's own merged "
                             "waterlevel_names/adaptation.slr_intensities union")
    args = parser.parse_args()

    materialized = Path(args.config)
    env = os.environ.copy()
    env["GFM_CONFIG_PATH"] = str(materialized)

    ok = run_simulate_stage(
        materialized, args.cores, args.mem_mb, env,
        return_periods=args.return_periods, waterlevel_names=args.waterlevel_names,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
