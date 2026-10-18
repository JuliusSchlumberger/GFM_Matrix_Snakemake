"""Materialize scenario configs: `config/<name>.yml` -> `config/<name>_materialized.yml`.

Each materialized file is a standalone, fully-merged config.yml + config_local.yml + the one
override file (src/config_utils.py::materialize_config - the same deep-merge semantics
load_config uses). Point GFM_CONFIG_PATH at it to run Snakemake on that scenario (see the
Snakefile's comment on why GFM_CONFIG_PATH is used instead of --configfile).

The `*_materialized.yml` files are generated and git-ignored; this script rebuilds them on a
fresh clone. It replaces build_delta_run_configs.py (deltas_ssp126/245 only) and covers every
override that has a materialized sibling. `deltas_design_materialized.yml` is not an override
merge (it carries per-delta computed values) and is still written by
analysis/prepare_delta_design_runs.py.

Usage:
    python snakemake_workflow/scripts/materialize_configs.py              # every known scenario
    python snakemake_workflow/scripts/materialize_configs.py deltas_ssp126 thailand_bangkok
    python snakemake_workflow/scripts/materialize_configs.py --out-dir <dir>  # write elsewhere (e.g. to diff)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import materialize_config  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Override files (config/<name>.yml) whose materialized sibling the pipeline/launchers use.
SCENARIOS = [
    "bangkok_chao_phraya",
    "calibration_500_tiles",
    "deltas_ssp126",
    "deltas_ssp245",
    "gbr_wales_scotland_friction9",
    "thailand_bangkok",
]


def main() -> None:
    """Parse CLI arguments and write `<name>_materialized.yml` for each requested scenario."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=SCENARIOS, help=f"scenario names (default: {' '.join(SCENARIOS)})")
    parser.add_argument("--out-dir", default=str(_CONFIG_DIR), help="where to write (default: the config directory)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        override_path = _CONFIG_DIR / f"{name}.yml"
        if not override_path.exists():
            raise FileNotFoundError(f"{override_path} does not exist")
        out_path = out_dir / f"{name}_materialized.yml"
        materialize_config(_CONFIG_DIR / "config.yml", [override_path], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
