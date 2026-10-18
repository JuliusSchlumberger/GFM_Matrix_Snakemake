"""Materialize scenario configs: `config/<name>.yml` -> `config/<name>_materialized.yml`.

Each output is config.yml + config_local.yml + the override + optional `<name>_pins.yml`
(values pinned to what the scenario's existing outputs were computed with), deep-merged
like load_config. Point GFM_CONFIG_PATH at it. `deltas_design_materialized.yml` is not
built here (analysis/prepare_delta_design_runs.py writes it).

Usage:
    python snakemake_workflow/scripts/materialize_configs.py              # every known scenario
    python snakemake_workflow/scripts/materialize_configs.py deltas_ssp126 bangkok_chao_phraya
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
        pins_path = _CONFIG_DIR / f"{name}_pins.yml"
        overrides = [override_path] + ([pins_path] if pins_path.exists() else [])
        out_path = out_dir / f"{name}_materialized.yml"
        materialize_config(_CONFIG_DIR / "config.yml", overrides, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
