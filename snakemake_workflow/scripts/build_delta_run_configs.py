"""Materialize the two global-deltas flood-hazard-map run configs
(ssp126, ssp245 - see config/deltas_ssp126.yml / deltas_ssp245.yml for the
full rationale), via src/config_utils.py::materialize_config - same
deep-merge semantics load_config/Snakemake's own --configfile already use.

Writes config/deltas_ssp126_materialized.yml and
deltas_ssp245_materialized.yml, flat alongside their override files - same
layout as gbr_wales_scotland_friction9_materialized.yml /
thailand_bangkok_materialized.yml, not the calibration sweep's nested
materialized/ subdirectory (no sweep here, just two fixed scenarios).

Each materialized file is a standalone, fully-merged config.yml +
config_local.yml + the one override file - point GFM_CONFIG_PATH at it to
run Snakemake (see Snakefile's own comment on why GFM_CONFIG_PATH is used
instead of --configfile).

Usage:
    python build_delta_run_configs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import materialize_config  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

SCENARIOS = ["ssp126", "ssp245"]


def main() -> None:
    for scenario in SCENARIOS:
        override_path = _CONFIG_DIR / f"deltas_{scenario}.yml"
        if not override_path.exists():
            raise FileNotFoundError(f"{override_path} does not exist")
        out_path = _CONFIG_DIR / f"deltas_{scenario}_materialized.yml"
        materialize_config(_CONFIG_DIR / "config.yml", [override_path], out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
