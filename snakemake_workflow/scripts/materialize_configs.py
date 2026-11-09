"""Materialize scenario configs: `config/<name>.yml` -> `config/<name>_materialized.yml`.

Each output is config.yml + config_local.yml + the override, deep-merged like load_config. Point GFM_CONFIG_PATH at it. --variants adds, per scenario,
`<name>__<variant>_materialized.yml`: the scenario running a flood-model variant
(flood_variants.variant_config) at its run in the SFINCS calibration's variant_selection.json,
with its own outputs and the scenario's tile inputs.

Usage:
    python snakemake_workflow/scripts/materialize_configs.py              # every known scenario
    python snakemake_workflow/scripts/materialize_configs.py deltas_ssp126 bangkok_chao_phraya
    python snakemake_workflow/scripts/materialize_configs.py --out-dir <dir>  # write elsewhere (e.g. to diff)
    python snakemake_workflow/scripts/materialize_configs.py england_local_input --variants all \
        --selection <root>/calibration_roughness_SFINCS/variant_selection.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from config_utils import materialize_config  # noqa: E402
from flood_variants import VARIANTS, variant_config  # noqa: E402

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Override files (config/<name>.yml) whose materialized sibling the pipeline/launchers use.
SCENARIOS = [
    "bangkok_chao_phraya",
    "calibration_500_tiles",
    "deltas_ssp126",
    "deltas_ssp245",
    "england_local_input",
    "england_global_input",
    "england_hop_test",
    "wales_local_input",
    "wales_global_input",
    "scotland_local_input",
    "sf_bay_local_input",
]


def main() -> None:
    """Parse CLI arguments and write `<name>_materialized.yml` for each requested scenario."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", default=SCENARIOS, help=f"scenario names (default: {' '.join(SCENARIOS)})")
    parser.add_argument("--out-dir", default=str(_CONFIG_DIR), help="where to write (default: the config directory)")
    parser.add_argument("--variants", nargs="+", default=[], choices=["all", *VARIANTS],
                        help="also write each scenario running these flood-model variants")
    parser.add_argument("--selection", default=None,
                        help="variant_selection.json of the SFINCS calibration (needed with --variants)")
    args = parser.parse_args()
    variants = list(VARIANTS) if "all" in args.variants else args.variants
    if variants and not args.selection:
        parser.error("--variants needs --selection")
    selection = json.loads(Path(args.selection).read_text())["variants"] if variants else {}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names:
        override_path = _CONFIG_DIR / f"{name}.yml"
        if not override_path.exists():
            raise FileNotFoundError(f"{override_path} does not exist")
        out_path = out_dir / f"{name}_materialized.yml"
        merged = materialize_config(_CONFIG_DIR / "config.yml", [override_path], out_path)
        print(f"wrote {out_path}")
        for variant in variants:
            run = selection.get(variant)
            if run is None and VARIANTS[variant]["friction_source"] is not None:
                raise SystemExit(f"{args.selection} has no run for {variant}")
            run = run or {"parameter": None, "alpha": None}
            v_path = out_dir / f"{name}__{variant}_materialized.yml"
            materialize_config(_CONFIG_DIR / "config.yml", [variant_config(merged, variant, run["parameter"],
                                                                          run["alpha"])], v_path)
            print(f"wrote {v_path}")


if __name__ == "__main__":
    main()
