"""Materialize one real, standalone config file for one (group, sweep_point)
combination of the ESP/FRA/NOR HPC calibration sweep
(docs/calibration_sweep_plan.md).

Layers, in order, via src/config_utils.py::materialize_config (deep-merge,
same semantics load_config/Snakemake's own --configfile already use):

    config.yml -> config_local.yml (auto) -> group scenario
    (config/calibration/{group}.yml) -> sweep-point delta
    (config/calibration/sweep/{sweep}.yml) -> computed path-isolation dict

The path-isolation dict gives every (group, sweep_point) combination its own
fully isolated `simulation.model_outputs`/`postprocessing.merged_outputs`/
`validation.output_dir`/`hpc.jobs_dir`, under
`{root}/calibration_esp_fra_nor/{run_tag}/` - so combinations never collide
with each other OR with production, even though `waterdepth_{rp}_{slr}.tif`
output filenames carry no parameter tag of their own.

`simulation.preprocessing_inputs_dir` is the one exception, deliberately NOT
isolated per run_tag - it's shared at the GROUP level instead
(`{root}/calibration_esp_fra_nor/{group}/model_outputs`), reused across all
14 sweep points of that group (2026-09-14, extended 2026-09-15 with the
exposure.exceedance_threshold_m points). None of the 14 OFAT sweep points
(friction_scale_factor, max_rounds, obstacle_coupling.*,
waterlevel_epsilon_m, exposure.exceedance_threshold_m) touch anything
preprocessing produces (dem/mask/friction/boundaries under inputs/) -
they're all solver-runtime or postprocessing/exposure-only parameters read
only after preprocessing has already finished - so every sweep point in a
group needs byte-identical preprocessing inputs. Sharing them cuts
preprocessing from 14x-per-group redundant work to 1x-per-group (2 total
across the whole 28-combination sweep, not 28). Safe specifically because
`results/waterdepth_*.tif` (the one output whose filename has no
sweep-point tag, and genuinely DOES differ per sweep point for the 4
solver-parameter families - see the methodology doc for why the
exceedance_threshold points are a documented exception to that) lives under
`simulation.model_outputs`, which stays isolated per run_tag as before - see
preprocessing.smk's own comment for the full inputs/-vs-results/ split.

Deliberately does NOT bake `config_hpc.yml` into the materialized file - the
generator scripts that consume it (generate_hpc_preprocess_job.py etc.)
auto-discover `config_hpc.yml` NEXT TO whatever --config path they're given,
so this script copies the real one alongside the materialized file instead,
keeping it a separate Linux-view override the way it's meant to be used
(see materialize_config's own docstring for why it doesn't just fold this
in like config_local.yml).

Reused as-is (2026-10-08) for other, unrelated country-validation studies via
`--calibration-dir-name` (default `calibration_esp_fra_nor`, kept for
backward compatibility with this study's own already-materialized configs) -
same layering/isolation logic, just a different output directory under
`{root}` so an unrelated study's runs never collide with this one's.

Usage:
    python build_run_config.py --group esp_fra_rp100 --sweep max_rounds_20
    python build_run_config.py --group nor_rp250 --sweep baseline
    python build_run_config.py --group new_brunswick_rp100 --sweep baseline \\
        --calibration-dir-name calibration_country_validation
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from config_utils import materialize_config  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "snakemake_workflow" / "config"


def build_run_config(
    group: str, sweep_point: str, calibration_dir_name: str = "calibration_esp_fra_nor",
    preprocessing_inputs_dir: str | None = None,
) -> Path:
    group_path = _CONFIG_DIR / "calibration" / f"{group}.yml"
    sweep_path = _CONFIG_DIR / "calibration" / "sweep" / f"{sweep_point}.yml"
    # Fail loudly on a typo'd --group/--sweep rather than silently
    # materializing an unsubsetted, unswept config - materialize_config's
    # own overrides argument tolerates missing files (same convention as
    # load_config's extra_override), which is right for genuinely optional
    # layers but wrong here, where both layers are required by construction.
    if not group_path.exists():
        raise FileNotFoundError(f"unknown --group {group!r}: {group_path} does not exist")
    if not sweep_path.exists():
        raise FileNotFoundError(f"unknown --sweep {sweep_point!r}: {sweep_path} does not exist")

    run_tag = f"{group}__{sweep_point}"
    # Default: group-level isolated preprocessing dir, shared across every sweep point of
    # THIS group (module docstring's own reasoning) - but a caller whose tiles are already
    # preprocessed under PRODUCTION's own model_outputs (e.g. a country-validation study
    # that only needs the solve + CSI scoring, not fresh dem/mask/friction/boundaries) can
    # pass --preprocessing-inputs-dir to reuse that directly instead, skipping preprocessing
    # entirely (2026-10-07, gbr_wales_scotland/new_brunswick country-validation runs).
    preprocessing_inputs_dir = preprocessing_inputs_dir or f"{{root}}/{calibration_dir_name}/{group}/model_outputs"
    path_overrides = {
        "simulation": {
            "model_outputs": f"{{root}}/{calibration_dir_name}/{run_tag}/model_outputs",
            "preprocessing_inputs_dir": preprocessing_inputs_dir,
        },
        "postprocessing": {"merged_outputs": f"{{root}}/{calibration_dir_name}/{run_tag}/merged_results"},
        "validation": {"output_dir": f"{{root}}/{calibration_dir_name}/{run_tag}/validation"},
        "hpc": {"jobs_dir": f"{{root}}/{calibration_dir_name}/{run_tag}/hpc_jobs"},
    }

    materialized_dir = _CONFIG_DIR / "calibration" / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    out_path = materialized_dir / f"{run_tag}.yml"
    materialize_config(_CONFIG_DIR / "config.yml", [group_path, sweep_path, path_overrides], out_path)

    real_hpc_cfg = _CONFIG_DIR / "config_hpc.yml"
    if real_hpc_cfg.exists():
        shutil.copyfile(real_hpc_cfg, materialized_dir / "config_hpc.yml")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="e.g. esp_fra_rp100, nor_rp250")
    parser.add_argument("--sweep", required=True, help="e.g. baseline, max_rounds_20, obstacle_coupling_off")
    parser.add_argument(
        "--calibration-dir-name", default="calibration_esp_fra_nor",
        help="output directory name under {root} - default matches the original ESP/FRA/NOR study "
             "(kept as the default for backward compatibility with existing materialized configs); "
             "pass a different name for an unrelated study, e.g. calibration_country_validation",
    )
    parser.add_argument(
        "--preprocessing-inputs-dir", default=None,
        help="override simulation.preprocessing_inputs_dir (default: an isolated group-level dir "
             "under --calibration-dir-name, as usual) - pass '{root}/model_outputs' (literal, "
             "{root} expanded later by load_config) to reuse already-preprocessed PRODUCTION tiles "
             "directly and skip preprocessing entirely for this study",
    )
    args = parser.parse_args()

    out_path = build_run_config(args.group, args.sweep, args.calibration_dir_name, args.preprocessing_inputs_dir)
    print(f"run_tag: {args.group}__{args.sweep}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
