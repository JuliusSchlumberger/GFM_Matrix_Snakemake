"""Single entry point for the full coastal flood-extent validation pipeline.

Discovers every country with a coastal benchmark in the validation catalog,
runs validate_country.py + plot_agreement_map.py for each (one subprocess
per step, mirroring analysis/run_analysis.py's own pattern - module
isolation, one country's crash doesn't abort the rest unless --fail-fast),
then concatenates every country's metrics CSV into one cross-country summary.

Usage:
    python snakemake_workflow/validation/run_validation.py \\
        [--config snakemake_workflow/config/config.yml] \\
        [--countries ESP FRA] \\
        [--fail-fast]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import atomic_write, get_data_catalog, load_config, retry_transient_io  # noqa: E402
from validation import fix_catalog_meta_encoding  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _discover_countries(cfg: dict) -> list[str]:
    """Every distinct country_iso with at least one hazard_type=='coastal' benchmark.

    Scans the whole catalog rather than a hardcoded country list (plan doc
    §5.2) - matches validate_country.py's own `_find_benchmark_keys`.
    """
    val_cfg = cfg["validation"]
    catalog = get_data_catalog(_REPO_ROOT / val_cfg["benchmark_catalog"], root=val_cfg["benchmark_root"])
    fix_catalog_meta_encoding(catalog, _REPO_ROOT / val_cfg["benchmark_catalog"])
    isos = set()
    for key in catalog.sources.keys():
        meta = catalog.get_source(key).meta or {}
        if meta.get("hazard_type") == "coastal" and meta.get("country_iso"):
            isos.add(str(meta["country_iso"]))
    return sorted(isos)


def _run(cmd: list[str], label: str, fail_fast: bool) -> bool:
    print(f"\n{'═' * 60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'═' * 60}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit {result.returncode}) after {elapsed:.0f}s — {label}")
        if fail_fast:
            print("  Aborting (--fail-fast).")
            sys.exit(result.returncode)
        return False
    print(f"\n  ✓ Done in {elapsed:.0f}s — {label}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _default_cfg = str(SCRIPTS_DIR.parent / "snakemake_workflow" / "config" / "config.yml")
    parser.add_argument("--config", default=_default_cfg)
    parser.add_argument("--countries", nargs="+", default=None,
                        help="ISO-3 codes to validate (default: every country with a coastal benchmark)")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="metrics CSVs only, skip agreement maps")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    val_cfg = cfg["validation"]
    rp, slr = val_cfg["return_period"], val_cfg["waterlevel_name"]

    countries = args.countries or _discover_countries(cfg)
    if not countries:
        print("No countries with a coastal benchmark found in the validation catalog.")
        sys.exit(1)
    print(f"Validating {len(countries)} countr{'y' if len(countries) == 1 else 'ies'}: {', '.join(countries)}")

    cfg_arg = ["--config", str(config_path)]
    results: dict[str, bool] = {}
    t_start = time.time()

    for country in countries:
        ok = _run(
            [PYTHON, str(SCRIPTS_DIR / "validate_country.py")] + cfg_arg + ["--country", country],
            f"validate_country: {country}", args.fail_fast,
        )
        results[f"{country}_metrics"] = ok
        if ok and not args.skip_plots:
            ok_plot = _run(
                [PYTHON, str(SCRIPTS_DIR / "plot_agreement_map.py")] + cfg_arg + ["--country", country],
                f"plot_agreement_map: {country}", args.fail_fast,
            )
            results[f"{country}_plot"] = ok_plot

    # ── Cross-country summary ───────────────────────────────────────────────
    frames = []
    for country in countries:
        csv_path = Path(val_cfg["output_dir"]) / country / f"metrics_{country}_{rp}_{slr}.csv"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    if frames:
        summary = pd.concat(frames, ignore_index=True)
        summary_path = Path(val_cfg["output_dir"]) / f"summary_{rp}_{slr}.csv"
        retry_transient_io(summary_path.parent.mkdir, parents=True, exist_ok=True)
        atomic_write(summary_path, lambda f: summary.to_csv(f, index=False), mode="w", encoding="utf-8", newline="")
        print(f"\nCross-country summary written: {summary_path} ({len(summary)} row(s))")
    else:
        print("\nNo per-country metrics CSVs found - nothing to summarize.")

    total = time.time() - t_start
    print(f"\n{'═' * 60}")
    print(f"  Validation pipeline complete ({total / 60:.1f} min)")
    print(f"{'═' * 60}")
    for step, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {step}")
    if not all(results.values()):
        print("\n  Some steps failed — check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
