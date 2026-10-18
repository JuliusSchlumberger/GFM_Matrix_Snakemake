"""Tabulate the median tipping year per strategy / design / SSP for countries and continents.

Median over 9 pathway samples (SLR p17/p50/p83 x 3 post-2100 population variants),
right-censored: non-crossing samples count as +inf, so if fewer than 5 of 9 cross the
median is ">{to_year}" rather than a mean of the early crossers. Continent EAI is the
sum of member-country EAI time series (each on its own population path), not of their
growth matrices. --continent-eai-threshold (default 5,000,000) is an arbitrary round
starting point larger than the per-country default, not a finding.

Writes tipping_points_{countries,continents}_median.csv to --outdir (default:
visualization.output_dir).

Usage:
    python analysis/compute_tipping_points_summary.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure [--outdir <figures>] \\
        [--countries NGA THA EGY] [--continent-eai-threshold 5000000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _eai_from_interp, _first_crossing, _growth_matrix_path, _growth_path  # noqa: E402
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import matrix_interpolator, slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import load_ssp_growth_factors  # noqa: E402
from analysis_figures import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, iso_to_continent, load_growth_matrix_csv, load_slr_trajectories,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
MEASURES = ["protect", "avoid", "retreat", "protect_avoid"]
LAST_SSP_YEAR = 2100


def _median_crossing(
    samples: list[float | None], to_year: int,
) -> tuple[str, int]:
    censored = [s if s is not None else np.inf for s in samples]
    n_reached = sum(s is not None for s in samples)
    med = float(np.median(censored))
    label = f">{to_year}" if not np.isfinite(med) else f"{med:.0f}"
    return label, n_reached


def _country_eai_samples(
    interp, ssp: str,
    slr_paths: dict[tuple[str, str], np.ndarray], growth_paths: dict[tuple[str, str], np.ndarray],
    variants: dict, threshold: float, years: np.ndarray,
) -> list[float | None]:
    out = []
    for v in variants:
        for p in ("p17", "p50", "p83"):
            eai = _eai_from_interp(interp, slr_paths[(ssp, p)], growth_paths[(ssp, v)])
            out.append(_first_crossing(eai, threshold, years))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--countries", nargs="+", default=None, help="default: visualization.tipping_points.countries")
    parser.add_argument("--continents", nargs="+", default=CONTINENTS)
    parser.add_argument("--continent-eai-threshold", type=float, default=5_000_000.0,
                        help="see module docstring - no principled default, override as needed")
    parser.add_argument("--to-year", type=int, default=None, help="default: visualization.tipping_points.to_year")
    args = parser.parse_args()

    cfg = load_config(args.config)
    viz = cfg["visualization"]
    tp = viz.get("tipping_points", {})
    exp_dir = Path(args.expdir)
    out_dir = Path(args.outdir or viz.get("output_dir", f"{cfg['paths']['root']}/figures"))
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

    countries = args.countries or tp.get("countries", ["VNM", "NLD", "NGA"])
    default_thr = float(tp.get("eai_threshold", 1_000_000))
    thr_by_iso = {k: float(v) for k, v in (tp.get("per_country_eai_threshold") or {}).items()}
    to_year = args.to_year or int(tp.get("to_year", LAST_SSP_YEAR))
    ssps = list(tp.get("ssps", cfg.get("population_growth", {}).get("ssps", ["SSP2", "SSP5"])))
    strategies = [s for s in tp.get("strategies", ["baseline"] + MEASURES)
                  if s == "baseline" or s in MEASURES]
    designs = list(cfg.get("adaptation", {}).get("slr_intensities", ["SLR_500", "SLR_1000"]))

    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    ssps = [s for s in ssps if all(f"{s}_{p}" in traj.columns for p in ("p17", "p50", "p83"))]
    years = np.arange(2020, to_year + 1)
    slr_paths = {(s, p): slr_pathway(traj, f"{s}_{p}", years) for s in ssps for p in ("p17", "p50", "p83")}

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    variants = (viz.get("extended_pathways", {}).get("variants") if to_year > LAST_SSP_YEAR else None) \
        or {"ssp": {"mode": "constant", "label": "SSP"}}
    ext = {v: extend_growth_factors(growth_df, spec, to_year=to_year) for v, spec in variants.items()}

    strategy_designs: list[tuple[str, str | None]] = [("baseline", None)]
    for meas in strategies:
        if meas == "baseline":
            continue
        for d in designs:
            if _growth_matrix_path(exp_dir, meas, d).exists():
                strategy_designs.append((meas, d))

    grids = {(s, d): load_growth_matrix_csv(_growth_matrix_path(exp_dir, s, d)) for s, d in strategy_designs}

    country_rows = []
    for iso in countries:
        thr = thr_by_iso.get(iso, default_thr)
        growth_paths = {(s, v): _growth_path(ext[v], s, iso, years) - 1.0 for s in ssps for v in variants}
        for strat, design in strategy_designs:
            eai_grid, slr_mm, growth_fracs = grids[(strat, design)]
            if iso not in eai_grid.index:
                continue
            matrix = build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs)
            interp = matrix_interpolator(matrix, slr_mm, growth_fracs)
            for ssp in ssps:
                samples = _country_eai_samples(interp, ssp, slr_paths, growth_paths, variants, thr, years)
                label, n_reached = _median_crossing(samples, to_year)
                country_rows.append({
                    "entity": iso, "entity_type": "country", "threshold": thr, "strategy": strat,
                    "design_intensity": design or "-", "ssp": ssp,
                    "n_samples_reached": f"{n_reached}/{len(samples)}", "median_tip_year": label,
                })
        print(f"  {iso}: done", flush=True)

    # Members resolved once from the baseline grid; measure grids cover the same countries.
    baseline_grid, _, _ = grids[("baseline", None)]
    all_isos = baseline_grid.index.tolist()
    cont_of = iso_to_continent(all_isos)
    continent_members = {c: [i for i, k in cont_of.items() if k == c and i in all_isos] for c in args.continents}

    continent_rows = []
    for cont in args.continents:
        members = continent_members[cont]
        if not members:
            continue
        thr = args.continent_eai_threshold
        growth_paths_by_iso = {
            iso: {(s, v): _growth_path(ext[v], s, iso, years) - 1.0 for s in ssps for v in variants}
            for iso in members
        }
        for strat, design in strategy_designs:
            eai_grid, slr_mm, growth_fracs = grids[(strat, design)]
            present = [i for i in members if i in eai_grid.index]
            interps = {
                i: matrix_interpolator(build_growth_matrix_from_grid(eai_grid.loc[[i]], slr_mm, growth_fracs),
                                       slr_mm, growth_fracs)
                for i in present
            }
            for ssp in ssps:
                samples = []
                for v in variants:
                    for p in ("p17", "p50", "p83"):
                        total = np.zeros(len(years))
                        for i in present:
                            total += _eai_from_interp(interps[i], slr_paths[(ssp, p)], growth_paths_by_iso[i][(ssp, v)])
                        samples.append(_first_crossing(total, thr, years))
                label, n_reached = _median_crossing(samples, to_year)
                continent_rows.append({
                    "entity": cont, "entity_type": "continent", "threshold": thr, "strategy": strat,
                    "design_intensity": design or "-", "ssp": ssp, "n_countries_included": len(present),
                    "n_samples_reached": f"{n_reached}/{len(samples)}", "median_tip_year": label,
                })
        print(f"  {cont}: done ({len(members)} countries)", flush=True)

    country_df = pd.DataFrame(country_rows)
    continent_df = pd.DataFrame(continent_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 300)
    print("\n=== Countries ===")
    print(country_df.to_string(index=False))
    print("\n=== Continents ===")
    print(continent_df.to_string(index=False))

    country_out = out_dir / "tipping_points_countries_median.csv"
    continent_out = out_dir / "tipping_points_continents_median.csv"
    country_df.to_csv(country_out, index=False)
    continent_df.to_csv(continent_out, index=False)
    print(f"\nWritten: {country_out} and {continent_out}")


if __name__ == "__main__":
    main()
