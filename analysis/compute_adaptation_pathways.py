"""Evaluate adaptation-pathway chains: when does each chain of measures tip?

Dynamic Adaptive Policy Pathways (Haasnoot et al. 2013): a measure holds until its EAI
reaches the threshold, then the chain switches to the next block. Blocks are 4 measures
x 2 designs (SLR_500="small", SLR_1000="large"); a chain is up to --max-stages distinct
blocks, never using a family's large design before its small one (324 chains for 3
stages). A chain succeeds if a stage's EAI never re-crosses before --to-year. The
SLR/population trajectory is measure-independent, so one EAI(t) curve per block is
reused. Single representative pathway per SSP: p50 SLR, first population variant.
Entities whose baseline never tips are reported as "no_adaptation_needed".

Writes pathways_countries.csv and pathways_continents.csv to --outdir (default:
visualization.output_dir); `stage_years` is ';'-separated, one switch year per stage.

Usage:
    python analysis/compute_adaptation_pathways.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure [--outdir <figures>] \\
        [--countries NGA THA EGY] [--continent-eai-threshold 5000000] [--max-stages 3]
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import _eai_from_interp, _growth_matrix_path, _growth_path  # noqa: E402
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import matrix_interpolator, slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import load_ssp_growth_factors  # noqa: E402
from analysis_figures import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, iso_to_continent, load_growth_matrix_csv, load_slr_trajectories,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
MEASURES = ["avoid", "protect", "retreat", "protect_avoid"]
DESIGN_LABEL = {"SLR_500": "small", "SLR_1000": "large"}
LAST_SSP_YEAR = 2100


def _first_crossing_after(eai: np.ndarray, threshold: float, years: np.ndarray, start_year: float) -> float | None:
    start_idx = int(np.searchsorted(years, start_year, side="left"))
    if start_idx >= len(years):
        return None
    above = np.nonzero(eai[start_idx:] >= threshold)[0]
    if len(above) == 0:
        return None
    k = start_idx + int(above[0])
    if k == start_idx:
        return float(years[k])
    e0, e1 = eai[k - 1], eai[k]
    return float(years[k - 1] + (threshold - e0) / (e1 - e0))


def _all_chains(blocks: list[tuple[str, str]], max_stages: int) -> list[tuple[tuple[str, str], ...]]:
    chains = []
    for n in range(1, max_stages + 1):
        for combo in itertools.combinations(blocks, n):
            for perm in itertools.permutations(combo):
                pos = {b: i for i, b in enumerate(perm)}
                ok = True
                for meas in {m for m, _ in combo}:
                    small, large = (meas, "SLR_500"), (meas, "SLR_1000")
                    if small in pos and large in pos and pos[small] > pos[large]:
                        ok = False
                        break
                if ok:
                    chains.append(perm)
    return chains


def _chain_label(chain: tuple[tuple[str, str], ...]) -> str:
    return " > ".join(f"{m}_{DESIGN_LABEL[d]}" for m, d in chain)


def _evaluate_chain(
    chain: tuple[tuple[str, str], ...], eai_by_block: dict[tuple[str, str], np.ndarray],
    threshold: float, years: np.ndarray, start_year: float, to_year: int,
) -> tuple[str, list[float], str | None]:
    t = start_year
    stage_years = []
    for i, block in enumerate(chain, start=1):
        eai = eai_by_block[block]
        nxt = _first_crossing_after(eai, threshold, years, t)
        if nxt is None:
            return f"success_by_stage_{i}", stage_years, None
        stage_years.append(nxt)
        t = nxt
    return "fails", stage_years, f"{stage_years[-1]:.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--countries", nargs="+", default=None, help="default: visualization.tipping_points.countries")
    parser.add_argument("--continents", nargs="+", default=CONTINENTS)
    parser.add_argument("--continent-eai-threshold", type=float, default=5_000_000.0)
    parser.add_argument("--to-year", type=int, default=None, help="default: visualization.tipping_points.to_year")
    parser.add_argument("--max-stages", type=int, default=3)
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
    designs = list(cfg.get("adaptation", {}).get("slr_intensities", ["SLR_500", "SLR_1000"]))

    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    ssps = [s for s in ssps if all(f"{s}_{p}" in traj.columns for p in ("p17", "p50", "p83"))]
    years = np.arange(2020, to_year + 1)
    slr_path_p50 = {s: slr_pathway(traj, f"{s}_p50", years) for s in ssps}

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    variants = (viz.get("extended_pathways", {}).get("variants") if to_year > LAST_SSP_YEAR else None) \
        or {"ssp": {"mode": "constant", "label": "SSP"}}
    v0 = next(iter(variants))
    ext0 = extend_growth_factors(growth_df, variants[v0], to_year=to_year)

    blocks = [(m, d) for m in MEASURES for d in designs]
    chains = _all_chains(blocks, args.max_stages)
    print(f"{len(blocks)} building blocks, {len(chains)} candidate chains (max {args.max_stages} stages)", flush=True)

    strategy_designs: list[tuple[str, str | None]] = [("baseline", None)] + [(m, d) for m, d in blocks]
    grids = {(s, d): load_growth_matrix_csv(_growth_matrix_path(exp_dir, s, d)) for s, d in strategy_designs
             if _growth_matrix_path(exp_dir, s, d).exists()}

    def rows_for_entity(entity: str, entity_type: str, thr: float, eai_by_ssp: dict[str, dict[tuple, np.ndarray]]):
        out = []
        for ssp in ssps:
            eai_by_block = eai_by_ssp[ssp]
            baseline_tip = _first_crossing_after(eai_by_block[("baseline", None)], thr, years, years[0])
            if baseline_tip is None:
                out.append({"entity": entity, "entity_type": entity_type, "ssp": ssp, "threshold": thr,
                           "baseline_tip_year": None, "chain": "(none needed)", "stage_years": "",
                           "outcome": "no_adaptation_needed", "final_tip_year": None})
                continue
            for chain in chains:
                if not all(b in eai_by_block for b in chain):
                    continue
                outcome, stage_years, final_tip = _evaluate_chain(chain, eai_by_block, thr, years, baseline_tip, to_year)
                out.append({"entity": entity, "entity_type": entity_type, "ssp": ssp, "threshold": thr,
                           "baseline_tip_year": f"{baseline_tip:.0f}", "chain": _chain_label(chain),
                           "stage_years": ";".join(f"{y:.0f}" for y in stage_years),
                           "outcome": outcome, "final_tip_year": final_tip})
        return out

    country_rows = []
    for iso in countries:
        thr = thr_by_iso.get(iso, default_thr)
        eai_by_ssp: dict[str, dict[tuple, np.ndarray]] = {ssp: {} for ssp in ssps}
        for strat, design in strategy_designs:
            if (strat, design) not in grids:
                continue
            eai_grid, slr_mm, growth_fracs = grids[(strat, design)]
            if iso not in eai_grid.index:
                continue
            interp = matrix_interpolator(build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs),
                                         slr_mm, growth_fracs)
            for ssp in ssps:
                gp = _growth_path(ext0, ssp, iso, years) - 1.0
                eai_by_ssp[ssp][(strat, design)] = _eai_from_interp(interp, slr_path_p50[ssp], gp)
        country_rows += rows_for_entity(iso, "country", thr, eai_by_ssp)
        print(f"  {iso}: done", flush=True)

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
        eai_by_ssp = {ssp: {} for ssp in ssps}
        for strat, design in strategy_designs:
            if (strat, design) not in grids:
                continue
            eai_grid, slr_mm, growth_fracs = grids[(strat, design)]
            present = [i for i in members if i in eai_grid.index]
            interps = {
                i: matrix_interpolator(build_growth_matrix_from_grid(eai_grid.loc[[i]], slr_mm, growth_fracs),
                                       slr_mm, growth_fracs)
                for i in present
            }
            for ssp in ssps:
                total = np.zeros(len(years))
                for i in present:
                    gp = _growth_path(ext0, ssp, i, years) - 1.0
                    total += _eai_from_interp(interps[i], slr_path_p50[ssp], gp)
                eai_by_ssp[ssp][(strat, design)] = total
        continent_rows += rows_for_entity(cont, "continent", thr, eai_by_ssp)
        print(f"  {cont}: done ({len(members)} countries)", flush=True)

    country_df = pd.DataFrame(country_rows)
    continent_df = pd.DataFrame(continent_rows)

    country_out = out_dir / "pathways_countries.csv"
    continent_out = out_dir / "pathways_continents.csv"
    country_df.to_csv(country_out, index=False)
    continent_df.to_csv(continent_out, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Shortest successful chain per (entity, ssp) ===")
    for df in (country_df, continent_df):
        succ = df[df["outcome"].str.startswith("success", na=False) | (df["outcome"] == "no_adaptation_needed")]
        for (entity, ssp), g in succ.groupby(["entity", "ssp"]):
            if (g["outcome"] == "no_adaptation_needed").any():
                print(f"  {entity} / {ssp}: no adaptation needed (baseline never tips by {to_year})")
                continue
            g = g.copy()
            g["n_stages"] = g["chain"].str.count(">") + 1
            best = g.sort_values("n_stages").iloc[0]
            print(f"  {entity} / {ssp}: {best['chain']}  (switches at {best['stage_years']})")
        no_success = df.groupby(["entity", "ssp"]).filter(
            lambda g: not ((g["outcome"].str.startswith("success", na=False)).any() or (g["outcome"] == "no_adaptation_needed").any())
        )
        for entity, ssp in no_success[["entity", "ssp"]].drop_duplicates().itertuples(index=False):
            print(f"  {entity} / {ssp}: NO chain (up to {args.max_stages} stages) reaches beyond {to_year}")

    print(f"\nWritten: {country_out} and {continent_out}")


if __name__ == "__main__":
    main()
