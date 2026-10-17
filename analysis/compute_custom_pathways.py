"""Evaluate a small, HAND-SPECIFIED set of adaptation pathway sequences
(not the full 324-chain enumeration compute_adaptation_pathways.py produces)
against every country/continent, for one SSP.

Exists because the user's requested sequences include two 4-stage chains,
beyond compute_adaptation_pathways.py's --max-stages=3 cap, so they're not
in its output CSVs (which only store final outcomes for the chains actually
enumerated, not the raw per-block EAI(t) curves needed to evaluate a new
one). Building block EAI(t) curves (baseline + 8 measure/design blocks, per
entity) is the expensive step shared with that script - see
src/visualization.py::build_growth_matrix_from_grid's own docstring for a
real fix applied there 2026-10-16 (a double Python-level MultiIndex-scalar-
lookup loop replaced with a vectorized unstack - confirmed bit-identical,
~10x faster per call) that makes this now take low minutes, not the ~1 hour
compute_adaptation_pathways.py's own run needed before that fix existed.

Writes --outdir/custom_pathways_{ssp}.csv: one row per (entity, sequence),
with each stage's own switch year and the final outcome - same schema
spirit as compute_adaptation_pathways.py's chain rows.

Usage:
    python analysis/compute_custom_pathways.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure --ssp SSP5 [--outdir <figures>] \\
        [--countries NGA THA EGY] [--continent-eai-threshold 5000000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_adaptation_pathways import (  # noqa: E402
    _eai_from_interp, _evaluate_chain, _first_crossing_after, _growth_matrix_path, _growth_path, _chain_label,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import get_data_catalog, load_config, retry_transient_io  # noqa: E402
from pathways import matrix_interpolator, slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from population_growth import load_ssp_growth_factors  # noqa: E402
from visualization import (  # noqa: E402
    CONTINENTS, build_growth_matrix_from_grid, iso_to_continent, load_growth_matrix_csv, load_slr_trajectories,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
LAST_SSP_YEAR = 2100

# The 4 user-specified sequences (measure, design) - "s"/"l" = SLR_500/SLR_1000.
# Sequence D's own last stage was written "avoid + protect (l) + retreat (l)" -
# read as protect_avoid_large THEN retreat_large (the same "> retreat (l)"
# final stage every other sequence but C ends on), not a single combined
# measure - flagged explicitly to the user, not assumed silently.
CUSTOM_SEQUENCES: dict[str, list[tuple[str, str]]] = {
    "A": [("avoid", "SLR_500"), ("avoid", "SLR_1000"), ("protect_avoid", "SLR_1000"), ("retreat", "SLR_1000")],
    "B": [("protect", "SLR_500"), ("protect", "SLR_1000"), ("retreat", "SLR_1000")],
    "C": [("retreat", "SLR_500"), ("retreat", "SLR_1000")],
    "D": [("protect", "SLR_500"), ("protect_avoid", "SLR_500"), ("protect_avoid", "SLR_1000"), ("retreat", "SLR_1000")],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True)
    parser.add_argument("--ssp", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--continents", nargs="+", default=CONTINENTS)
    parser.add_argument("--continent-eai-threshold", type=float, default=5_000_000.0)
    parser.add_argument("--continent-threshold-overrides", nargs="+", default=[],
                        help="per-continent overrides, e.g. Oceania=300000 (continents are far from uniform in "
                             "scale - Oceania never reaches even the 1,000,000 default, so its own panel would "
                             "otherwise always be trivial)")
    parser.add_argument("--to-year", type=int, default=None)
    args = parser.parse_args()
    continent_thr_overrides = {}
    for item in args.continent_threshold_overrides:
        name, _, val = item.partition("=")
        continent_thr_overrides[name] = float(val)

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
    ssp = args.ssp

    traj = load_slr_trajectories(viz["slr_trajectories_csv"])
    years = np.arange(2020, to_year + 1)
    slr_path = slr_pathway(traj, f"{ssp}_p50", years)

    catalog = get_data_catalog(_REPO_ROOT / cfg["paths"]["hydromt_data_catalog"], root=cfg["paths"]["root"])
    growth_df = load_ssp_growth_factors(Path(catalog.get_source("ssp_population_growth_factors").path))
    variants = (viz.get("extended_pathways", {}).get("variants") if to_year > LAST_SSP_YEAR else None) \
        or {"ssp": {"mode": "constant", "label": "SSP"}}
    v0 = next(iter(variants))
    ext0 = extend_growth_factors(growth_df, variants[v0], to_year=to_year)

    blocks = sorted({b for seq in CUSTOM_SEQUENCES.values() for b in seq})
    strategy_designs: list[tuple[str, str | None]] = [("baseline", None)] + blocks
    grids = {(s, d): load_growth_matrix_csv(_growth_matrix_path(exp_dir, s, d)) for s, d in strategy_designs
             if _growth_matrix_path(exp_dir, s, d).exists()}

    def rows_for_entity(entity: str, entity_type: str, thr: float, eai_by_block: dict[tuple, np.ndarray]):
        out = []
        baseline_tip = _first_crossing_after(eai_by_block[("baseline", None)], thr, years, years[0])
        if baseline_tip is None:
            for seq_id in CUSTOM_SEQUENCES:
                out.append({"entity": entity, "entity_type": entity_type, "ssp": ssp, "threshold": thr,
                           "sequence_id": seq_id, "chain": "(none needed)", "baseline_tip_year": None,
                           "stage_years": "", "outcome": "no_adaptation_needed", "final_tip_year": None})
            return out
        for seq_id, chain in CUSTOM_SEQUENCES.items():
            if not all(b in eai_by_block for b in chain):
                out.append({"entity": entity, "entity_type": entity_type, "ssp": ssp, "threshold": thr,
                           "sequence_id": seq_id, "chain": _chain_label(tuple(chain)),
                           "baseline_tip_year": f"{baseline_tip:.0f}", "stage_years": "",
                           "outcome": "missing_data", "final_tip_year": None})
                continue
            outcome, stage_years, final_tip = _evaluate_chain(tuple(chain), eai_by_block, thr, years, baseline_tip, to_year)
            out.append({"entity": entity, "entity_type": entity_type, "ssp": ssp, "threshold": thr,
                       "sequence_id": seq_id, "chain": _chain_label(tuple(chain)),
                       "baseline_tip_year": f"{baseline_tip:.0f}",
                       "stage_years": ";".join(f"{y:.0f}" for y in stage_years),
                       "outcome": outcome, "final_tip_year": final_tip})
        return out

    all_rows = []

    # ── Countries ────────────────────────────────────────────────────────────
    for iso in countries:
        thr = thr_by_iso.get(iso, default_thr)
        eai_by_block = {}
        for strat, design in strategy_designs:
            if (strat, design) not in grids:
                continue
            eai_grid, slr_mm, growth_fracs = grids[(strat, design)]
            if iso not in eai_grid.index:
                continue
            interp = matrix_interpolator(build_growth_matrix_from_grid(eai_grid.loc[[iso]], slr_mm, growth_fracs),
                                         slr_mm, growth_fracs)
            gp = _growth_path(ext0, ssp, iso, years) - 1.0
            eai_by_block[(strat, design)] = _eai_from_interp(interp, slr_path, gp)
        all_rows += rows_for_entity(iso, "country", thr, eai_by_block)
        print(f"  {iso}: done", flush=True)

    # ── Continents ───────────────────────────────────────────────────────────
    baseline_grid, _, _ = grids[("baseline", None)]
    all_isos = baseline_grid.index.tolist()
    cont_of = iso_to_continent(all_isos)
    continent_members = {c: [i for i, k in cont_of.items() if k == c and i in all_isos] for c in args.continents}

    for cont in args.continents:
        members = continent_members[cont]
        if not members:
            continue
        thr = continent_thr_overrides.get(cont, args.continent_eai_threshold)
        eai_by_block = {}
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
            total = np.zeros(len(years))
            for i in present:
                gp = _growth_path(ext0, ssp, i, years) - 1.0
                total += _eai_from_interp(interps[i], slr_path, gp)
            eai_by_block[(strat, design)] = total
        all_rows += rows_for_entity(cont, "continent", thr, eai_by_block)
        print(f"  {cont}: done ({len(members)} countries)", flush=True)

    df = pd.DataFrame(all_rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    out_path = out_dir / f"custom_pathways_{ssp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
