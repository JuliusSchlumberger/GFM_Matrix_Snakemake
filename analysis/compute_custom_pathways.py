"""Evaluate a hand-specified set of adaptation pathway sequences for one SSP.

Covers 4-stage chains beyond compute_adaptation_pathways.py's --max-stages=3 cap.
Writes --out-dir/custom_pathways_{ssp}.csv, one row per (entity, sequence).

Usage:
    python analysis/compute_custom_pathways.py --config snakemake_workflow/config/config.yml \\
        --expdir <merged_results>/exposure --ssp SSP5 [--out-dir <figures>] \\
        [--countries NGA THA EGY] [--continent-eai-threshold 5000000]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    _country_interpolator, _eai_from_interp, _growth_matrix_path, _growth_path, load_pathway_setup,
)
from compute_adaptation_pathways import _chain_label, _evaluate_chain, _first_crossing_after  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pathways import slr_pathway  # noqa: E402
from population_extension import extend_growth_factors  # noqa: E402
from analysis_figures import CONTINENTS, iso_to_continent, load_growth_matrix_csv  # noqa: E402

# D's last stage "avoid + protect (l) + retreat (l)" is read as protect_avoid_large then
# retreat_large, not one combined measure.
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
    parser.add_argument("--out-dir", "--outdir", dest="outdir", default=None)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--continents", nargs="+", default=CONTINENTS)
    parser.add_argument("--continent-eai-threshold", type=float, default=5_000_000.0)
    parser.add_argument("--continent-threshold-overrides", nargs="+", default=[],
                        help="per-continent overrides, e.g. Oceania=300000 (continents are far from uniform in "
                             "scale - Oceania never reaches even the --continent-eai-threshold default, so its own panel would "
                             "otherwise always be trivial)")
    parser.add_argument("--to-year", type=int, default=None)
    args = parser.parse_args()
    continent_thr_overrides = {}
    for item in args.continent_threshold_overrides:
        name, _, val = item.partition("=")
        continent_thr_overrides[name] = float(val)

    st = load_pathway_setup(args)
    exp_dir, out_dir, countries, to_year, years = st.exp_dir, st.out_dir, st.countries, st.to_year, st.years
    default_thr, thr_by_iso, variants = st.default_thr, st.thr_by_iso, st.variants
    ssp = args.ssp
    slr_path = slr_pathway(st.traj, f"{ssp}_p50", years)

    v0 = next(iter(variants))
    ext0 = extend_growth_factors(st.growth_df, variants[v0], to_year=to_year)

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

    for iso in countries:
        thr = thr_by_iso.get(iso, default_thr)
        eai_by_block = {}
        for strat, design in strategy_designs:
            if (strat, design) not in grids:
                continue
            if iso not in grids[(strat, design)][0].index:
                continue
            interp = _country_interpolator(grids[(strat, design)], iso)
            gp = _growth_path(ext0, ssp, iso, years) - 1.0
            eai_by_block[(strat, design)] = _eai_from_interp(interp, slr_path, gp)
        all_rows += rows_for_entity(iso, "country", thr, eai_by_block)
        print(f"  {iso}: done", flush=True)

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
            present = [i for i in members if i in grids[(strat, design)][0].index]
            interps = {i: _country_interpolator(grids[(strat, design)], i) for i in present}
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
