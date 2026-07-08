"""Generate burning-ember (impact matrix) figures from compute_exposure_analysis.py outputs.

Two data sources feed each figure:
  - exposure_{label}_growth_matrix.csv — scenario-neutral SLR x generic-growth-rate
                                          grid (File 2), used for the heatmap/contour
                                          background (including avoid). The SLR axis
                                          is already dense (linearly interpolated onto
                                          visualization.slr_interp's grid at
                                          file-creation time) — no plot-time
                                          interpolation needed.
  - Real SSP growth factors + slr_trajectories_csv — used only for the SSP
                                          trajectory overlay lines/markers, independent
                                          of the per-scenario CSVs.

X-axis: global mean SLR (m). Y-axis: population growth rate relative to 2020 (%),
−50 % → +150 %. Colour: EAI (Expected Annual Impact, people exposed beyond
protection). SSP overlays: coloured lines tracing each SSP's real growth/SLR
trajectory through the scenario-neutral matrix, stopping at the last highlight
year. Second subplot: P17-P83 SLR uncertainty per SSP at each highlight year.

Usage:
    python snakemake_workflow/analysis/plot_burning_ember.py \\
        --config  snakemake_workflow/config/config.yml \\
        --expdir  D:/GFM/merged_results/exposure \\
        [--outdir D:/GFM/figures/burning_ember]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visualization import (  # noqa: E402
    load_growth_matrix_csv,
    build_growth_matrix_from_grid,
    plot_burning_ember,
    load_slr_trajectories,
)
from population_growth import (  # noqa: E402
    load_ssp_growth_factors,
    interpolate_growth_factor,
    _build_name_to_iso,
)


def _expand(s, root, code_root=""):
    return str(s).replace("{root}", root).replace("{code_root}", code_root or root)


def _ssp_growth_for_isos(growth_df, iso_codes: list[str], ssp: str, year: int) -> float:
    """Mean growth factor across a list of ISO codes."""
    factors = [interpolate_growth_factor(growth_df, ssp, iso, year, 1.0) for iso in iso_codes]
    return float(np.mean(factors)) if factors else 1.0


def _build_ssp_growth(
    growth_df, slr_traj, ssps: list[str], isos: list[str], mean: bool,
) -> dict[str, pd.Series]:
    """{SSP: pd.Series(year -> growth factor)} for the trajectory overlay.

    `mean=True` averages across all `isos` (global/continent aggregates);
    `mean=False` uses `isos[0]` directly (single-country plots).
    """
    result: dict[str, pd.Series] = {}
    if slr_traj is None or growth_df is None:
        return result
    for ssp in ssps:
        if ssp not in slr_traj.columns:
            continue
        yrs = sorted(c for c in slr_traj.index if isinstance(c, (int, float)))
        if mean:
            result[ssp] = pd.Series(
                {yr: _ssp_growth_for_isos(growth_df, isos, ssp, int(yr)) for yr in yrs}
            )
        else:
            iso = isos[0]
            result[ssp] = pd.Series(
                {yr: interpolate_growth_factor(growth_df, ssp, iso, int(yr), 1.0) for yr in yrs}
            )
    return result


def _make_ember(
    matrix, slr_mm, growth_rates, title, slr_traj, ssp_growth_series,
    slr_uncertainty, highlight_years, viz, dpi, vmax, outpath,
):
    fig = plot_burning_ember(
        matrix, slr_mm, growth_rates, title=title,
        slr_trajectories=slr_traj, ssp_growth=ssp_growth_series,
        slr_uncertainty=slr_uncertainty,
        cmap=viz.get("cmap_ember", "YlOrRd"), vmax=vmax, dpi=dpi,
        ssp_colors=viz.get("ssp_colors"), highlight_years=highlight_years,
        ssp_rcp_codes=viz.get("ssp_rcp_codes"),
    )
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", required=True, help="directory with exposure_*_growth_matrix.csv files")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    root = cfg.get("paths", {}).get("root", "")
    code_root = cfg.get("paths", {}).get("code_root", root)
    ex = lambda s: _expand(s, root, code_root)

    viz = cfg.get("visualization", {})
    adapt_cfg = cfg.get("adaptation", {})
    pg_cfg = cfg.get("population_growth", {})

    slr_intensities = adapt_cfg.get("slr_intensities", [])
    ssps = pg_cfg.get("ssps", ["SSP1", "SSP2", "SSP3", "SSP5"])
    highlight_years = [2050, 2100]

    exp_dir = Path(ex(args.expdir))
    out_dir = Path(ex(args.outdir or str(
        Path(ex(viz.get("output_dir", "{root}/figures"))) / "burning_ember"
    )))

    # SSP SLR trajectories: median (p50) drives the overlay line; full
    # p17/p50/p83 (at the highlight years only) drives the uncertainty subplot.
    traj_path = ex(viz.get("slr_trajectories_csv", ""))
    slr_traj = None
    slr_uncertainty: dict[str, pd.DataFrame] = {}
    if traj_path and Path(traj_path).exists():
        traj_full = load_slr_trajectories(traj_path)
        p50_cols = {c: c.replace("_p50", "") for c in traj_full.columns if c.endswith("_p50")}
        slr_traj = traj_full[list(p50_cols)].rename(columns=p50_cols)
        for ssp in ssps:
            cols = {f"{ssp}_p17": "p17", f"{ssp}_p50": "p50", f"{ssp}_p83": "p83"}
            present = {c: n for c, n in cols.items() if c in traj_full.columns}
            if len(present) == 3:
                slr_uncertainty[ssp] = traj_full[list(present)].rename(columns=present)

    # SSP growth factors for the trajectory overlay only (the heatmap
    # background comes entirely from the growth_matrix CSV — see below).
    xlsx_path = Path(ex(pg_cfg.get("factors_xlsx", "{root}/inputs/SSPs/getting_SSP_population_growth_factors.xlsx")))
    growth_df = load_ssp_growth_factors(xlsx_path) if xlsx_path.exists() else None

    dpi = int(viz.get("dpi", 200))

    # ISO -> country name, for per-country plot titles (Natural Earth, via
    # the same lookup population_growth.py already builds for SSP matching).
    iso_to_name = {v: k for k, v in _build_name_to_iso().items()}

    # Discover scenario growth-matrix CSVs (File 2 — the only file this
    # script needs; unlike File 1/3, avoid has one too).
    scenario_files: dict[str, Path] = {
        "baseline": exp_dir / "exposure_baseline_growth_matrix.csv"
    }
    for slr_int in slr_intensities:
        for meas in ["protect", "retreat", "avoid"]:
            p = exp_dir / f"exposure_{meas}_{slr_int}_growth_matrix.csv"
            if p.exists():
                scenario_files[f"{meas}_{slr_int}"] = p

    for scenario_label, growth_matrix_path in scenario_files.items():
        if not growth_matrix_path.exists():
            print(f"  Skipping {scenario_label}: missing {growth_matrix_path.name}.")
            continue

        eai_grid, slr_mm_grid, growth_fracs = load_growth_matrix_csv(growth_matrix_path)
        growth_rates = np.array(growth_fracs)

        # ── Global aggregate ──────────────────────────────────────────────────
        (out_dir / "global").mkdir(parents=True, exist_ok=True)
        all_isos = eai_grid.index.tolist()
        ssp_growth_global = _build_ssp_growth(growth_df, slr_traj, ssps, all_isos, mean=True)

        matrix = build_growth_matrix_from_grid(eai_grid, slr_mm_grid, growth_fracs)
        _make_ember(matrix, slr_mm_grid, growth_rates, f"Global — {scenario_label}",
                    slr_traj, ssp_growth_global, slr_uncertainty, highlight_years, viz, dpi,
                    float(matrix.max()) or 1.0,
                    out_dir / "global" / f"burning_ember_global_{scenario_label}.png")

        # ── Per country ───────────────────────────────────────────────────────
        country_dir = out_dir / "country"
        country_dir.mkdir(exist_ok=True)
        for iso in eai_grid.index:
            row_grid = eai_grid.loc[[iso]]
            matrix_c = build_growth_matrix_from_grid(row_grid, slr_mm_grid, growth_fracs)
            if matrix_c.max() == 0:
                continue
            ssp_growth_c = _build_ssp_growth(growth_df, slr_traj, ssps, [iso], mean=False)
            title = f"{iso_to_name.get(iso, iso)} — {scenario_label}"
            _make_ember(matrix_c, slr_mm_grid, growth_rates, title,
                        slr_traj, ssp_growth_c, slr_uncertainty, highlight_years, viz, dpi,
                        2_000_000.0,
                        country_dir / f"burning_ember_{iso}_{scenario_label}.png")

        print(f"  {scenario_label}: done")

    print(f"\nFigures written to: {out_dir}")


if __name__ == "__main__":
    main()
