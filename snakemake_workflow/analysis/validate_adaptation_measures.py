"""Synthetic sense-check of the baseline/protect/avoid/retreat exposure formulas.

Not part of the Snakemake DAG or run_analysis.py - a standalone diagnostic
that runs the REAL production formulas from src/exposure_analysis.py
(protect_exposure_grid, compute_retreat_capacity/compute_retreat,
compute_avoid_redirected/compute_avoid) against a small synthetic 100x100
grid instead of real pipeline outputs, so the adaptation measures' relative
behaviour can be checked quickly without a full run.

Grid setup
----------
One fixed exposure/population grid (E0) and one sequence of flood-fraction
grids, one per synthetic "severity" level (mapped directly onto the ember
plot's SLR axis - no return-period dimension: EAI is simplified to the
adaptation formula's output for that single representative severity level,
skipping the real pipeline's trapezoidal RP integration). The whole grid is
treated as a single "country" (a constant geo_ids array + a 1-entry
iso_lookup) so compute_retreat/compute_avoid's real country-redistribution
logic runs unmodified.

Flood-fraction generation is a seeded cellular-automaton-style contagion
process: at each more-severe step, already-wet cells' flood_fraction grows
multiplicatively (capped at 1.0), and each dry cell 4-connected to a wet
cell gets a fixed probability of activating. A bigger wet frontier yields
proportionally more new activations the next step, compounding both cell
count and per-cell value into roughly exponential growth in total flooded
extent.

Output: a 4x3 grid of burning-ember plots - columns = baseline/protect/
avoid/retreat, rows = three "modes" describing which axis of the
(severity x growth) EAI matrix is actually left free to vary:
  - SLR driven:        growth_factor forced to 1.0 for every row (both of
                        avoid's growth-dependent terms - organic-growth and
                        redirected-growth - use max(0, growth_factor - 1),
                        so they vanish here and avoid should collapse onto
                        protect exactly).
  - Population driven: severity forced to the level-0 (mostly dry) grid for
                        every column; only growth varies.
  - Mixed drivers:      both vary for real, the closest analogue to
                        production behaviour.
plot_burning_ember() itself is reused unmodified (one call per panel); since
it always builds its own standalone figure, the 12 individual PNGs are also
stitched into one 4x3 contact sheet for convenience.

Usage:
    python snakemake_workflow/analysis/validate_adaptation_measures.py
        [--outdir D:/GFM/figures/validation]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exposure_analysis import (  # noqa: E402
    compute_avoid,
    compute_avoid_redirected,
    compute_retreat,
    compute_retreat_capacity,
    protect_exposure_grid,
)
from visualization import plot_burning_ember  # noqa: E402

GRID_SIZE = 100
SEED = 42
N_SEVERITY = 7
SLR_MM = np.linspace(0, 1500, N_SEVERITY)
GROWTH_RATES = np.arange(-0.5, 1.5 + 1e-9, 0.1)

# Design/protection standard: a single flood_fraction cutoff, applied
# uniformly across the grid (a spatially-flat stand-in for the real
# pipeline's build_adapt_protection_fraction map). Must sit strictly between
# SEED_WET_VALUE and INITIAL_WET_VALUE below - see their comments for why.
DESIGN_STANDARD = 0.20

# Flood-fraction contagion parameters (see module docstring).
# SEED_WET_VALUE (> DESIGN_STANDARD) is the level-0 seed patch's starting
# value: it's already "established" flooding that exceeds the design
# standard from severity 0 onward, which is what gives "Population driven"
# (severity pinned to level 0) a non-zero protect/avoid/retreat signal -
# without a value already above the standard at severity 0, every
# adaptation measure reads exactly 0 there regardless of population growth.
# INITIAL_WET_VALUE (< DESIGN_STANDARD) is what a cell newly reached by the
# contagion frontier starts at: below standard, so a freshly-flooded cell
# needs a few growth steps before it breaches protection - this is what
# makes protect/avoid/retreat diverge from baseline at low-to-mid severity
# and converge again as severity climbs (once most of the domain has had
# time to grow past the standard), instead of exactly tracking baseline at
# every severity level the way a self-referential threshold (e.g. "0.7x
# this same cell's own level-0 value") would.
SEED_WET_VALUE = 0.35
INITIAL_WET_VALUE = 0.12
ACTIVATION_PROB = 0.30     # per-step chance a dry neighbour of a wet cell activates
WET_GROWTH_FACTOR = 1.3    # per-step multiplicative growth of already-wet cells

MEASURES = ["baseline", "protect", "avoid", "retreat"]
MODES = ["SLR driven", "Population driven", "Mixed drivers"]

# Single-"country" bookkeeping: the whole grid is one region, so
# compute_retreat/compute_avoid's real redistribution logic (which sums and
# redistributes within each country) pools across the entire grid.
_GEO_IDS = np.zeros((GRID_SIZE, GRID_SIZE), dtype="int32")
_ISO_LOOKUP = {0: "SYN"}


def generate_population(rng: np.random.Generator) -> np.ndarray:
    """Fixed synthetic exposure/population grid, reused by every panel."""
    return rng.lognormal(mean=3.0, sigma=1.0, size=(GRID_SIZE, GRID_SIZE))


def generate_flood_fraction_sequence(rng: np.random.Generator) -> list[np.ndarray]:
    """One flood_fraction grid per severity level, via seeded contagion growth."""
    ff = np.zeros((GRID_SIZE, GRID_SIZE), dtype="float64")
    ff[GRID_SIZE // 2 - 3 : GRID_SIZE // 2 + 3, 0:5] = SEED_WET_VALUE  # seed patch

    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    sequence = [ff.copy()]
    for _ in range(1, N_SEVERITY):
        wet = ff > 0
        ff = np.where(wet, np.minimum(ff * WET_GROWTH_FACTOR, 1.0), ff)
        dilated = ndimage.binary_dilation(wet, structure=structure)
        candidate = dilated & ~wet
        draw = rng.random(ff.shape)
        newly_wet = candidate & (draw < ACTIVATION_PROB)
        ff = np.where(newly_wet, INITIAL_WET_VALUE, ff)
        sequence.append(ff.copy())
    return sequence


def compute_eai_matrix(
    measure: str,
    mode: str,
    ff_sequence: list[np.ndarray],
    population: np.ndarray,
) -> np.ndarray:
    """(n_growth, n_severity) global EAI matrix for one (measure, mode) panel.

    baseline/protect/retreat are exactly linear in population growth (the
    design-threshold comparison never depends on population), so their EAI
    is computed once per severity level at growth_factor=1 and scaled
    afterward - mirrors apply_growth_rates_to_eai's production shortcut.
    avoid is genuinely non-linear in growth (compute_avoid_redirected), so
    it's recomputed at every (severity, growth) cell directly.
    """
    n_growth = len(GROWTH_RATES)
    matrix = np.zeros((n_growth, N_SEVERITY))

    apf_grid = (
        np.zeros((GRID_SIZE, GRID_SIZE))
        if measure == "baseline"
        else np.full((GRID_SIZE, GRID_SIZE), DESIGN_STANDARD)
    )

    if measure == "retreat":
        eff_pop = compute_retreat_capacity(apf_grid, population, _GEO_IDS, _ISO_LOOKUP)

    # SLR driven forces growth_factor == 1 for every row - this is also what
    # makes compute_avoid_redirected's max(0, growth_factor - 1) term vanish,
    # so avoid should collapse onto protect's numbers in that mode.
    if mode == "SLR driven":
        growth_factors = np.ones(n_growth)
    else:
        growth_factors = 1.0 + GROWTH_RATES

    for s in range(N_SEVERITY):
        # Population driven pins severity to the level-0 (mostly dry) grid.
        ff_col = ff_sequence[0] if mode == "Population driven" else ff_sequence[s]

        if measure in ("baseline", "protect"):
            eai0 = protect_exposure_grid(ff_col, apf_grid, population).sum()
            matrix[:, s] = eai0 * growth_factors
        elif measure == "retreat":
            eai0 = compute_retreat(ff_col, apf_grid, eff_pop).sum()
            matrix[:, s] = eai0 * growth_factors
        elif measure == "avoid":
            for gi, gf in enumerate(growth_factors):
                redirected = compute_avoid_redirected(
                    apf_grid, population, gf, _GEO_IDS, _ISO_LOOKUP
                )
                matrix[gi, s] = compute_avoid(ff_col, apf_grid, population, redirected, gf).sum()

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=None, help="output directory (default: <repo>/validation_output)")
    args = parser.parse_args()

    out_dir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parents[2] / "validation_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    population = generate_population(rng)
    ff_sequence = generate_flood_fraction_sequence(rng)

    print("Wet cell count / max flood_fraction per severity level:")
    for s, ff in enumerate(ff_sequence):
        print(f"  severity {s} (SLR={SLR_MM[s]:.0f}mm): {int((ff > 0).sum())} wet cells, max ff={ff.max():.2f}")

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for mode in MODES:
        for measure in MEASURES:
            matrices[(mode, measure)] = compute_eai_matrix(measure, mode, ff_sequence, population)

    # One global vmax/colour scale across all 12 panels, so absolute EAI is
    # directly comparable across modes and measures, not just within a row.
    # Contour LEVELS are placed independently per panel (contour_vmax below)
    # so a panel whose own range is much smaller than the global vmax still
    # gets a full, evenly-spaced set of contour lines describing its own
    # shape, instead of most levels falling outside its data.
    global_vmax = 1500.0
    print(f"\nGlobal vmax (shared colour scale): {global_vmax:,.0f}")

    mode_vmax = {
        mode: max(float(matrices[(mode, m)].max()) for m in MEASURES) or 1.0
        for mode in MODES
    }
    print("Per-mode max (used for contour spacing only):")
    for mode, vmax in mode_vmax.items():
        print(f"  {mode}: {vmax:,.0f}")

    print("\nSense-check numbers (max-severity, max-growth corner):")
    for mode in MODES:
        row = {m: matrices[(mode, m)][-1, -1] for m in MEASURES}
        print(f"  {mode}: " + ", ".join(f"{m}={v:,.0f}" for m, v in row.items()))

    print("\nInvariant: baseline >= every adaptation measure, everywhere:")
    all_ok = True
    for mode in MODES:
        baseline = matrices[(mode, "baseline")]
        for measure in ("protect", "avoid", "retreat"):
            diff = matrices[(mode, measure)] - baseline
            ok = bool(np.all(diff <= 1e-6))
            all_ok &= ok
            status = "OK" if ok else f"VIOLATED (max excess={float(diff.max()):,.1f})"
            print(f"  {mode} / {measure} <= baseline: {status}")
    print("  ALL PASS" if all_ok else "  SOME INVARIANTS VIOLATED - see above")

    slr_avoid = matrices[("SLR driven", "avoid")]
    slr_protect = matrices[("SLR driven", "protect")]
    max_rel_diff = float(np.max(np.abs(slr_avoid - slr_protect) / np.maximum(slr_protect, 1e-9)))
    print(
        f"\nSLR driven: avoid vs protect max relative difference = {max_rel_diff:.2%} "
        "(expected ~0 - both of avoid's growth-dependent terms vanish when growth_factor==1 everywhere)"
    )

    # One shared diverging colour scale (vmin/vcenter/vmax), used identically
    # on all 12 panels - not just recomputed per panel - so every panel maps
    # a given EAI value to the exact same colour. vmin/vcenter come from
    # "Population driven" (the row this scale was designed for: growth is
    # its only driver, so its own low end and its own growth=0% reference
    # are the most meaningful anchors); vmax is the same global_vmax already
    # shared by the plain colour scale.
    zero_growth_idx = int(np.argmin(np.abs(GROWTH_RATES)))
    pop_driven_matrices = [matrices[("Population driven", m)] for m in MEASURES]
    diverging_vmin = min(float(m.min()) for m in pop_driven_matrices)
    diverging_center = float(matrices[("Population driven", "baseline")][zero_growth_idx, 0])
    print(
        f"\nShared diverging colour scale: vmin={diverging_vmin:,.0f}, "
        f"vcenter={diverging_center:,.0f}, vmax={global_vmax:,.0f}"
    )

    panel_paths: dict[tuple[str, str], Path] = {}
    for mode in MODES:
        for measure in MEASURES:
            safe_mode = mode.replace(" ", "_")
            out_path = out_dir / f"validation_{safe_mode}_{measure}.png"
            matrix = matrices[(mode, measure)]
            fig = plot_burning_ember(
                matrix, SLR_MM, GROWTH_RATES,
                title=f"{mode} — {measure}",
                vmax=global_vmax, dpi=150, figsize=(7, 5.5),
                n_contours=9, fontsize=13,
                diverging_center=diverging_center,
                diverging_vmin=diverging_vmin,
                contour_vmax=float(matrix.max()) or 1.0,
            )
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            panel_paths[(mode, measure)] = out_path
            print(f"  wrote {out_path.name}")

    # Stitch the 12 individual panels into one 4x3 contact sheet for a
    # quick side-by-side look (plot_burning_ember always builds its own
    # standalone figure, so this composes the PNGs after the fact rather
    # than modifying that function to support subplot embedding).
    images = {key: Image.open(p) for key, p in panel_paths.items()}
    cell_w = max(im.width for im in images.values())
    cell_h = max(im.height for im in images.values())
    sheet = Image.new("RGB", (cell_w * len(MEASURES), cell_h * len(MODES)), "white")
    for row, mode in enumerate(MODES):
        for col, measure in enumerate(MEASURES):
            sheet.paste(images[(mode, measure)], (col * cell_w, row * cell_h))
    sheet_path = out_dir / "validation_grid_4x3.png"
    sheet.save(sheet_path)
    print(f"\nContact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
