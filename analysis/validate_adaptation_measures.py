"""Sense-check the production adaptation exposure formulas on a synthetic 100x100 grid.

Standalone diagnostic (not in the DAG). EAI is the formula output at one severity
level per SLR step (no RP integration); the whole grid is one "country".

Usage:
    python analysis/validate_adaptation_measures.py
        [--out-dir D:/GFM/figures/validation]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import retry_transient_io  # noqa: E402
from exposure_analysis import (  # noqa: E402
    compute_avoid,
    compute_avoid_redirected,
    compute_retreat,
    compute_retreat_capacity,
    protect_exposure_grid,
)
from analysis_figures import plot_burning_ember  # noqa: E402

GRID_SIZE = 100
SEED = 42
N_SEVERITY = 7
SLR_MM = np.linspace(0, 1500, N_SEVERITY)
GROWTH_RATES = np.arange(-0.5, 1.5 + 1e-9, 0.1)

# Flood-fraction cutoff inside the design zone only (cells wet at DESIGN_LEVEL), 0 elsewhere,
# like build_adapt_protection_fraction. Must exceed INITIAL_WET_VALUE and lie inside
# SEED_WET_RANGE; a uniform (not zone-limited) floodplain would make avoid/retreat move people
# into cells no safer than the ones they left.
DESIGN_STANDARD = 0.50
DESIGN_LEVEL = 4  # severity level (SLR_MM[4] = 1 m) the design zone is taken from

# Contagion: each step wet cells grow x WET_GROWTH_FACTOR (cap 1.0) and 4-connected dry neighbours
# activate with ACTIVATION_PROB, giving roughly exponential growth of flooded extent.
# The seed gradient straddles DESIGN_STANDARD so "Population driven" (level 0) has both
# unstoppable and protected flooding. New cells start below standard, so measures diverge from
# baseline at low-to-mid severity and converge again as cells grow past it.
SEED_WET_RANGE = (0.80, 0.20)  # seed patch flood fraction, first to last column
INITIAL_WET_VALUE = 0.12
ACTIVATION_PROB = 0.30     # per-step chance a dry neighbour of a wet cell activates
WET_GROWTH_FACTOR = 1.3    # per-step multiplicative growth of already-wet cells

MEASURES = ["baseline", "protect", "avoid", "retreat", "protect_avoid"]
MODES = ["SLR driven", "Population driven", "Mixed drivers"]

FIGURE_ROWS = [("Population driven", "Population driven"), ("SLR driven", "SLR driven"),
               ("Mixed drivers", "Combination driven")]
FIGURE_COLUMNS = {"baseline": "No adaptation", "protect": "Protect", "avoid": "Avoid", "retreat": "Retreat",
                  "protect_avoid": "Protect + Avoid"}

# One "country": retreat/avoid redistribution pools across the whole grid.
_GEO_IDS = np.zeros((GRID_SIZE, GRID_SIZE), dtype="int32")
_ISO_LOOKUP = {0: "SYN"}


def generate_population(rng: np.random.Generator) -> np.ndarray:
    """Generate the fixed lognormal population grid shared by every panel."""
    return rng.lognormal(mean=3.0, sigma=1.0, size=(GRID_SIZE, GRID_SIZE))


def generate_flood_fraction_sequence(rng: np.random.Generator) -> list[np.ndarray]:
    """Generate one flood-fraction grid per severity level by seeded contagion."""
    ff = np.zeros((GRID_SIZE, GRID_SIZE), dtype="float64")
    ff[GRID_SIZE // 2 - 3 : GRID_SIZE // 2 + 3, 0:5] = np.linspace(*SEED_WET_RANGE, 5)

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


def design_grid(ff_sequence: list[np.ndarray]) -> np.ndarray:
    """Return DESIGN_STANDARD where wet at DESIGN_LEVEL, else 0."""
    return np.where(ff_sequence[DESIGN_LEVEL] > 0, DESIGN_STANDARD, 0.0)


def compute_eai_matrix(
    measure: str,
    mode: str,
    ff_sequence: list[np.ndarray],
    population: np.ndarray,
) -> np.ndarray:
    """Compute the (n_growth, n_severity) EAI matrix for one (measure, mode) panel.

    baseline/protect/retreat are linear in growth, so they are scaled from
    growth 1; avoid/protect + avoid are non-linear and recomputed per cell.
    SLR driven: growth 1 everywhere, so avoid == baseline, protect + avoid == protect.
    Population driven: severity pinned to level 0.
    """
    n_growth = len(GROWTH_RATES)
    matrix = np.zeros((n_growth, N_SEVERITY))

    apf_grid = (
        np.zeros((GRID_SIZE, GRID_SIZE))
        if measure == "baseline"
        else design_grid(ff_sequence)
    )

    if measure == "retreat":
        eff_pop = compute_retreat_capacity(apf_grid, population, _GEO_IDS, _ISO_LOOKUP)

    if mode == "SLR driven":
        growth_factors = np.ones(n_growth)
    else:
        growth_factors = 1.0 + GROWTH_RATES

    for s in range(N_SEVERITY):
        ff_col = ff_sequence[0] if mode == "Population driven" else ff_sequence[s]

        if measure in ("baseline", "protect"):
            eai0 = protect_exposure_grid(ff_col, apf_grid, population).sum()
            matrix[:, s] = eai0 * growth_factors
        elif measure == "retreat":
            eai0 = compute_retreat(ff_col, apf_grid, eff_pop).sum()
            matrix[:, s] = eai0 * growth_factors
        elif measure in ("avoid", "protect_avoid"):
            # avoid: existing population keeps baseline protection (none here)
            existing = np.zeros((GRID_SIZE, GRID_SIZE)) if measure == "avoid" else None
            for gi, gf in enumerate(growth_factors):
                redirected = compute_avoid_redirected(
                    apf_grid, population, gf, _GEO_IDS, _ISO_LOOKUP
                )
                matrix[gi, s] = compute_avoid(ff_col, apf_grid, population, redirected, gf,
                                              existing_prot_frac=existing).sum()

    return matrix


def plot_conceptual_grid(
    matrices: dict[tuple[str, str], np.ndarray], out_path: Path,
    vmin: float, vcenter: float, vmax: float, n_contours: int = 9,
) -> None:
    """Plot all 15 panels as one unnumbered manuscript figure (.png and .pdf)."""
    import matplotlib.colors as mcolors

    slr_x = SLR_MM / 1000.0
    gr_y = GROWTH_RATES * 100.0
    extent = [slr_x[0], slr_x[-1], gr_y[0], gr_y[-1]]
    norm = mcolors.TwoSlopeNorm(vmin=min(vmin, vcenter - 1e-9), vcenter=vcenter, vmax=max(vmax, vcenter + 1e-9))

    fig, axes = plt.subplots(len(FIGURE_ROWS), len(FIGURE_COLUMNS), figsize=(21, 11.5),
                             sharex=True, sharey=True, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    im = None
    for r, (mode, row_label) in enumerate(FIGURE_ROWS):
        for c, (measure, col_label) in enumerate(FIGURE_COLUMNS.items()):
            ax = axes[r, c]
            matrix = matrices[(mode, measure)]
            im = ax.imshow(matrix, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", norm=norm,
                           interpolation="bilinear")
            hi = float(matrix.max()) or 1.0
            levels = np.linspace(vcenter, hi, n_contours + 1)[1:]
            if hi > vcenter:
                ax.contour(slr_x, gr_y, matrix, levels=levels, colors="k", linewidths=1.0, alpha=0.7)
            ax.axhline(0.0, color="0.25", linestyle="--", linewidth=1.0)
            ax.set_xticks([])
            ax.set_yticks([0.0])
            ax.set_yticklabels(["0"])
            ax.tick_params(axis="y", labelsize=16, length=5)
            if r == 0:
                ax.set_title(col_label, fontsize=20, fontweight="bold", pad=12)
            if r == len(FIGURE_ROWS) - 1:
                ax.set_xlabel("Sea level rise →", fontsize=17)
            if c == 0:
                ax.set_ylabel("← Population change →", fontsize=17)
                ax.annotate(row_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-62, 0),
                            textcoords="offset points", rotation=90, ha="right", va="center",
                            fontsize=20, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_ticks([])
    cbar.set_label("Expected annual impact (people)", fontsize=17)
    cbar.ax.text(0.5, -0.015, "lower", transform=cbar.ax.transAxes, ha="center", va="top", fontsize=15)
    cbar.ax.text(0.5, 1.015, "higher", transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=15)

    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nManuscript figure: {out_path.with_suffix('.png')} (+ .pdf)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", "--outdir", dest="outdir", default=None, help="output directory (default: <repo>/adaptation_validation_output)")
    args = parser.parse_args()

    out_dir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parents[1] / "adaptation_validation_output"
    retry_transient_io(out_dir.mkdir, parents=True, exist_ok=True)

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

    # Shared colour scale across panels; contour levels per panel (contour_vmax).
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
        for measure in ("protect", "avoid", "retreat", "protect_avoid"):
            diff = matrices[(mode, measure)] - baseline
            ok = bool(np.all(diff <= 1e-6))
            all_ok &= ok
            status = "OK" if ok else f"VIOLATED (max excess={float(diff.max()):,.1f})"
            print(f"  {mode} / {measure} <= baseline: {status}")
    print("  ALL PASS" if all_ok else "  SOME INVARIANTS VIOLATED - see above")

    print("\nInvariant: at growth <= 0, avoid == baseline and protect + avoid == protect:")
    no_growth = GROWTH_RATES <= 1e-9
    for mode in MODES:
        rows = np.ones(len(GROWTH_RATES), dtype=bool) if mode == "SLR driven" else no_growth
        for measure, ref in (("avoid", "baseline"), ("protect_avoid", "protect")):
            a, b = matrices[(mode, measure)][rows], matrices[(mode, ref)][rows]
            rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-9)))
            ok = rel < 1e-9
            all_ok &= ok
            print(f"  {mode} / {measure} == {ref}: {'OK' if ok else f'VIOLATED (max rel diff {rel:.2%})'}")
    print("  ALL PASS" if all_ok else "  SOME INVARIANTS VIOLATED - see above")

    # vmin/vcenter anchored on "Population driven" (growth is its only driver).
    zero_growth_idx = int(np.argmin(np.abs(GROWTH_RATES)))
    pop_driven_matrices = [matrices[("Population driven", m)] for m in MEASURES]
    diverging_vmin = min(float(m.min()) for m in pop_driven_matrices)
    diverging_center = float(matrices[("Population driven", "baseline")][zero_growth_idx, 0])
    print(
        f"\nShared diverging colour scale: vmin={diverging_vmin:,.0f}, "
        f"vcenter={diverging_center:,.0f}, vmax={global_vmax:,.0f}"
    )

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
                # levels start at diverging_center; panels wholly below it get none
                contour_vmax=max(float(matrix.max()), diverging_center * 1.05),
            )
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  wrote {out_path.name}")

    plot_conceptual_grid(matrices, out_dir / "validation_grid_5x3.png",
                         vmin=diverging_vmin, vcenter=diverging_center, vmax=global_vmax)


if __name__ == "__main__":
    main()
