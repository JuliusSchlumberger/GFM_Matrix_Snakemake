"""Fully deterministic, hand-specified small test case for cross-machine
verification against a live, working Julia Eikonal.jl build - see the
drafted prompt for the second machine. Prints Python's full final vertex
`t` array (11x9, high precision) for exact diffing against Julia's.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import solve_eikonal_dense  # noqa: E402

friction = np.array([
    [0.00095135, 0.00058277, 0.00104446, 0.00086710, 0.00020360, 0.00117318, 0.00093725, 0.00096467],
    [0.00024093, 0.00059542, 0.00050788, 0.00111944, 0.00080825, 0.00100504, 0.00058776, 0.00034996],
    [0.00071004, 0.00017020, 0.00101039, 0.00079483, 0.00093390, 0.00048998, 0.00116777, 0.00108243],
    [0.00095622, 0.00031410, 0.00061339, 0.00014818, 0.00026972, 0.00085135, 0.00091924, 0.00116426],
    [0.00045841, 0.00050751, 0.00061651, 0.00030842, 0.00024291, 0.00062328, 0.00034960, 0.00083680],
    [0.00058087, 0.00101595, 0.00087029, 0.00044360, 0.00101549, 0.00098524, 0.00052623, 0.00041716],
    [0.00085075, 0.00025373, 0.00031990, 0.00010810, 0.00096562, 0.00083134, 0.00087568, 0.00095880],
    [0.00060481, 0.00072562, 0.00025378, 0.00022598, 0.00083524, 0.00061821, 0.00072176, 0.00094150],
    [0.00079819, 0.00070894, 0.00071513, 0.00043435, 0.00013390, 0.00058039, 0.00033604, 0.00054938],
    [0.00103874, 0.00035733, 0.00016413, 0.00040952, 0.00042295, 0.00082811, 0.00071274, 0.00096229],
], dtype=np.float32)

# seed cells: (row, col) 0-indexed, row 0 = first/top row, col 0 = first/left col
seed_rows = np.array([0, 0, 9])
seed_cols = np.array([0, 7, 0])
seed_values = np.array([-3.500000, -3.520000, -3.480000], dtype=np.float32)

epsilon = float(friction.min()) / (30.0 * 10.0)  # matches core.jl's minimum(friction)/(resolution*10)

t3 = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon, sweep_budget=3)
t_conv = solve_eikonal_dense(friction, seed_rows, seed_cols, seed_values, epsilon)

np.set_printoptions(precision=9, suppress=False, floatmode="unique", linewidth=200)
print("epsilon =", epsilon)
print("\n=== Python, exactly 3 sweeps (o1,o2,o3) ===")
for row in t3:
    print("  [" + ", ".join(f"{v:.9e}" for v in row) + "]")

print("\n=== Python, full convergence ===")
for row in t_conv:
    print("  [" + ", ".join(f"{v:.9e}" for v in row) + "]")

print(f"\nmax abs diff (3-sweep vs full-conv): {np.abs(t3 - t_conv).max():.6e}")
