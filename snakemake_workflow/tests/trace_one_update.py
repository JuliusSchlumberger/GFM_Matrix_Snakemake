"""Trace the exact update() computation for vertex (10,8) in the
cross-machine test case - the cleanest discrepant cell, since it's touched
by only ONE orthant (orthant 1; orthant 2's row range excludes the last
row), and its two inputs (t[9,8], t[10,7]) are CONFIRMED bit-identical
between the Python and Julia runs. Any difference in the output must
originate inside this single update() evaluation, not from upstream input
differences.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eikonal import _update  # noqa: E402

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

t_a = np.float32(-3.513029575e+00)  # t[9,8]
t_b = np.float32(-3.512866259e+00)  # t[10,7]
v = friction[9, 7]  # 0.00096229

print(f"t_a (t[9,8])  = {t_a!r} = {t_a:.15e}")
print(f"t_b (t[10,7]) = {t_b!r} = {t_b:.15e}")
print(f"v (friction[9,7]) = {v!r} = {v:.15e}")

# replicate _update's exact arithmetic, printing every intermediate
a = 2.0
b = -2.0 * (t_a + t_b)
c = t_a * t_a + t_b * t_b - v * v
disc = b * b - 4.0 * a * c
print(f"\na = {a!r}  (type {type(a)})")
print(f"b = {b!r}  (type {type(b)})")
print(f"c = {c!r}  (type {type(c)})")
print(f"disc = {disc!r}  (type {type(disc)})")

cand = None
if disc >= 0.0:
    cand = (-b + np.sqrt(disc)) / (2.0 * a)
    print(f"cand (quadratic root) = {cand!r}")
    print(f"max(t_a,t_b) = {max(t_a,t_b)!r}")
    print(f"cand > max(t_a,t_b)? {cand > max(t_a, t_b)}")

fallback_a = t_a + v
fallback_b = t_b + v
print(f"\nt_a + v = {fallback_a!r}")
print(f"t_b + v = {fallback_b!r}")
print(f"fallback = min(t_a+v, t_b+v) = {min(fallback_a, fallback_b)!r}")

result = _update(t_a, t_b, v)
print(f"\n_update(t_a, t_b, v) = {result!r} = {result:.9e}")
print(f"(type of result: {type(result)})")

# also compute in pure float64 (no float32 anywhere) for reference
t_a64 = float(t_a)
t_b64 = float(t_b)
v64 = float(v)
b64 = -2.0 * (t_a64 + t_b64)
c64 = t_a64*t_a64 + t_b64*t_b64 - v64*v64
disc64 = b64*b64 - 8.0*c64
cand64 = (-b64 + np.sqrt(disc64)) / 4.0 if disc64 >= 0 else None
fallback64 = min(t_a64+v64, t_b64+v64)
result64 = min(cand64, fallback64) if cand64 is not None and cand64 > max(t_a64,t_b64) else fallback64
print(f"\npure-float64 result = {result64:.15e}")
