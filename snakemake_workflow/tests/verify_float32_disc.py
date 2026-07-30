"""Does replicating Julia's EXACT literal formula (a=2 as Int-like, b, c,
disc=b^2-4ac) in native float32 - not the algebraically-simplified
disc2=2v^2-(t_a-t_b)^2 form tried earlier - reproduce Julia's actual traced
output for the cross-machine test cell?

Julia's live-instrumented trace (second machine, Eikonal v0.1.1) for
t_a=-3.513029575347900f0, t_b=-3.512866258621216f0, v=9.622899815440178f-4:
    b = 1.40517921447753906e+01  (Float32)
    c = 2.46816062927246094e+01  (Float32)
    Δ = 1.52587890625000000e-05  (Float32)   [[exactly 2^-16]]
    cand = -3.51197147369384766e+00
    final result = -3.51206731796264648e+00
"""
import numpy as np

t_a = np.float32(-3.513029575347900)
t_b = np.float32(-3.512866258621216)
v = np.float32(9.622899815440178e-04)

neg_two = np.float32(-2.0)
eight = np.float32(8.0)
four = np.float32(4.0)

b = neg_two * (t_a + t_b)
c = t_a * t_a + t_b * t_b - v * v
disc = b * b - eight * c

print(f"b    = {b!r}  ({b.dtype})")
print(f"c    = {c!r}  ({c.dtype})")
print(f"disc = {disc!r}  ({disc.dtype})")
print(f"disc as exponent of 2: log2 = {np.log2(float(disc)):.6f}  (Julia's was exactly -16)")

if disc >= 0:
    cand = (-b + np.sqrt(disc)) / four
    print(f"cand (before causality check) = {cand!r}")
    accepted = cand > max(t_a, t_b)
    print(f"cand > max(t_a,t_b)? {accepted}")
    fallback = min(t_a + v, t_b + v)
    result = min(cand, fallback) if accepted else fallback
else:
    result = min(t_a + v, t_b + v)

print(f"final result = {result!r}")
print()
print("Julia's actual reported values: b=14.05179214477539, c=24.68160629272461, "
      "disc=1.52587890625e-05, cand=-3.51197147369384766, final=-3.51206731796264648")
