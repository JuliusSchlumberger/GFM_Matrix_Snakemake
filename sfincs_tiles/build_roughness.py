"""Decode a tile's friction.tif into a real Manning's n GeoTIFF for SFINCS.

friction.tif stores Manning's_n / 100 (an eikonal-solver-specific "slowness"
convention, decode /1_000_000 per src/rasters.py::decode_friction_int16) -
NOT a real Manning's n. SFINCS's roughness.create() wants real Manning's n
(~0.01-0.15 range), so this multiplies the decoded value back up by 100.
See sfincs_tiles' own plan doc for the full reasoning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retry_io import retry_transient_io  # noqa: E402

FRICTION_SCALE = 1_000_000

# Generous physical envelope for real Manning's n (open water ~0.02 to dense
# forest/urban ~0.15-0.2) - NOT the "~0.01-0.15 for real land cover" figure
# printed below, which is a tighter expectation for real data and not
# something a bare land-cover lookup table is guaranteed to respect exactly.
# This wider bound exists to catch a unit-convention regression, not unusual
# land cover: forgetting the *100 fix above produces values ~100x too small
# (order 1e-4, comfortably below MANNING_N_MIN), and applying it twice
# produces values ~100x too large (order 1-15, comfortably above
# MANNING_N_MAX) - both real, plausible mistakes given friction.tif's own
# non-obvious "Manning's_n / 100" on-disk convention (see module docstring).
MANNING_N_MIN = 0.001
MANNING_N_MAX = 1.0


def build_manning_n(friction_path: Path, out_path: Path) -> tuple[float, float]:
    with retry_transient_io(rasterio.open, friction_path) as src:
        friction_int16 = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

    valid = friction_int16 != nodata if nodata is not None else np.ones_like(friction_int16, dtype=bool)
    slowness = friction_int16.astype(np.float32) / np.float32(FRICTION_SCALE)  # = Manning's_n / 100
    manning_n = np.where(valid, slowness * 100.0, np.nan)

    finite_check = manning_n[np.isfinite(manning_n)]
    if finite_check.size and (finite_check.min() < MANNING_N_MIN or finite_check.max() > MANNING_N_MAX):
        raise ValueError(
            f"Manning's n range [{finite_check.min():.6f}, {finite_check.max():.6f}] falls outside the "
            f"physically-plausible envelope [{MANNING_N_MIN}, {MANNING_N_MAX}] - almost certainly a unit-"
            f"conversion bug (the *100 fix above missing or applied twice), not real land cover."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(manning_n.astype(np.float32), 1)

    finite = manning_n[np.isfinite(manning_n)]
    return float(finite.min()), float(finite.max())


if __name__ == "__main__":
    import argparse
    import sys

    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--base-dir-name", default="validation_sfincs", help="output root directory name under paths.root (default: validation_sfincs)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gfm_config import read_root

    root = read_root(Path(args.config))
    tile_dir = root / "model_outputs" / args.tile_id / "inputs"
    out_path = root / args.base_dir_name / args.tile_id / "sfincs_model" / "manning_n.tif"

    lo, hi = build_manning_n(tile_dir / "friction.tif", out_path)
    print(f"Manning's n range: {lo:.5f} to {hi:.5f} (expected ~0.01-0.15 for real land cover)")
    print(f"Wrote {out_path}")
