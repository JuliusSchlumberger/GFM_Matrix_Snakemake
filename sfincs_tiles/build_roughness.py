"""Decode a tile's friction.tif into a real Manning's n GeoTIFF for SFINCS.

friction.tif stores Manning's_n / 100 (an eikonal-solver-specific "slowness"
convention, decode /1_000_000 per src/rasters.py::decode_friction_int16) -
NOT a real Manning's n. SFINCS's roughness.create() wants real Manning's n
(~0.01-0.15 range), so this multiplies the decoded value back up by 100.
See sfincs_tiles' own plan doc for the full reasoning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

FRICTION_SCALE = 1_000_000


def build_manning_n(friction_path: Path, out_path: Path) -> tuple[float, float]:
    with rasterio.open(friction_path) as src:
        friction_int16 = src.read(1)
        nodata = src.nodata
        profile = src.profile.copy()

    valid = friction_int16 != nodata if nodata is not None else np.ones_like(friction_int16, dtype=bool)
    slowness = friction_int16.astype(np.float32) / np.float32(FRICTION_SCALE)  # = Manning's_n / 100
    manning_n = np.where(valid, slowness * 100.0, np.nan)

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
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gfm_config import read_root

    root = read_root(Path(args.config))
    tile_dir = root / "model_outputs" / args.tile_id / "inputs"
    out_path = root / "validation_sfincs" / args.tile_id / "sfincs_model" / "manning_n.tif"

    lo, hi = build_manning_n(tile_dir / "friction.tif", out_path)
    print(f"Manning's n range: {lo:.5f} to {hi:.5f} (expected ~0.01-0.15 for real land cover)")
    print(f"Wrote {out_path}")
