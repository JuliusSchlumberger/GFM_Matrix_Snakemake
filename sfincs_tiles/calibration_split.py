"""Split the SFINCS tiles into a calibration and a validation group, and check the balance.

A tile's group follows from a hash of its id alone, so it never changes when tiles are
added. Writes {base}/calibration_split.csv and calibration_split_balance.csv.

Usage:
    python calibration_split.py --config <config.yml> [--base-dir-name calibration_roughness_SFINCS]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gfm_config import read_root

CALIBRATION_SHARE = 0.3
GROUPS = ("calibration", "validation")
LAT_BANDS = (0.0, 23.5, 45.0, 90.0)
BALANCE_COLUMNS = ("n_cells", "ocean_frac", "river_frac", "sfincs_flooded_km2")


def split_group(tile_id: int | str, share: float = CALIBRATION_SHARE) -> str:
    """Return "calibration" for a `share` of tile ids (by SHA-256 of the id), else "validation"."""
    h = int.from_bytes(hashlib.sha256(str(int(tile_id)).encode()).digest()[:8], "big") / 2.0**64
    return "calibration" if h < share else "validation"


def tile_groups(tile_ids, share: float = CALIBRATION_SHARE) -> pd.Series:
    """Return the group per tile id."""
    ids = [int(t) for t in tile_ids]
    return pd.Series([split_group(t, share) for t in ids], index=pd.Index(ids, name="tile_id"), name="group")


def balance_table(tiles: pd.DataFrame) -> pd.DataFrame:
    """Compare the groups: tile share per latitude band and hemisphere, quartiles of BALANCE_COLUMNS."""
    rows = []
    band = pd.cut(tiles.lat.abs(), LAT_BANDS, include_lowest=True).astype(str)
    hemi = np.where(tiles.lat >= 0, "N", "S")
    for name, key in (("latitude band", band), ("hemisphere", pd.Series(hemi, index=tiles.index))):
        share = pd.crosstab(key, tiles.group, normalize="columns")
        for level, r in share.iterrows():
            rows.append({"what": f"{name} {level}", **{g: round(float(r.get(g, 0.0)), 3) for g in GROUPS}})
    for col in BALANCE_COLUMNS:
        if col not in tiles or tiles[col].isna().all():
            continue
        q = tiles.groupby("group")[col].quantile([0.25, 0.5, 0.75]).unstack()
        for p in q.columns:
            rows.append({"what": f"{col} p{int(p * 100)}", **{g: float(q.loc[g, p]) if g in q.index else None
                                                               for g in GROUPS}})
    rows.append({"what": "tiles", **{g: int((tiles.group == g).sum()) for g in GROUPS}})
    return pd.DataFrame(rows)


def _sfincs_flooded_km2(base_dir: Path, tile_ids) -> dict:
    out = {}
    for t in tile_ids:
        p = base_dir / str(t) / "outputs" / "calibration_result.json"
        if p.exists():
            out[int(t)] = json.loads(p.read_text())["sfincs"]["flooded_km2"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir-name", default="calibration_roughness_SFINCS")
    args = parser.parse_args()
    base_dir = read_root(Path(args.config)) / args.base_dir_name
    meta = pd.read_csv(base_dir / "tile_selection_metadata.csv")
    tiles = meta[meta.tile_id.astype(str).map(lambda t: (base_dir / t / "sfincs_model").is_dir())].copy()
    tiles["group"] = tile_groups(tiles.tile_id).to_numpy()
    tiles["sfincs_flooded_km2"] = tiles.tile_id.map(_sfincs_flooded_km2(base_dir, tiles.tile_id))
    tiles[["tile_id", "group"]].to_csv(base_dir / "calibration_split.csv", index=False)
    balance = balance_table(tiles)
    balance.to_csv(base_dir / "calibration_split_balance.csv", index=False)
    print(balance.to_string(index=False))


if __name__ == "__main__":
    main()
