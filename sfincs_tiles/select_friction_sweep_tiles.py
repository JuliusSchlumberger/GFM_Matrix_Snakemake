"""Selects the tile set for the friction_scale_factor sensitivity sweep
(see docs/methods_04a_SFINCSvalidation.md's friction-sweep section): every
validation_sfincs_v5 tile with fresh eikonal data, EXCLUDING tiles where
eikonal (at the current production friction_scale_factor=30.0) already
over-predicts flood extent by more than `--max-eikonal-only-km2` relative
to SFINCS (eikonal_only_km2, i.e. false-alarm area).

Rationale (2026-10-02): a large eikonal-only area usually means SFINCS's
own domain boundary sits too far offshore to let water reach the area at
all for this tile, not that eikonal's friction is too low - fitting a
global friction scale against those tiles would push friction UP just to
compensate for a SFINCS boundary-placement artifact, contaminating the fit
for every other tile. Excluding them keeps the sweep's objective meaningful
(SFINCS maps that already look reasonable as a flood-extent reference).

Usage:
    python select_friction_sweep_tiles.py --base-dir-name validation_sfincs_v5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_validation_results import DATA_ROOT, collect_summaries  # noqa: E402


def select_tiles(base_dir: Path, max_eikonal_only_km2: float) -> tuple[list[int], list[int]]:
    """Returns (included_tile_ids, excluded_tile_ids), both sorted."""
    df = collect_summaries(base_dir, stale_cutoff=None)
    fresh = ~df["eikonal_stale"].fillna(False)
    has_eikonal = df["eikonal_matched_km2"].notna()
    df = df[fresh & has_eikonal]

    included = df[df["eikonal_only_km2"] <= max_eikonal_only_km2]
    excluded = df[df["eikonal_only_km2"] > max_eikonal_only_km2]
    return (
        sorted(included["tile_id"].astype(int).tolist()),
        sorted(excluded["tile_id"].astype(int).tolist()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir-name", default="validation_sfincs_v5")
    parser.add_argument("--max-eikonal-only-km2", type=float, default=10.0,
                         help="exclude tiles where baseline eikonal_only_km2 exceeds this (default: 10.0)")
    parser.add_argument("--out", default=None,
                         help="output text file, one tile_id per line (default: "
                              "{base_dir}/friction_sweep_tile_ids.txt)")
    args = parser.parse_args()

    base_dir = DATA_ROOT / args.base_dir_name
    included, excluded = select_tiles(base_dir, args.max_eikonal_only_km2)

    print(f"{len(included) + len(excluded)} tile(s) with fresh eikonal data under {base_dir}")
    print(f"  included (eikonal_only_km2 <= {args.max_eikonal_only_km2}): {len(included)}")
    print(f"  excluded (eikonal_only_km2 >  {args.max_eikonal_only_km2}): {len(excluded)}")
    print(f"  excluded tile_ids: {excluded}")

    out_path = Path(args.out) if args.out else base_dir / "friction_sweep_tile_ids.txt"
    out_path.write_text("\n".join(str(t) for t in included), encoding="utf-8")
    print(f"wrote {out_path} ({len(included)} tile_id(s))")


if __name__ == "__main__":
    main()
