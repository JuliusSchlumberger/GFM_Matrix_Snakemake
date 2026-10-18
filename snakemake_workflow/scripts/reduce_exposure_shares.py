"""Reduce pass-1 exposure batches into global per-country shares before pass 2.

Sums every batch's (amount, capacity) per (slr_intensity, country), then
share = amount / capacity where capacity > 0. Must finish before any pass-2
batch starts: retreat/avoid tasks need the globally-reduced share.

Usage:
    python reduce_exposure_shares.py \\
        --parts-glob ".../exposure/pass1_batch_*.json" \\
        --out .../exposure/shares_by_intensity.json
"""

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    """Write the globally-reduced per-country shares as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts-glob", required=True, help="glob pattern matching every pass-1 batch's output JSON")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    part_paths = sorted(glob.glob(args.parts_glob))
    if not part_paths:
        raise SystemExit(f"No pass-1 part files matched {args.parts_glob!r}")

    amt_total: dict[str, dict[str, float]] = {}
    cap_total: dict[str, dict[str, float]] = {}
    for p in part_paths:
        with open(p, encoding="utf-8") as f:
            part = json.load(f)
        for slr_int, per_iso in part.items():
            amt_total.setdefault(slr_int, {})
            cap_total.setdefault(slr_int, {})
            for iso, (amt, cap) in per_iso.items():
                amt_total[slr_int][iso] = amt_total[slr_int].get(iso, 0.0) + amt
                cap_total[slr_int][iso] = cap_total[slr_int].get(iso, 0.0) + cap

    shares = {
        slr_int: {
            iso: amt_total[slr_int][iso] / cap_total[slr_int][iso]
            for iso in cap_total[slr_int] if cap_total[slr_int][iso] > 0.0
        }
        for slr_int in amt_total
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shares, f)
    n_shares = sum(len(v) for v in shares.values())
    print(f"Reduced {len(part_paths)} pass-1 parts -> {out_path} ({n_shares} (slr_intensity, iso) shares)")


if __name__ == "__main__":
    main()
