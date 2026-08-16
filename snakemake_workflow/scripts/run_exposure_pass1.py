"""HPC batch worker: pass 1 of the chunk-partitioned exposure pipeline.

Streams this batch's chunk subset ONCE, accumulating retreat/avoid's raw
per-country (amount, capacity) sums for every slr_intensity. See
analysis/compute_exposure_analysis.py's module docstring and
pass1_shares_all_intensities for the full design - this must complete (and
be reduced by reduce_exposure_shares.py) before any pass-2 batch starts.

Usage:
    python run_exposure_pass1.py --config path/to/config.yml \\
        --chunks-file batch_000_chunks.txt --out pass1_batch_000.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402
from compute_exposure_analysis import load_analysis_context, pass1_shares_all_intensities  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--chunks-file", required=True, help="one chunk_id per line")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ctx = load_analysis_context(cfg)
    chunk_ids = [
        line.strip() for line in Path(args.chunks_file).read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    print(f"Pass 1: {len(chunk_ids)} chunks x {len(ctx.slr_intensities)} slr_intensities…")
    raw = pass1_shares_all_intensities(
        chunk_ids, ctx.chunks_dir, ctx.flood_frac_dir, ctx.return_periods, ctx.slr_scenarios,
        ctx.slr_intensities, ctx.rp_applied, ctx.iso_lookup,
    )

    out_path = Path(args.out)
    retry_transient_io(out_path.parent.mkdir, parents=True, exist_ok=True)
    # Atomic (temp file + os.replace) - a --resume dispatch trusts this
    # file's mere existence as "this batch is done"; a truncated file from
    # a killed job must never appear at this exact path (see
    # config_utils.atomic_write's own docstring).
    atomic_write(out_path, lambda f: json.dump(raw, f), mode="w", encoding="utf-8")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
