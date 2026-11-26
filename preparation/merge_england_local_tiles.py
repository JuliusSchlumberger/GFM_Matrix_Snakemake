"""Merge a local-input scenario's tiles into native-CRS chunks, plus one VRT over them per event.

The tiles' native grid is the local DEM's own projected CRS (e.g. EPSG:27700 for England and
Wales), not the global EPSG:4326 chunk grid merge_chunk assumes; they are never reprojected.
Chunks are squares of postprocessing.native_chunk_m aligned to multiples of it
(merge.native_chunk_plan), packed by tile count into --n-batches jobs; --vrt then writes
local_native/waterdepth_<RP>_<SLR>.vrt, which the *_local validation scripts read. Every
tile must have its output: a failed triple raises here, rerun it first. Not Snakemake-tracked;
generate_hpc_postprocess_job.py runs it for a postprocessing.merge_only scenario with a
dem_source_manifest.

Usage:
    python merge_england_local_tiles.py --config <materialized.yml> --return-periods RP100 \\
        --waterlevel-name SLR_0 [--batch K --n-batches N | --vrt]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config_utils import load_config, retry_transient_io  # noqa: E402
from hpc_common import lpt_pack  # noqa: E402
from merge import merge_native_chunk, native_chunk_plan, native_tile_bounds  # noqa: E402


def chunk_path(native_dir: Path, kind: str, event: str, chunk_id: str) -> Path:
    return native_dir / "chunks" / f"{kind}_{event}_{chunk_id}.tif"


def vrt_path(native_dir: Path, event: str) -> Path:
    return native_dir / f"waterdepth_{event}.vrt"


def _tile_rasters(model_outputs: Path, event: str) -> list[Path]:
    tile_ids = sorted(int(p.name) for p in model_outputs.iterdir() if p.name.isdigit())
    rasters = [model_outputs / str(tid) / "results" / f"waterdepth_{event}.tif" for tid in tile_ids]
    missing = [p for p in rasters if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{event}: {len(missing)} missing waterdepth output(s), e.g. {missing[:5]}")
    return rasters


def batch_chunks(plan: dict, batch: int, n_batches: int) -> list[str]:
    """Return the chunk ids of batch `batch` of `n_batches`, packed by tile count."""
    if not 0 <= batch < n_batches:
        raise ValueError(f"--batch {batch} outside 0..{n_batches - 1}")
    return sorted(lpt_pack({cid: len(tiles) for cid, (_, tiles) in plan.items()}, n_batches)[batch])


def build_vrt(chunk_files: list[Path], out_path: Path) -> None:
    from osgeo import gdal

    gdal.UseExceptions()
    tmp = out_path.with_name(out_path.name + ".tmp.vrt")
    vrt = gdal.BuildVRT(str(tmp), [str(p) for p in chunk_files])
    vrt.FlushCache()
    del vrt
    # GDAL writes the chunk paths relative to the VRT; same directory, so the rename keeps them valid.
    tmp.replace(out_path)


def run(config: dict, return_periods: list[str], waterlevel_name: str, batch: int = 0, n_batches: int = 1,
        vrt: bool = False) -> list[Path]:
    model_outputs = Path(config["simulation"]["model_outputs"])
    post = config["postprocessing"]
    native_dir = Path(post["merged_outputs"]) / "local_native"
    retry_transient_io((native_dir / "chunks").mkdir, parents=True, exist_ok=True)

    written = []
    for rp in return_periods:
        event = f"{rp}_{waterlevel_name}"
        plan = native_chunk_plan(native_tile_bounds(_tile_rasters(model_outputs, event)),
                                 float(post["native_chunk_m"]))
        if vrt:
            chunks = [chunk_path(native_dir, "waterdepth", event, cid) for cid in plan]
            missing = [p for p in chunks if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{event}: {len(missing)} of {len(chunks)} chunk(s) missing, e.g. {missing[:5]}")
            build_vrt(chunks, vrt_path(native_dir, event))
            print(f"{event}: VRT over {len(chunks)} chunk(s) -> {vrt_path(native_dir, event)}")
            written.append(vrt_path(native_dir, event))
            continue
        mine = batch_chunks(plan, batch, n_batches)
        print(f"{event}: {len(plan)} chunk(s) of {post['native_chunk_m']} m, batch {batch}/{n_batches}: {len(mine)}",
              flush=True)
        for cid in mine:
            bounds, tiles = plan[cid]
            wd = chunk_path(native_dir, "waterdepth", event, cid)
            merge_native_chunk(tiles, bounds, wd, chunk_path(native_dir, "provenance", event, cid),
                               post["block_size"], config["raster_format"])
            print(f"  {cid}: {len(tiles)} tile(s) -> {wd.name}", flush=True)
            written.append(wd)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--return-periods", nargs="+", default=["RP100", "RP250"])
    parser.add_argument("--waterlevel-name", default="SLR_0")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--n-batches", type=int, default=1)
    parser.add_argument("--vrt", action="store_true", help="write the VRT over the merged chunks instead")
    args = parser.parse_args()
    run(load_config(args.config), args.return_periods, args.waterlevel_name, args.batch, args.n_batches, args.vrt)


if __name__ == "__main__":
    main()
