"""Write the HPC jobs of the chunk-partitioned exposure analysis; entry point `generate()`.

pass1 batches -> reduce_shares -> pass2 batches -> reduce_write, each afterany. `add_exposure_jobs()`
is also called by generate_hpc_postprocess_job.py (tile-grid chunk ids, appended to its chain);
standalone, the chunk ids come from the flood_fraction files on the local mount. pass1/pass2 skip
unpopulated chunks inline. Embedded paths use config_hpc.yml.

Usage:
    python generate_hpc_exposure_jobs.py [--config path/to/config.yml] [--resume] [--profile-set P.F=V]
    bash <printed submit script>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import load_config, load_hpc_config, retry_transient_io  # noqa: E402
from hpc_common import JobChain, add_profile_set_argument, apply_profile_sets, even_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from _common import discover_chunk_ids  # noqa: E402


def _paths(local_config: dict, linux_config: dict) -> dict:
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    exposure_dir = f"{linux_jobs_dir}/exposure"
    return {
        "code_root": linux_config["paths"]["code_root"],
        "local_dir": Path(local_config["hpc"]["jobs_dir"]) / "exposure",
        "linux_dir": exposure_dir,
        "scripts": f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts",
        # resolved_config.yml so a scenario dispatch's own config reaches these jobs.
        "config": f"{linux_jobs_dir}/resolved_config.yml",
        "shares": f"{exposure_dir}/shares_by_intensity.json",
        "outdir": f"{linux_config['postprocessing']['merged_outputs']}/exposure",
    }


def _add(chain: JobChain, p: dict, name: str, body: list[str], after: list[str], stage: bool) -> str:
    lines = [f'cd "{p["code_root"]}"']
    if stage:
        lines.append(f'stage_configfile_locally "{p["config"]}" LOCAL_CONFIGFILE || exit 1')
    chain.add(name, "exposure", lines + body, after=after, local_dir=p["local_dir"], linux_dir=p["linux_dir"],
              helpers=("stage_config",) if stage else ())
    return name


def _add_pass(chain: JobChain, p: dict, pass_no: int, batch_id: str, after: list[str], resume: bool) -> str:
    """Pass 1 writes the batch's share JSON; pass 2 reads the reduced shares and writes its EAI pickle."""
    chunks_file = f"{p['linux_dir']}/batch_{batch_id}_chunks.txt"
    if pass_no == 1:
        command = (f'"$GFM_PY" "{p["scripts"]}/run_exposure_pass1.py" --config "$LOCAL_CONFIGFILE" '
                   f'--chunks-file "{chunks_file}" --out "{p["linux_dir"]}/pass1_batch_{batch_id}.json"')
    else:
        command = (f'"$GFM_PY" "{p["scripts"]}/run_exposure_pass2.py" --config "$LOCAL_CONFIGFILE" '
                   f'--chunks-file "{chunks_file}" --shares "{p["shares"]}" '
                   f'--out "{p["linux_dir"]}/pass2_batch_{batch_id}.pkl"')
    banner = f"Exposure pass {pass_no}{' (resume)' if resume else ''}, batch {batch_id}"
    return _add(chain, p, f"exposure_pass{pass_no}_batch_{batch_id}", [f'echo "=== {banner} ==="', command],
                after, stage=True)


def add_exposure_jobs(chain: JobChain, local_config: dict, linux_config: dict, chunk_ids: list[str],
                      after: list[str]) -> list[str]:
    """Add pass1 -> reduce_shares -> pass2 -> reduce_write for `chunk_ids`; return the last job's name.

    `chunk_ids` are split evenly in order over hpc.n_nodes; batch_<id>_chunks.txt keeps the split
    for a resume. Raises SystemExit if `chunk_ids` is empty.
    """
    if not chunk_ids:
        raise SystemExit("exposure analysis: no chunk ids - nothing to plan.")
    p = _paths(local_config, linux_config)
    retry_transient_io((p["local_dir"] / "logs").mkdir, parents=True, exist_ok=True)
    batches = even_split(chunk_ids, linux_config["hpc"]["n_nodes"])
    batch_ids = [f"{i:03d}" for i in range(len(batches))]
    for batch_id, batch_chunks in zip(batch_ids, batches):
        with open(p["local_dir"] / f"batch_{batch_id}_chunks.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch_chunks) + "\n")

    pass1 = [_add_pass(chain, p, 1, b, after, resume=False) for b in batch_ids]
    shares = _add_reduce_shares(chain, p, pass1)
    pass2 = [_add_pass(chain, p, 2, b, [shares], resume=False) for b in batch_ids]
    last = _add_reduce_write(chain, p, pass2)
    print(f"Exposure analysis: {len(batches)} chunk batch(es) covering {len(chunk_ids)} chunk(s).")
    return last


def add_exposure_resume_jobs(chain: JobChain, local_config: dict, linux_config: dict, after: list[str]) -> str:
    """Add only the pass1/pass2 batches missing output, then both reduce jobs; return the last job's name.

    Reuses each batch's batch_<id>_chunks.txt: rebalancing would break the batch id <-> output file
    convention. Raises SystemExit if there are none.
    """
    p = _paths(local_config, linux_config)
    batch_ids = sorted(q.stem.replace("batch_", "").replace("_chunks", "")
                       for q in p["local_dir"].glob("batch_*_chunks.txt"))
    if not batch_ids:
        raise SystemExit(f"No batch_*_chunks.txt under {p['local_dir']} - nothing to resume. Run a fresh dispatch first.")
    retry_transient_io((p["local_dir"] / "logs").mkdir, parents=True, exist_ok=True)
    missing1 = [b for b in batch_ids if not (p["local_dir"] / f"pass1_batch_{b}.json").exists()]
    missing2 = [b for b in batch_ids if not (p["local_dir"] / f"pass2_batch_{b}.pkl").exists()]
    print(f"Exposure resume: {len(batch_ids)} batch(es), {len(missing1)} pass1 and {len(missing2)} pass2 missing.")

    pass1 = [_add_pass(chain, p, 1, b, after, resume=True) for b in missing1]
    # The reduce jobs glob every batch's output, so they run in full again.
    shares = _add_reduce_shares(chain, p, pass1 or after)
    pass2 = [_add_pass(chain, p, 2, b, [shares], resume=True) for b in missing2]
    return _add_reduce_write(chain, p, pass2 or [shares])


def _add_reduce_shares(chain: JobChain, p: dict, after: list[str]) -> str:
    return _add(chain, p, "exposure_reduce_shares", [
        'echo "=== Reducing pass-1 shares ==="',
        f'"$GFM_PY" "{p["scripts"]}/reduce_exposure_shares.py" '
        f'--parts-glob "{p["linux_dir"]}/pass1_batch_*.json" --out "{p["shares"]}"',
    ], after, stage=False)


def _add_reduce_write(chain: JobChain, p: dict, after: list[str]) -> str:
    return _add(chain, p, "exposure_reduce_write", [
        'echo "=== Reducing pass-2 results and writing exposure CSVs ==="',
        f'"$GFM_PY" "{p["scripts"]}/reduce_exposure_write.py" --config "$LOCAL_CONFIGFILE" '
        f'--parts-glob "{p["linux_dir"]}/pass2_batch_*.pkl" --shares "{p["shares"]}" --out-dir "{p["outdir"]}"',
    ], after, stage=True)


def generate(config_path: str | Path, *, resume: bool = False, profile_sets: list[str] | None = None) -> str:
    """Write the standalone exposure chain (or its resume) and its submit script; return its Linux path."""
    config_path = Path(config_path)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)
    apply_profile_sets(linux_config["hpc"], profile_sets)
    p = _paths(local_config, linux_config)
    chain = JobChain(linux_config["hpc"], p["local_dir"], p["linux_dir"])
    if resume:
        add_exposure_resume_jobs(chain, local_config, linux_config, after=[])
        return chain.write_submit("submit_resume_exposure_analysis.sh")
    # Local view: the chunk glob must run against this machine's mount.
    merged_dir = Path(local_config["postprocessing"]["merged_outputs"])
    chunk_ids, _ = discover_chunk_ids(merged_dir / "chunks" / "flood_fraction", merged_dir / "chunks")
    if not chunk_ids:
        raise SystemExit(f"No populated chunk files found under {merged_dir}/chunks - run postprocessing first.")
    add_exposure_jobs(chain, local_config, linux_config, chunk_ids, after=[])
    return chain.write_submit("submit_exposure_analysis.sh")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "config.yml"))
    parser.add_argument("--resume", action="store_true",
                        help="only the pass1/pass2 batches still missing output, with their original chunks")
    add_profile_set_argument(parser)
    args = parser.parse_args()
    submit = generate(args.config, resume=args.resume, profile_sets=args.profile_set)
    print(f"Submit on the cluster with: bash {submit}")


if __name__ == "__main__":
    main()
