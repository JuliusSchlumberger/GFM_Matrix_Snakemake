"""Generate HPC sbatch scripts for the chunk-partitioned exposure-analysis pipeline.

`generate_exposure_dispatch()` is used standalone (main(): chunk_ids globbed from
real flood_fraction files on the local mount) and by generate_hpc_postprocess_job.py
(unfiltered tile-grid-derived chunk_ids, appended to postprocessing's own submit
script). The whole chain is submitted once from the login node: a compute-node job
calling sbatch again has hung before. pass1/pass2 skip unpopulated chunks inline.
Uses hpc.n_nodes/hpc.sbatch (I/O-bound jobs); embedded paths use config_hpc.yml.

Usage:
    python generate_hpc_exposure_jobs.py [--config path/to/config.yml]
    bash <printed submit_exposure_analysis.sh path>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import load_config, load_hpc_config, retry_transient_io  # noqa: E402
from hpc_common import sbatch_header  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from _common import discover_chunk_ids  # noqa: E402


def _job_header(name: str, sbatch_cfg: dict, linux_jobs_dir: str, linux_code_root: str) -> list[str]:
    """Return an exposure job's sbatch header followed by the cd into the code root."""
    return [*sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs"), f'cd "{linux_code_root}"']


def _write_pass_script(
    pass_no: int,
    batch_id: str,
    *,
    local_jobs_dir: Path,
    linux_jobs_dir: str,
    linux_code_root: str,
    linux_config_path: str,
    linux_shares_path: str,
    sbatch_cfg: dict,
    resume: bool,
) -> str:
    """Write `exposure_pass{pass_no}_batch_{batch_id}.sbatch` and return its Linux path.

    Pass 1 writes the batch's share JSON; pass 2 reads the reduced shares and
    writes the batch's EAI pickle. `resume` only marks the log banner.
    """
    linux_scripts_dir = f"{linux_code_root}/snakemake_workflow/scripts"
    chunks_file = f"{linux_jobs_dir}/batch_{batch_id}_chunks.txt"
    name = f"exposure_pass{pass_no}_batch_{batch_id}"
    if pass_no == 1:
        command = (
            f'python "{linux_scripts_dir}/run_exposure_pass1.py" --config "{linux_config_path}" '
            f'--chunks-file "{chunks_file}" --out "{linux_jobs_dir}/pass1_batch_{batch_id}.json"'
        )
    else:
        command = (
            f'python "{linux_scripts_dir}/run_exposure_pass2.py" --config "{linux_config_path}" '
            f'--chunks-file "{chunks_file}" --shares "{linux_shares_path}" '
            f'--out "{linux_jobs_dir}/pass2_batch_{batch_id}.pkl"'
        )
    banner = f"Exposure pass {pass_no}{' (resume)' if resume else ''}, batch {batch_id}"
    lines = _job_header(name, sbatch_cfg, linux_jobs_dir, linux_code_root) + [
        f'echo "=== {banner} ==="',
        command,
        "",
    ]
    with open(local_jobs_dir / f"{name}.sbatch", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return f"{linux_jobs_dir}/{name}.sbatch"


def generate_exposure_dispatch(
    local_config: dict,
    linux_config: dict,
    chunk_ids: list[str],
    submit_lines: list[str],
    prev_ids_expr: str,
) -> list[str]:
    """Write pass1 -> reduce_shares -> pass2 -> reduce_write sbatch scripts.

    `chunk_ids` are split evenly across hpc.n_nodes and used as given.
    `prev_ids_expr` is a bash expression (e.g. `"$POSTPROCESS_IDS"`) pass1
    depends on via afterany, or `""`. Returns the extended `submit_lines`
    without writing them. Raises SystemExit if `chunk_ids` is empty.
    """
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"]) / "exposure"
    linux_jobs_dir = f"{linux_config['hpc']['jobs_dir']}/exposure"
    linux_code_root = linux_config["paths"]["code_root"]
    linux_scripts_dir = f"{linux_code_root}/snakemake_workflow/scripts"
    # resolved_config.yml so a scenario dispatch's own config reaches these jobs.
    linux_config_path = f"{linux_config['hpc']['jobs_dir']}/resolved_config.yml"
    linux_outdir = f"{linux_config['postprocessing']['merged_outputs']}/exposure"
    linux_shares_path = f"{linux_jobs_dir}/shares_by_intensity.json"
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    if not chunk_ids:
        raise SystemExit("generate_exposure_dispatch: chunk_ids is empty - nothing to plan.")

    n_batches = min(n_nodes, len(chunk_ids))
    k, m = divmod(len(chunk_ids), n_batches)
    batches = [
        chunk_ids[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        for i in range(n_batches)
    ]
    batch_ids = [f"{i:03d}" for i in range(n_batches)]

    for batch_id, batch_chunks in zip(batch_ids, batches):
        chunks_path = local_jobs_dir / f"batch_{batch_id}_chunks.txt"
        with open(chunks_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(batch_chunks) + "\n")

    sbatch_cfg = hpc_cfg["sbatch"]
    pass_kw = dict(
        local_jobs_dir=local_jobs_dir, linux_jobs_dir=linux_jobs_dir, linux_code_root=linux_code_root,
        linux_config_path=linux_config_path, linux_shares_path=linux_shares_path,
        sbatch_cfg=sbatch_cfg, resume=False,
    )

    pass1_script_paths = []
    pass2_script_paths = []
    for batch_id in batch_ids:
        pass1_script_paths.append(_write_pass_script(1, batch_id, **pass_kw))
        pass2_script_paths.append(_write_pass_script(2, batch_id, **pass_kw))

    print(f"  wrote {n_batches} pass1 + {n_batches} pass2 batch script(s) to {local_jobs_dir}")

    reduce_shares_lines = _job_header("exposure_reduce_shares", sbatch_cfg, linux_jobs_dir, linux_code_root) + [
        'echo "=== Reducing pass-1 shares ==="',
        (
            f'python "{linux_scripts_dir}/reduce_exposure_shares.py" '
            f'--parts-glob "{linux_jobs_dir}/pass1_batch_*.json" --out "{linux_shares_path}"'
        ),
        "",
    ]
    reduce_shares_path = local_jobs_dir / "exposure_reduce_shares.sbatch"
    with open(reduce_shares_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(reduce_shares_lines))
    print(f"  wrote {reduce_shares_path}")

    reduce_write_lines = _job_header("exposure_reduce_write", sbatch_cfg, linux_jobs_dir, linux_code_root) + [
        'echo "=== Reducing pass-2 results and writing exposure CSVs ==="',
        (
            f'python "{linux_scripts_dir}/reduce_exposure_write.py" --config "{linux_config_path}" '
            f'--parts-glob "{linux_jobs_dir}/pass2_batch_*.pkl" --shares "{linux_shares_path}" '
            f'--out-dir "{linux_outdir}"'
        ),
        "",
    ]
    reduce_write_path = local_jobs_dir / "exposure_reduce_write.sbatch"
    with open(reduce_write_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(reduce_write_lines))
    print(f"  wrote {reduce_write_path}")

    # pass1 batches -> reduce_shares -> pass2 batches -> reduce_write, each afterany.
    submit_lines = list(submit_lines)
    submit_lines.append("\n# exposure analysis: pass 1 (shares)")
    submit_lines.append('PASS1_IDS=""')
    for script in pass1_script_paths:
        if prev_ids_expr:
            submit_lines.append(f'JID=$(sbatch --parsable --dependency=afterany:{prev_ids_expr} "{script}")')
        else:
            submit_lines.append(f'JID=$(sbatch --parsable "{script}")')
        submit_lines += [
            f'echo "submitted {script} -> job $JID"',
            'PASS1_IDS="${PASS1_IDS:+$PASS1_IDS:}$JID"',
        ]
    submit_lines += [
        "",
        (
            'SHARES_JID=$(sbatch --parsable --dependency=afterany:$PASS1_IDS '
            f'"{linux_jobs_dir}/exposure_reduce_shares.sbatch")'
        ),
        (
            f'echo "submitted {linux_jobs_dir}/exposure_reduce_shares.sbatch -> job $SHARES_JID '
            f'(depends on {n_batches} pass1 batches)"'
        ),
        "",
        'PASS2_IDS=""',
    ]
    for script in pass2_script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable --dependency=afterany:$SHARES_JID "{script}")',
            f'echo "submitted {script} -> job $JID"',
            'PASS2_IDS="${PASS2_IDS:+$PASS2_IDS:}$JID"',
        ]
    submit_lines += [
        "",
        (
            'JID=$(sbatch --parsable --dependency=afterany:$PASS2_IDS '
            f'"{linux_jobs_dir}/exposure_reduce_write.sbatch")'
        ),
        (
            f'echo "submitted {linux_jobs_dir}/exposure_reduce_write.sbatch -> job $JID '
            f'(depends on {n_batches} pass2 batches)"'
        ),
    ]

    print(f"\nExposure analysis: {n_batches} chunk batch(es) covering {len(chunk_ids)} chunk(s).")
    return submit_lines


def generate_exposure_resume_dispatch(
    local_config: dict,
    linux_config: dict,
    submit_lines: list[str],
    prev_ids_expr: str,
) -> list[str]:
    """Regenerate and resubmit only the exposure pass1/pass2 batches missing output.

    Reuses each batch's original `batch_{id}_chunks.txt`: rebalancing would break
    the batch_id <-> output filename convention. Returns the extended
    `submit_lines`. Raises SystemExit if no `batch_*_chunks.txt` exist.
    """
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"]) / "exposure"
    linux_jobs_dir = f"{linux_config['hpc']['jobs_dir']}/exposure"
    linux_code_root = linux_config["paths"]["code_root"]
    # resolved_config.yml so a scenario dispatch's own config reaches these jobs.
    linux_config_path = f"{linux_config['hpc']['jobs_dir']}/resolved_config.yml"
    linux_outdir = f"{linux_config['postprocessing']['merged_outputs']}/exposure"
    linux_shares_path = f"{linux_jobs_dir}/shares_by_intensity.json"
    hpc_cfg = linux_config["hpc"]
    sbatch_cfg = hpc_cfg["sbatch"]

    batch_ids = sorted(
        p.stem.replace("batch_", "").replace("_chunks", "")
        for p in local_jobs_dir.glob("batch_*_chunks.txt")
    )
    if not batch_ids:
        raise SystemExit(
            f"No batch_*_chunks.txt found under {local_jobs_dir} - nothing to resume. "
            "Run a fresh dispatch first."
        )

    pass_kw = dict(
        local_jobs_dir=local_jobs_dir, linux_jobs_dir=linux_jobs_dir, linux_code_root=linux_code_root,
        linux_config_path=linux_config_path, linux_shares_path=linux_shares_path,
        sbatch_cfg=sbatch_cfg, resume=True,
    )

    missing_pass1 = [bid for bid in batch_ids if not (local_jobs_dir / f"pass1_batch_{bid}.json").exists()]
    missing_pass2 = [bid for bid in batch_ids if not (local_jobs_dir / f"pass2_batch_{bid}.pkl").exists()]
    print(f"Exposure resume: {len(batch_ids)} batch(es) total, "
          f"{len(missing_pass1)} pass1 missing, {len(missing_pass2)} pass2 missing.")

    submit_lines = list(submit_lines)

    submit_lines.append("\n# exposure analysis resume: pass 1 (shares) - only missing batches")
    submit_lines.append('PASS1_IDS=""')
    for bid in missing_pass1:
        script = _write_pass_script(1, bid, **pass_kw)
        if prev_ids_expr:
            submit_lines.append(f'JID=$(sbatch --parsable --dependency=afterany:{prev_ids_expr} "{script}")')
        else:
            submit_lines.append(f'JID=$(sbatch --parsable "{script}")')
        submit_lines += [
            f'echo "submitted {script} -> job $JID"',
            'PASS1_IDS="${PASS1_IDS:+$PASS1_IDS:}$JID"',
        ]

    # The reduce scripts are pure glob consumers, so they are resubmitted unchanged.
    submit_lines.append("")
    shares_dep = "$PASS1_IDS" if missing_pass1 else prev_ids_expr
    if shares_dep:
        submit_lines.append(
            f'SHARES_JID=$(sbatch --parsable --dependency=afterany:{shares_dep} '
            f'"{linux_jobs_dir}/exposure_reduce_shares.sbatch")'
        )
    else:
        submit_lines.append(f'SHARES_JID=$(sbatch --parsable "{linux_jobs_dir}/exposure_reduce_shares.sbatch")')
    submit_lines.append(f'echo "submitted {linux_jobs_dir}/exposure_reduce_shares.sbatch -> job $SHARES_JID"')

    submit_lines.append("\n# exposure analysis resume: pass 2 (EAI) - only missing batches")
    submit_lines.append('PASS2_IDS=""')
    for bid in missing_pass2:
        script = _write_pass_script(2, bid, **pass_kw)
        submit_lines += [
            f'JID=$(sbatch --parsable --dependency=afterany:$SHARES_JID "{script}")',
            f'echo "submitted {script} -> job $JID"',
            'PASS2_IDS="${PASS2_IDS:+$PASS2_IDS:}$JID"',
        ]

    submit_lines.append("")
    write_dep = "$PASS2_IDS" if missing_pass2 else "$SHARES_JID"
    submit_lines.append(
        f'JID=$(sbatch --parsable --dependency=afterany:{write_dep} '
        f'"{linux_jobs_dir}/exposure_reduce_write.sbatch")'
    )
    submit_lines.append(f'echo "submitted {linux_jobs_dir}/exposure_reduce_write.sbatch -> job $JID"')

    print(f"Exposure resume: {len(missing_pass1)} pass1 + {len(missing_pass2)} pass2 batch(es) "
          f"regenerated; reduce_shares/reduce_write resubmitted unchanged. outdir={linux_outdir}")
    return submit_lines


def main() -> None:
    """Write (or with --resume, resubmit) the standalone exposure dispatch."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--resume", action="store_true",
        help="regenerate + resubmit only the pass1/pass2 batches still missing output, "
             "reusing their original chunk assignment - see generate_exposure_resume_dispatch.",
    )
    args = parser.parse_args()

    if args.resume:
        config_path = Path(args.config)
        local_config = load_config(config_path)
        linux_config = load_hpc_config(config_path)
        submit_lines = generate_exposure_resume_dispatch(
            local_config, linux_config, submit_lines=["#!/bin/bash", "set -euo pipefail"], prev_ids_expr="",
        )
        local_jobs_dir = Path(local_config["hpc"]["jobs_dir"]) / "exposure"
        submit_script_path = local_jobs_dir / "submit_resume_exposure_analysis.sh"
        with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(submit_lines) + "\n")
        linux_jobs_dir = f"{linux_config['hpc']['jobs_dir']}/exposure"
        print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/submit_resume_exposure_analysis.sh")
        return

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_hpc_config(config_path)

    # Local view: the chunk glob must run against this machine's mount.
    merged_dir = Path(local_config["postprocessing"]["merged_outputs"])
    chunk_ids, _ = discover_chunk_ids(merged_dir / "chunks" / "flood_fraction", merged_dir / "chunks")
    if not chunk_ids:
        raise SystemExit(f"No populated chunk files found under {merged_dir}/chunks - run postprocessing first.")

    submit_lines = generate_exposure_dispatch(
        local_config, linux_config, chunk_ids,
        submit_lines=["#!/bin/bash", "set -euo pipefail"],
        prev_ids_expr="",
    )

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"]) / "exposure"
    submit_script_path = local_jobs_dir / "submit_exposure_analysis.sh"
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    linux_jobs_dir = f"{linux_config['hpc']['jobs_dir']}/exposure"
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/submit_exposure_analysis.sh")


if __name__ == "__main__":
    main()
