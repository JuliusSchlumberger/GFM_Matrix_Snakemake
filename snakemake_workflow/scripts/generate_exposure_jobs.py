"""Generate HPC sbatch scripts for the chunk-partitioned exposure-analysis
pipeline (analysis/compute_exposure_analysis.py's pass1_shares_all_
intensities / pass2_all_tasks design - see that module's docstring for why
CHUNKS, not individual scenario/adaptation tasks, are the partitioning unit).

The actual generation logic lives in `generate_exposure_dispatch()`, shared
by two entry points:
  - `main()` below (`python generate_exposure_jobs.py`) - standalone use,
    e.g. regenerating/resuming the exposure dispatch after postprocessing
    has already run on its own. Discovers chunk_ids by globbing real
    flood_fraction files (same filter compute_exposure_analysis.py's own
    main() uses), and writes its own self-contained submit_exposure_analysis.sh
    with no incoming dependency (pass1 batches start immediately).
  - `generate_hpc_postprocess_job.py` (2026-08-15, new) - calls
    generate_exposure_dispatch() directly with the SAME tile-grid-derived
    chunk_ids list postprocessing's own batches use (not a file glob, since
    postprocessing hasn't produced the real files yet at the moment both
    are being planned together), appending exposure analysis's own phases
    onto postprocessing's OWN submit script, continuing the same
    afterany-chain rather than writing a separate script. This is what lets
    the whole postprocessing -> exposure-analysis handoff be submitted
    ONCE, synchronously, from wherever the driver script runs (the login
    node) - critically, NO phase anywhere in that combined chain calls
    `sbatch` again from within a running compute-node job. That pattern
    (a job submitting more jobs from a compute node) is what hung job
    243654 for ~6h earlier this run (see generate_hpc_preprocess_job.py's
    build_shared_inputs docstring) - `submit_waves.sh` itself never had
    that problem because it submits every wave's jobs upfront, all at
    once, with SLURM's own --dependency=afterany handling the timing; this
    combined postprocessing+exposure chain follows that same proven shape
    instead of the one that already broke.

  Because chunk_ids may now be the UNFILTERED tile-grid-derived list (not
  pre-filtered to non-empty population), pass1_shares_all_intensities/
  pass2_all_tasks (compute_exposure_analysis.py) check each chunk's
  population file for real content themselves now (_chunk_is_populated,
  2026-08-15) rather than assuming the caller already filtered it - an
  ocean/buffer chunk with a zero-byte population placeholder is skipped
  inline instead of ever being handed to _load_chunk.

Reuses hpc.n_nodes/hpc.sbatch rather than a separate config block - same
reasoning as generate_hpc_preprocess_job.py's own choice (one place for
partition/mem choices). These jobs are I/O-bound chunk reads plus small
per-country arrays, not the memory-heavy tile solves hpc.sbatch_large exists
for, so hpc.sbatch alone is enough - no "large" size class here.

Uses the same local-view/Linux-view dual path resolution as
generate_hpc_preprocess_job.py (config_hpc.yml, if present) for the paths
embedded INTO the generated sbatch scripts - but chunk discovery in the
standalone path (a filesystem glob, needs to actually run against whatever
mount the CURRENT machine sees) uses the local config, not the Linux one,
so this works correctly regardless of which machine generates the scripts.

Usage:
    python generate_exposure_jobs.py [--config path/to/config.yml]
    bash <printed submit_exposure_analysis.sh path>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, retry_transient_io  # noqa: E402


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


def _discover_chunk_ids(flood_frac_dir: Path, chunks_dir: Path) -> list[str]:
    """Mirrors compute_exposure_analysis.discover_chunk_ids - kept as a plain
    path glob here (no need to import the analysis module for this) since
    generation-time discovery only needs the file listing. Standalone
    entry point only - the combined postprocessing+exposure entry point
    passes an already-known chunk_ids list in instead (see module docstring).
    """
    all_chunk_ids = sorted(set(
        p.stem.split("_RP")[0].replace("flood_fraction_", "")
        for p in flood_frac_dir.glob("flood_fraction_*.tif")
    ))
    return [
        cid for cid in all_chunk_ids
        if (chunks_dir / f"exposure_population_grid_{cid}.tif").exists()
        and (chunks_dir / f"exposure_population_grid_{cid}.tif").stat().st_size > 0
    ]


def generate_exposure_dispatch(
    local_config: dict,
    linux_config: dict,
    chunk_ids: list[str],
    submit_lines: list[str],
    prev_ids_expr: str,
) -> list[str]:
    """Write the exposure-analysis pass1/reduce_shares/pass2/reduce_write
    sbatch scripts and APPEND their submission onto an existing
    `submit_lines` script, rather than building a fresh standalone one.

    `chunk_ids` is taken as given - NOT filtered for non-empty population
    here (that filtering now happens inline in pass1_shares_all_intensities/
    pass2_all_tasks, see _chunk_is_populated - necessary because this may be
    the tile-grid-derived list, planned before the real population files
    exist).

    `prev_ids_expr`: a bash expression (e.g. `"$POSTPROCESS_IDS"`, or `""`
    for no incoming dependency) that pass1's own batches should depend on
    via `--dependency=afterany`. Returns the extended `submit_lines` list -
    does NOT write it to disk itself, so the caller can keep appending
    further phases before writing once at the end.
    """
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"]) / "exposure"
    linux_jobs_dir = f"{linux_config['hpc']['jobs_dir']}/exposure"
    linux_code_root = linux_config["paths"]["code_root"]
    linux_scripts_dir = f"{linux_code_root}/snakemake_workflow/scripts"
    linux_config_path = f"{linux_code_root}/snakemake_workflow/config/config.yml"
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

    def _job_header(name: str) -> list[str]:
        return [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={sbatch_cfg['partition']}",
            *_account_line(sbatch_cfg),
            f"#SBATCH --time={sbatch_cfg['time']}",
            f"#SBATCH --mem={sbatch_cfg['mem']}",
            f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -euo pipefail",
            sbatch_cfg["env_activate_cmd"],
            "",
            f'cd "{linux_code_root}"',
        ]

    pass1_script_paths = []
    pass2_script_paths = []
    for batch_id in batch_ids:
        chunks_file = f"{linux_jobs_dir}/batch_{batch_id}_chunks.txt"

        name1 = f"exposure_pass1_batch_{batch_id}"
        lines1 = _job_header(name1) + [
            f'echo "=== Exposure pass 1, batch {batch_id} ==="',
            (
                f'python "{linux_scripts_dir}/run_exposure_pass1.py" --config "{linux_config_path}" '
                f'--chunks-file "{chunks_file}" --out "{linux_jobs_dir}/pass1_batch_{batch_id}.json"'
            ),
            "",
        ]
        p1_path = local_jobs_dir / f"{name1}.sbatch"
        with open(p1_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines1))
        pass1_script_paths.append(f"{linux_jobs_dir}/{name1}.sbatch")

        name2 = f"exposure_pass2_batch_{batch_id}"
        lines2 = _job_header(name2) + [
            f'echo "=== Exposure pass 2, batch {batch_id} ==="',
            (
                f'python "{linux_scripts_dir}/run_exposure_pass2.py" --config "{linux_config_path}" '
                f'--chunks-file "{chunks_file}" --shares "{linux_shares_path}" '
                f'--out "{linux_jobs_dir}/pass2_batch_{batch_id}.pkl"'
            ),
            "",
        ]
        p2_path = local_jobs_dir / f"{name2}.sbatch"
        with open(p2_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines2))
        pass2_script_paths.append(f"{linux_jobs_dir}/{name2}.sbatch")

    print(f"  wrote {n_batches} pass1 + {n_batches} pass2 batch script(s) to {local_jobs_dir}")

    reduce_shares_lines = _job_header("exposure_reduce_shares") + [
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

    reduce_write_lines = _job_header("exposure_reduce_write") + [
        'echo "=== Reducing pass-2 results and writing exposure CSVs ==="',
        (
            f'python "{linux_scripts_dir}/reduce_exposure_write.py" --config "{linux_config_path}" '
            f'--parts-glob "{linux_jobs_dir}/pass2_batch_*.pkl" --shares "{linux_shares_path}" '
            f'--outdir "{linux_outdir}"'
        ),
        "",
    ]
    reduce_write_path = local_jobs_dir / "exposure_reduce_write.sbatch"
    with open(reduce_write_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(reduce_write_lines))
    print(f"  wrote {reduce_write_path}")

    # Append (not replace) - pass1 batches (parallel amongst themselves, but
    # gated on prev_ids_expr if given) -> reduce_shares (afterany all pass1)
    # -> pass2 batches (afterany reduce_shares) -> reduce_write (afterany
    # all pass2). Same afterany-join pattern as submit_preprocess_and_
    # dispatch.sh / submit_waves.sh - see module docstring for why this
    # stays a single synchronous submission rather than a job calling
    # sbatch again later.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    # Local view (this machine's own reachable mount), not linux_config's -
    # see module docstring.
    merged_dir = Path(local_config["postprocessing"]["merged_outputs"])
    chunk_ids = _discover_chunk_ids(merged_dir / "chunks" / "flood_fraction", merged_dir / "chunks")
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
