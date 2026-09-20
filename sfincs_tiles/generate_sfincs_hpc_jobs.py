"""Generate SLURM batch scripts to RUN already-built SFINCS tile models on
Deltares' Hydrax/h7 cluster, via Apptainer, split across N nodes at
CPUS_PER_TASK_DEFAULT cores each (per Deltares' own SFINCS-on-HPC guidance:
the `sfincs` binary itself is invoked through a container image, no conda
env needed on the compute node for the run step). SFINCS's own OpenMP
parallelism (OMP_NUM_THREADS, set to match the node's own allocated core
count exactly) is what actually uses the extra cores - originally 1
core/1vcpu, switched to 4 cores/4vcpu (2026-09) after live evidence several
of the 258 test tiles' real SFINCS UTM grids run up to 34M cells (see
generate_sfincs_hpc_jobs.py's own PARTITION_DEFAULT/CPUS_PER_TASK_DEFAULT
comments) - single-threaded was too slow for those.

Two-phase split, deliberately:
  - BUILD (SfincsModel via hydromt_sfincs - sfincs.inp, .dep, .msk, .man,
    .bnd, .bzs) stays on Windows, under the hydromt-sfincs-dev env, exactly
    as done so far (build_elevation.py -> ... -> build_sfincs_tile.py) -
    hydromt_sfincs itself is NOT available as a container/HPC-side install
    here, only the compiled sfincs solver is.
  - RUN (the actual `sfincs` solve) happens on HPC via this script's own
    generated sbatch jobs - pure bash + apptainer, no Python needed on the
    compute node at all.
  - POSTPROCESS (max inundation/extent from sfincs_map.nc) stays local
    again afterward - cheap, plain rasterio/xarray, no reason to burn an
    HPC allocation on it. Once a tile's sfincs_map.nc has been copied back
    to {root}/validation_sfincs/{tile_id}/sfincs_model/ by its own batch
    job, run:
        python run_sfincs_tile.py --tile-id <id> --skip-run
    (under hydromt-sfincs-dev, or any env with rasterio/xarray - see that
    script's own --skip-run flag).

This script itself needs no hydromt_sfincs - just config_utils.py (for the
same local/Linux dual path-view load_config() every other HPC job
generator in this repo uses) and stdlib/PyYAML. Run it under the MAIN
gfm_python_preprocessing env, NOT hydromt-sfincs-dev - consistent with
plot_sfincs_build.py's own note on which of these two envs actually has a
working install of what each script needs.

Node-local staging (the P:\\ latency note): SFINCS itself reads/writes many
small files repeatedly during a run (dtmapout snapshots, progress checks),
and generate_aqueduct_jobs.py's own _stage_configfile_lines() already
documents real, live SMB flakiness reading small files repeatedly from the
shared P:\\ mount from a Hydrax compute node. Rather than run SFINCS
directly against the P:\\-mounted sfincs_model/ dir, each generated job
copies that tile's ALREADY-BUILT input directory to node-local scratch
($TMPDIR) first (with a retry loop, same spirit as
_stage_configfile_lines), runs entirely there, and copies back only the
real OUTPUT files (sfincs_map.nc, sfincs.log) afterward - not the whole
directory, since the (larger, unchanged) input files never need
re-uploading.

Batches are fully independent (unlike the eikonal model's own hop_distance
wave-barrier dispatch - every SFINCS tile here is self-contained, no
neighbour-seeding), so submit_sfincs_batches.sh submits all of them
immediately, with no --dependency chain at all.

Usage:
    python generate_sfincs_hpc_jobs.py --tile-ids-file ../validation_sfincs/test_tile_selection_ids.txt
    bash <printed submit_sfincs_batches.sh path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402

N_NODES_DEFAULT = 20
PARTITION_DEFAULT = "4vcpu"  # 4 cores per node, per user instruction (2026-09: switched
# from the original 1vcpu/1-core design after live evidence several of the 258 test
# tiles' real SFINCS UTM grids run up to 34M cells single-threaded - see
# generate_sfincs_hpc_jobs.py's own git history / conversation for the investigation)
TIME_DEFAULT = "08:00:00"
CPUS_PER_TASK_DEFAULT = 4
MEM_DEFAULT = "30G"  # matches the main pipeline's own hpc.sbatch_large convention for the
# 4vcpu partition (config.yml) - Hydrax's regular partitions scale RAM at a fixed 8GB/vCPU,
# with a bit less than the nominal amount actually usable (see hpc.md's own note for 1vcpu)
SIF_PATH_DEFAULT = (
    # NOT .../SFINCS_2026_branches/v2.4.0_Galibier_Release_CPU_apptainer/... -
    # that path (from the guidance this was originally built from) doesn't
    # exist on the share (confirmed live: apptainer's own "no such file or
    # directory" on tile 37's first real HPC run) - verified this IS the
    # real path by browsing the share directly (`find ... -iname "*.sif"`).
    "/p/11202255-sfincs/executables/SFINCS_2026/SFINCS_2026_01/"
    "v2.4.0_Galibier_Release_CPU_apptainer/sfincs-cpu_v2.4.0-Galibier-Release.sif"
)


def _stage_tile_lines(cpus_per_task: int) -> list[str]:
    """Bash function: stage one tile's ALREADY-BUILT sfincs_model/ dir to
    node-local scratch (retrying the copy a few times - see module
    docstring on real, documented SMB flakiness for repeated small-file
    reads from the same compute node), run SFINCS there via apptainer, then
    copy back only the real output files. Never aborts the batch on a
    single tile's failure (that's `run_tile`'s own job, not the caller's) -
    logs to FAIL_LOG and returns.
    """
    return [
        "run_tile() {",
        '  local tile_id="$1"',
        '  local remote_dir="$SFINCS_ROOT/validation_sfincs/${tile_id}/sfincs_model"',
        '  local local_dir="${TMPDIR:-/tmp}/sfincs_${tile_id}_${SLURM_JOB_ID:-$$}"',
        "",
        '  if [ ! -f "$remote_dir/sfincs.inp" ]; then',
        '    echo "$tile_id  missing sfincs.inp (build step not done on Windows side)" >> "$FAIL_LOG"',
        "    return",
        "  fi",
        "",
        '  rm -rf "$local_dir"; mkdir -p "$local_dir"',
        "  local attempt=1",
        '  while [ "$attempt" -le 5 ]; do',
        '    cp -r "$remote_dir"/. "$local_dir"/ 2>/dev/null',
        '    [ -f "$local_dir/sfincs.inp" ] && break',
        '    echo "  [stage retry $attempt/4] tile $tile_id input copy failed - retrying in 5s..." >&2',
        "    sleep 5",
        "    attempt=$((attempt + 1))",
        "  done",
        '  if [ ! -f "$local_dir/sfincs.inp" ]; then',
        '    echo "$tile_id  failed to stage input to node-local scratch after 5 attempts" >> "$FAIL_LOG"',
        '    rm -rf "$local_dir"',
        "    return",
        "  fi",
        "",
        f"  export OMP_NUM_THREADS={cpus_per_task}",  # match the node's own allocated core count exactly
        '  ( cd "$local_dir" && apptainer exec -B "$local_dir":/mnt/data "$SIF_PATH" sfincs ) '
        '> "$local_dir/sfincs_hpc_run.log" 2>&1',
        "  local run_rc=$?",
        "",
        '  if [ "$run_rc" -ne 0 ] || [ ! -f "$local_dir/sfincs_map.nc" ]; then',
        '    echo "$tile_id  sfincs run failed (exit $run_rc) or produced no sfincs_map.nc" >> "$FAIL_LOG"',
        '    cp -f "$local_dir/sfincs_hpc_run.log" "$remote_dir/" 2>/dev/null',
        '    rm -rf "$local_dir"',
        "    return",
        "  fi",
        "",
        '  cp -f "$local_dir/sfincs_map.nc" "$remote_dir/"',
        '  cp -f "$local_dir/sfincs.log" "$remote_dir/" 2>/dev/null',
        '  cp -f "$local_dir/sfincs_hpc_run.log" "$remote_dir/"',
        '  rm -rf "$local_dir"',
        '  echo "tile $tile_id done"',
        "}",
    ]


def generate_sfincs_batches(
    tile_ids: list[str],
    n_nodes: int,
    partition: str,
    time_limit: str,
    mem: str,
    cpus_per_task: int,
    account: str,
    sif_path: str,
    linux_root: str,
    linux_jobs_dir: str,
    local_jobs_dir: Path,
    submit_path: Path,
) -> None:
    n_batches = min(n_nodes, len(tile_ids))
    k, m = divmod(len(tile_ids), n_batches)
    batches = []
    for i in range(n_batches):
        batch_tiles = tile_ids[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
        if batch_tiles:
            batches.append((f"{i:03d}", batch_tiles))

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    script_paths = []
    for batch_id, batch_tiles in batches:
        name = f"sfincs_batch_{batch_id}"
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={partition}",
        ]
        if account:
            lines.append(f"#SBATCH --account={account}")
        lines += [
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",  # not -e: one tile's failure must not abort the rest of this node's batch
            f'SFINCS_ROOT="{linux_root}"',
            f'SIF_PATH="{sif_path}"',
            f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
            ': > "$FAIL_LOG"',
            "",
            *_stage_tile_lines(cpus_per_task),
            "",
        ]
        for tile_id in batch_tiles:
            lines.append(f"run_tile {tile_id}")
        lines.append("")

        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({len(batch_tiles)} tile(s))")

    # Fully independent batches - no wave/dependency barrier needed (unlike
    # the eikonal model's own hop_distance neighbour-seeding), so every
    # batch submits immediately.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", ""]
    for script in script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
        ]
    with open(submit_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    print(f"\nDone. {len(batches)} batch(es), {len(tile_ids)} tile(s) total.")
    print(f"Wrote {submit_path}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(_repo_root / "snakemake_workflow" / "config" / "config.yml"))
    parser.add_argument("--tile-ids-file", default=None, help="text file, one tile ID per line (default: validation_sfincs/test_tile_selection_ids.txt)")
    parser.add_argument("--n-nodes", type=int, default=N_NODES_DEFAULT)
    parser.add_argument("--partition", default=PARTITION_DEFAULT)
    parser.add_argument("--time", default=TIME_DEFAULT)
    parser.add_argument("--mem", default=MEM_DEFAULT)
    parser.add_argument("--cpus-per-task", type=int, default=CPUS_PER_TASK_DEFAULT)
    parser.add_argument("--account", default="")
    parser.add_argument("--sif-path", default=SIF_PATH_DEFAULT)
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_root = Path(local_config["paths"]["root"])
    linux_root = linux_config["paths"]["root"]

    tile_ids_file = Path(args.tile_ids_file) if args.tile_ids_file else local_root / "validation_sfincs" / "test_tile_selection_ids.txt"
    tile_ids = [line.strip() for line in tile_ids_file.read_text().splitlines() if line.strip()]
    if not tile_ids:
        raise ValueError(f"no tile IDs found in {tile_ids_file}")
    print(f"{len(tile_ids)} tile(s) from {tile_ids_file}")

    local_jobs_dir = local_root / "validation_sfincs" / "hpc_jobs"
    linux_jobs_dir = f"{linux_root}/validation_sfincs/hpc_jobs"

    generate_sfincs_batches(
        tile_ids=tile_ids,
        n_nodes=args.n_nodes,
        partition=args.partition,
        time_limit=args.time,
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        account=args.account,
        sif_path=args.sif_path,
        linux_root=linux_root,
        linux_jobs_dir=linux_jobs_dir,
        local_jobs_dir=local_jobs_dir,
        submit_path=local_jobs_dir / "submit_sfincs_batches.sh",
    )
    print(f"\nSubmit on Hydrax with: bash {linux_jobs_dir}/submit_sfincs_batches.sh")


if __name__ == "__main__":
    main()
