"""Generate one sbatch script per HPC node batch to run Aqueduct in parallel.

Invoked by rule generate_aqueduct_jobs (hpc_dispatch.smk) once every
preprocessing output exists. Does two path resolutions side by side, since
it runs on the local preprocessing machine but writes instructions meant for
a Linux HPC:
  - snakemake.params.model_outputs (this machine's own, already-expanded
    view) is used to check each tile/scenario's boundaries file for
    emptiness and write nodata placeholders locally, exactly like
    scripts/run_aqueduct.py's Snakemake-driven equivalent does per-job.
  - config_utils.load_config(..., extra_override="config_hpc.yml") is
    loaded separately to get the Linux-expanded view (model_outputs,
    aqueduct_executable, code_root) embedded into the generated sbatch
    scripts and resolved_config.yml - see config_hpc.yml.example.

Tiles (not individual jobs) are grouped whole and split evenly across
hpc.n_nodes batches, so a tile's full return_period x waterlevel_name set
always runs on one node.
"""

import os
import sys
from pathlib import Path

import geopandas as gpd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, retry_transient_io  # noqa: E402
from rasters import save_nodata_raster  # noqa: E402

model_outputs = snakemake.params.model_outputs  # noqa: F821
tile_ids = snakemake.params.tile_ids  # noqa: F821
return_periods = snakemake.params.return_periods  # noqa: F821
waterlevel_names = snakemake.params.waterlevel_names  # noqa: F821
raster_config = snakemake.params.raster_config  # noqa: F821
hpc_cfg = snakemake.params.hpc_cfg  # noqa: F821
base_config_path = Path(snakemake.params.base_config_path)  # noqa: F821

# ── Local view: resolve/exclude boundaries-empty jobs on this machine ───────
skipped_dir = os.path.join(model_outputs, "skipped_tiles")
jobs_by_tile: dict[str, list[tuple[str, str]]] = {tid: [] for tid in tile_ids}
n_excluded = 0

for tile_id in tile_ids:
    tile_dir = os.path.join(model_outputs, str(tile_id))
    dem_path = os.path.join(tile_dir, "inputs", "dem.tif")
    for rp in return_periods:
        for slr in waterlevel_names:
            scenario_name = f"{rp}_{slr}"
            boundaries_path = os.path.join(tile_dir, "inputs", f"boundaries_{scenario_name}.gpkg")
            boundaries = retry_transient_io(gpd.read_file, boundaries_path)
            if boundaries.empty:
                output_path = os.path.join(tile_dir, "results", f"waterdepth_{scenario_name}.tif")
                retry_transient_io(os.makedirs, os.path.dirname(output_path), exist_ok=True)
                save_nodata_raster(dem_path, output_path, raster_config)
                n_excluded += 1
            else:
                jobs_by_tile[str(tile_id)].append((rp, slr))

print(f"Excluded {n_excluded} job(s) with no boundary stations (nodata placeholder written locally).")

# ── Linux view: fully Linux-expanded config, for the sbatch scripts ─────────
linux_config = load_config(base_config_path, extra_override=base_config_path.parent / "config_hpc.yml")
linux_aqueduct_cli = f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts/run_aqueduct_cli.py"
linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

local_jobs_dir = Path(hpc_cfg["jobs_dir"])
retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

with open(snakemake.output.resolved_config, "w", encoding="utf-8") as f:  # noqa: F821
    yaml.safe_dump(linux_config, f)

# ── Partition tiles evenly across n_nodes batches (whole tiles per batch) ───
n_nodes = hpc_cfg["n_nodes"]
tile_id_list = [str(t) for t in tile_ids]
k, m = divmod(len(tile_id_list), n_nodes)
batches = [
    tile_id_list[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
    for i in range(n_nodes)
]

sbatch_cfg = hpc_cfg["sbatch"]
for i, batch_tiles in enumerate(batches):
    batch_id = f"{i:03d}"
    script_path = snakemake.output.scripts[i]  # noqa: F821 - order matches _HPC_BATCH_IDS in hpc_dispatch.smk

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=aqueduct_batch_{batch_id}",
        f"#SBATCH --partition={sbatch_cfg['partition']}",
        f"#SBATCH --account={sbatch_cfg['account']}",
        f"#SBATCH --time={sbatch_cfg['time']}",
        f"#SBATCH --mem={sbatch_cfg['mem']}",
        f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
        f"#SBATCH --output={linux_jobs_dir}/logs/batch_{batch_id}_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/batch_{batch_id}_%j.err",
        "",
        "set -uo pipefail",  # not -e: one failed (tile,rp,slr) must not abort the rest of this node's batch
        sbatch_cfg["env_activate_cmd"],
        "",
        f'FAIL_LOG="{linux_jobs_dir}/logs/batch_{batch_id}_failures.txt"',
        ': > "$FAIL_LOG"',
        "",
        "run_job() {",
        f'  python "{linux_aqueduct_cli}" \\',
        f'    --config "{linux_resolved_config}" \\',
        '    --tile-id "$1" --return-period "$2" --waterlevel-name "$3" \\',
        '    || echo "$1 $2 $3" >> "$FAIL_LOG"',
        "}",
        "",
    ]
    for tile_id in batch_tiles:
        for rp, slr in jobs_by_tile[tile_id]:
            lines.append(f"run_job {tile_id} {rp} {slr}")
    lines.append("")

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"  wrote {script_path} ({len(batch_tiles)} tiles, {sum(len(jobs_by_tile[t]) for t in batch_tiles)} jobs)")

with open(snakemake.output.submit_all, "w", encoding="utf-8", newline="\n") as f:  # noqa: F821
    f.write("#!/bin/bash\n")
    f.write(f'for f in "{linux_jobs_dir}"/aqueduct_batch_*.sbatch; do sbatch "$f"; done\n')

print(f"\nDone. {len(batches)} sbatch script(s) + resolved_config.yml + submit_all.sh written to {local_jobs_dir}")
print(f"Fill in {base_config_path.parent / 'config_hpc.yml'} and hpc.sbatch.* placeholders before submitting for real.")
