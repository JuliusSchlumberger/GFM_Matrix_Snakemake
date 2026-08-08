"""Generate per-wave sbatch scripts to run the flood solver in parallel on HPC.

Invoked by rule generate_aqueduct_jobs (hpc_dispatch.smk) once every
preprocessing output exists. Does two path resolutions side by side, since
it runs on the local preprocessing machine but writes instructions meant for
a Linux HPC:
  - snakemake.params.model_outputs (this machine's own, already-expanded
    view) is used to pre-check wave-0 boundaries for emptiness and write
    nodata placeholders locally, exactly like scripts/run_aqueduct.py's
    Snakemake-driven equivalent does per-job.
  - config_utils.load_config(..., extra_override="config_hpc.yml") is
    loaded separately to get the Linux-expanded view (model_outputs,
    code_root) embedded into the generated sbatch scripts and
    resolved_config.yml - see config_hpc.yml.example.

Wave-0 (hop_distance == 0) tiles with no boundary stations are resolved
immediately here (nodata placeholder written locally) and dropped from the
sbatch scripts, same as before. Hop_distance >= 1 tiles are NEVER
pre-excluded this way: whether a hinterland tile has any upstream flooding
to seed from can only be known once its lower-hop neighbour has actually
run (run_aqueduct_cli.py discovers "no seeds yet" live and writes its own
zero-waterdepth result) - pre-filtering them here on a same-run-as-wave-0
basis would incorrectly exclude every hop>=1 tile, since a hinterland tile
has no nearby boundary stations by definition. This is also why waves must
be submitted as SLURM dependency barriers (see submit_waves.sh below): a
hop>=1 job started before its neighbour's job has finished would read
"no neighbour output yet" as "no flooding", not as "not run yet".

snakemake.params.batches (built in hpc_dispatch.smk from the tile grid's
hop_distance column AND each tile's estimated pixel count) already carries
the full (wave, size_class, batch_id, tile_ids) partition - this script
only turns each batch into one sbatch script (using hpc.sbatch or the
bigger-RAM hpc.sbatch_large, per the batch's size_class) and groups them
into submit_waves.sh's per-wave dependency chain (both size classes within
a wave submit together - size only picks a batch's partition, not when it
starts).
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
hop_by_tile = snakemake.params.hop_by_tile  # noqa: F821
batches = snakemake.params.batches  # noqa: F821
return_periods = snakemake.params.return_periods  # noqa: F821
waterlevel_names = snakemake.params.waterlevel_names  # noqa: F821
raster_config = snakemake.params.raster_config  # noqa: F821
hpc_cfg = snakemake.params.hpc_cfg  # noqa: F821
base_config_path = Path(snakemake.params.base_config_path)  # noqa: F821

# ── Local view: resolve/exclude wave-0 boundaries-empty jobs on this machine
jobs_by_tile: dict[str, list[tuple[str, str]]] = {tid: [] for tid in map(str, tile_ids)}
n_excluded = 0

for tile_id in map(str, tile_ids):
    tile_dir = os.path.join(model_outputs, tile_id)
    dem_path = os.path.join(tile_dir, "inputs", "dem.tif")
    is_wave0 = hop_by_tile[tile_id] == 0
    for rp in return_periods:
        for slr in waterlevel_names:
            scenario_name = f"{rp}_{slr}"
            if is_wave0:
                boundaries_path = os.path.join(tile_dir, "inputs", f"boundaries_{scenario_name}.gpkg")
                boundaries = retry_transient_io(gpd.read_file, boundaries_path)
                if boundaries.empty:
                    output_path = os.path.join(tile_dir, "results", f"waterdepth_{scenario_name}.tif")
                    retry_transient_io(os.makedirs, os.path.dirname(output_path), exist_ok=True)
                    save_nodata_raster(dem_path, output_path, raster_config)
                    n_excluded += 1
                    continue
            jobs_by_tile[tile_id].append((rp, slr))

print(f"Excluded {n_excluded} wave-0 job(s) with no boundary stations (nodata placeholder written locally).")

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

# ── Write one sbatch script per (wave, size_class, batch), in the same
#    order as snakemake.output.scripts (hpc_dispatch.smk built both from the
#    same `batches` list, so the order always matches).
scripts_by_wave: dict[int, list[str]] = {}

for (wave, size_class, batch_id, batch_tiles), script_path in zip(batches, snakemake.output.scripts):  # noqa: F821
    sbatch_cfg = hpc_cfg["sbatch_large"] if size_class == "large" else hpc_cfg["sbatch"]
    name = f"wave{wave}_{size_class}_batch_{batch_id}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={sbatch_cfg['partition']}",
    ]
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        lines.append(f"#SBATCH --account={sbatch_cfg['account']}")
    lines += [
        f"#SBATCH --time={sbatch_cfg['time']}",
        f"#SBATCH --mem={sbatch_cfg['mem']}",
        f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
        f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
        "",
        "set -uo pipefail",  # not -e: one failed (tile,rp,slr) must not abort the rest of this node's batch
        sbatch_cfg["env_activate_cmd"],
        "",
        f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
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
    n_jobs = 0
    for tile_id in batch_tiles:
        for rp, slr in jobs_by_tile[tile_id]:
            lines.append(f"run_job {tile_id} {rp} {slr}")
            n_jobs += 1
    lines.append("")

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    scripts_by_wave.setdefault(wave, []).append(f"{linux_jobs_dir}/{name}.sbatch")
    print(f"  wrote {script_path} (wave {wave}, {size_class}, {len(batch_tiles)} tiles, {n_jobs} jobs)")

# ── submit_waves.sh: submits every wave to SLURM in order, each wave
#    depending on EVERY job in the previous wave (afterany - a wave starts
#    once its predecessor has finished, success or failure, since a failed
#    tile there just means that tile's neighbours fall back to a real-zero
#    result, not that the whole wave should be blocked).
submit_lines = [
    "#!/bin/bash",
    "set -euo pipefail",
    "",
    "PREV_IDS=\"\"",
]
for wave in sorted(scripts_by_wave):
    submit_lines.append(f"\n# wave {wave} ({len(scripts_by_wave[wave])} batch(es))")
    submit_lines.append("IDS=\"\"")
    for script in scripts_by_wave[wave]:
        submit_lines += [
            'if [ -z "$PREV_IDS" ]; then',
            f'  JID=$(sbatch --parsable "{script}")',
            "else",
            f'  JID=$(sbatch --parsable --dependency=afterany:$PREV_IDS "{script}")',
            "fi",
            f'echo "wave {wave}: submitted {script} -> job $JID"',
            'IDS="${IDS:+$IDS:}$JID"',
        ]
    submit_lines.append('PREV_IDS="$IDS"')

with open(snakemake.output.submit_waves, "w", encoding="utf-8", newline="\n") as f:  # noqa: F821
    f.write("\n".join(submit_lines) + "\n")

n_waves = len(scripts_by_wave)
n_scripts = sum(len(v) for v in scripts_by_wave.values())
print(f"\nDone. {n_waves} wave(s), {n_scripts} sbatch script(s) + resolved_config.yml + submit_waves.sh written to {local_jobs_dir}")
print(f"Fill in {base_config_path.parent / 'config_hpc.yml'} and hpc.sbatch.*/hpc.sbatch_large.* placeholders before submitting for real.")
