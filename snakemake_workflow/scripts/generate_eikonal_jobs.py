"""Generate per-wave sbatch scripts to run the flood solver in parallel on HPC.

`generate_wave_dispatch()` is shared by rule generate_eikonal_jobs (gated on all
preprocessing outputs) and generate_hpc_simulation_jobs.py (no DAG check; the
caller must know preprocessing is done). Sbatch scripts and resolved_config.yml
embed the Linux-expanded config (config_hpc.yml).

Every (tile, rp, slr) is included unconditionally: whether a hop>=1 tile has
upstream flooding is only known once its lower-hop neighbour ran, which is why
waves are SLURM dependency barriers. Each batch's tiles run concurrently up to
cpus_per_task via run_simulation_batch.py, reading `{batch_name}_triples.csv`.
"""

import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from config_utils import atomic_write, load_config, retry_transient_io  # noqa: E402
from hpc_common import _stage_configfile_lines, _stage_tile_grid_lines  # noqa: E402


def _append_postprocess_bridge(
    submit_lines: list[str],
    hpc_cfg: dict,
    linux_config: dict,
    linux_jobs_dir: str,
    local_jobs_dir: Path,
    linux_resolved_config: str,
) -> list[str]:
    """Append the postprocessing+exposure bridge job after all waves (afterany).

    `linux_resolved_config` MUST be passed to generate_hpc_postprocess_job.py's
    --config: otherwise it defaults to production config.yml and a scenario run
    silently computes production-scale postprocessing in production's jobs_dir.
    """
    sbatch_cfg = hpc_cfg["sbatch"]  # lightweight - just generates + submits further scripts
    code_root = linux_config["paths"]["code_root"]
    name = "gfm_generate_postprocess_and_dispatch"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={sbatch_cfg['partition']}",
    ]
    if sbatch_cfg.get("account"):
        lines.append(f"#SBATCH --account={sbatch_cfg['account']}")
    lines += [
        f"#SBATCH --time={sbatch_cfg['time']}",
        f"#SBATCH --mem={sbatch_cfg['mem']}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
        "",
        "set -euo pipefail",
        sbatch_cfg["env_activate_cmd"],
        "",
        f'cd "{code_root}"',
        *_stage_configfile_lines(),
        # Deliberately NOT staging tile_grid.path here: this config is what
        # generate_hpc_postprocess_job.py embeds into downstream scripts, so a
        # node-local scratch path would leak into them and fail once this job ends.
        f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
        "",
        'echo "=== Generating postprocessing + exposure dispatch ==="',
        'python snakemake_workflow/scripts/generate_hpc_postprocess_job.py --config "$LOCAL_CONFIGFILE"',
        "",
        'echo "=== Submitting postprocessing + exposure ==="',
        f'bash "{linux_jobs_dir}/submit_postprocess_and_exposure.sh"',
        "",
    ]
    script_path = local_jobs_dir / f"{name}.sbatch"
    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))

    submit_lines = list(submit_lines)
    submit_lines += [
        "",
        "# bridge: postprocessing + exposure analysis, once ALL simulation waves are done",
        'if [ -z "$ALL_WAVE_IDS" ]; then',
        f'  JID=$(sbatch --parsable "{linux_jobs_dir}/{name}.sbatch")',
        "else",
        f'  JID=$(sbatch --parsable --dependency=afterany:$ALL_WAVE_IDS "{linux_jobs_dir}/{name}.sbatch")',
        "fi",
        f'echo "submitted {linux_jobs_dir}/{name}.sbatch -> job $JID (depends on all simulation waves)"',
    ]
    return submit_lines


def generate_wave_dispatch(
    model_outputs: str,
    tile_ids: list,
    hop_by_tile: dict,
    batches: list,
    return_periods: list,
    waterlevel_names: list,
    raster_config: dict,
    hpc_cfg: dict,
    base_config_path: Path,
    script_paths: list,
    resolved_config_path: str,
    submit_waves_path: str,
    wave_sbatch_cfg: dict | None = None,
) -> None:
    """Write every wave's sbatch scripts + resolved_config.yml + submit_waves.sh.

    `model_outputs`, `hop_by_tile` and `raster_config` are accepted but unused.
    `batches` are `(wave, batch_id, tile_ids)` in the same order as `script_paths`.
    `wave_sbatch_cfg` is a `{wave: sbatch_cfg}` override of `hpc_cfg["sbatch"]`.
    """
    # Empty-boundary wave-0 jobs are not pre-filtered: run_eikonal_cli.py
    # checks that inline instead of ~110,000 file reads here.
    jobs_by_tile: dict[str, list[tuple[str, str]]] = {
        str(tid): [(rp, slr) for rp in return_periods for slr in waterlevel_names]
        for tid in tile_ids
    }

    linux_config = load_config(base_config_path, extra_override=base_config_path.parent / "config_hpc.yml")
    linux_batch_runner = f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts/run_simulation_batch.py"
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"
    linux_tile_grid_path = linux_config["tile_grid"]["path"]

    # local_jobs_dir MUST come from the local config: the caller may pass the
    # Linux hpc_cfg, whose "/p/..." jobs_dir silently parses as a
    # drive-relative WindowsPath and writes to the wrong location.
    local_config = load_config(base_config_path)
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # atomic_write: a compute node reading this over P: right after a
    # non-atomic write can see a partial file (UnicodeDecodeError).
    atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")

    scripts_by_wave: dict[int, list[str]] = {}

    for (wave, batch_id, batch_tiles), script_path in zip(batches, script_paths):
        sbatch_cfg = wave_sbatch_cfg[wave] if wave_sbatch_cfg and wave in wave_sbatch_cfg else hpc_cfg["sbatch"]
        name = f"wave{wave}_batch_{batch_id}"

        local_triples_path = Path(script_path).parent / f"{name}_triples.csv"
        linux_triples_path = f"{linux_jobs_dir}/{name}_triples.csv"
        n_jobs = 0
        with open(local_triples_path, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            for tile_id in batch_tiles:
                for rp, slr in jobs_by_tile[tile_id]:
                    writer.writerow([tile_id, rp, slr])
                    n_jobs += 1

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
            *_stage_configfile_lines(),
            *_stage_tile_grid_lines(),
            f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
            ': > "$FAIL_LOG"',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{linux_tile_grid_path}" LOCAL_TILE_GRID || exit 1',
            "python -c 'import sys, yaml; cfg_path, new_path = sys.argv[1], sys.argv[2]; "
            "cfg = yaml.safe_load(open(cfg_path, encoding=\"utf-8\")); cfg[\"tile_grid\"][\"path\"] = new_path; "
            "yaml.safe_dump(cfg, open(cfg_path, \"w\", encoding=\"utf-8\"))' "
            '"$LOCAL_CONFIGFILE" "$LOCAL_TILE_GRID"',
            "",
            f'python "{linux_batch_runner}" \\',
            f'  --triples-file "{linux_triples_path}" \\',
            '  --config "$LOCAL_CONFIGFILE" \\',
            f'  --workers {sbatch_cfg["cpus_per_task"]} \\',
            '  --fail-log "$FAIL_LOG"',
            "",
        ]

        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        scripts_by_wave.setdefault(wave, []).append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} (wave {wave}, {len(batch_tiles)} tiles, "
              f"{n_jobs} jobs, {sbatch_cfg['cpus_per_task']}-way concurrent on {sbatch_cfg['partition']})")

    # afterany: a failed tile only makes its neighbours fall back to a real-zero
    # result, so it must not block the next wave.
    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "PREV_IDS=\"\"",
        "ALL_WAVE_IDS=\"\"",
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
        submit_lines.append('ALL_WAVE_IDS="${ALL_WAVE_IDS:+$ALL_WAVE_IDS:}$IDS"')

    submit_lines = _append_postprocess_bridge(
        submit_lines, hpc_cfg, linux_config, linux_jobs_dir, local_jobs_dir, linux_resolved_config,
    )

    with open(submit_waves_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_waves = len(scripts_by_wave)
    n_scripts = sum(len(v) for v in scripts_by_wave.values())
    print(f"\nDone. {n_waves} wave(s), {n_scripts} sbatch script(s) + resolved_config.yml + submit_waves.sh written to {local_jobs_dir}")
    print(f"Fill in {base_config_path.parent / 'config_hpc.yml'} and hpc.sbatch.* placeholders before submitting for real.")


def generate_resume_dispatch(
    hpc_cfg: dict,
    base_config_path: Path,
    batches: list,
    script_paths: list,
    resolved_config_path: str,
    submit_waves_path: str,
    wave_sbatch_cfg: dict | None = None,
) -> None:
    """Write sbatch scripts for a resume/rebalance dispatch.

    Each `batches` entry is `(wave, batch_id, items)` with `items` a flat list of
    `(tile_id, rp, slr)` triples, re-split evenly across the full node budget.
    Waves with no remaining items are skipped; the dependency chain carries over.
    """
    linux_config = load_config(base_config_path, extra_override=base_config_path.parent / "config_hpc.yml")
    linux_batch_runner = f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts/run_simulation_batch.py"
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"
    linux_tile_grid_path = linux_config["tile_grid"]["path"]

    # Local config view, not hpc_cfg (see generate_wave_dispatch).
    local_config = load_config(base_config_path)
    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    # atomic_write: see generate_wave_dispatch.
    atomic_write(resolved_config_path, lambda f: yaml.safe_dump(linux_config, f), encoding="utf-8", newline="")

    scripts_by_wave: dict[int, list[str]] = {}

    for (wave, batch_id, items), script_path in zip(batches, script_paths):
        sbatch_cfg = wave_sbatch_cfg[wave] if wave_sbatch_cfg and wave in wave_sbatch_cfg else hpc_cfg["sbatch"]
        name = f"resume_wave{wave}_batch_{batch_id}"

        local_triples_path = Path(script_path).parent / f"{name}_triples.csv"
        linux_triples_path = f"{linux_jobs_dir}/{name}_triples.csv"
        with open(local_triples_path, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            for tile_id, rp, slr in items:
                writer.writerow([tile_id, rp, slr])

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={sbatch_cfg['partition']}",
        ]
        if sbatch_cfg.get("account"):
            lines.append(f"#SBATCH --account={sbatch_cfg['account']}")
        lines += [
            f"#SBATCH --time={sbatch_cfg['time']}",
            f"#SBATCH --mem={sbatch_cfg['mem']}",
            f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -uo pipefail",
            sbatch_cfg["env_activate_cmd"],
            "",
            *_stage_configfile_lines(),
            *_stage_tile_grid_lines(),
            f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
            ': > "$FAIL_LOG"',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{linux_tile_grid_path}" LOCAL_TILE_GRID || exit 1',
            "python -c 'import sys, yaml; cfg_path, new_path = sys.argv[1], sys.argv[2]; "
            "cfg = yaml.safe_load(open(cfg_path, encoding=\"utf-8\")); cfg[\"tile_grid\"][\"path\"] = new_path; "
            "yaml.safe_dump(cfg, open(cfg_path, \"w\", encoding=\"utf-8\"))' "
            '"$LOCAL_CONFIGFILE" "$LOCAL_TILE_GRID"',
            "",
            f'python "{linux_batch_runner}" \\',
            f'  --triples-file "{linux_triples_path}" \\',
            '  --config "$LOCAL_CONFIGFILE" \\',
            f'  --workers {sbatch_cfg["cpus_per_task"]} \\',
            '  --fail-log "$FAIL_LOG"',
            "",
        ]

        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        scripts_by_wave.setdefault(wave, []).append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} (wave {wave}, {len(items)} jobs, "
              f"{sbatch_cfg['cpus_per_task']}-way concurrent on {sbatch_cfg['partition']})")

    submit_lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "PREV_IDS=\"\"",
        "ALL_WAVE_IDS=\"\"",
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
        submit_lines.append('ALL_WAVE_IDS="${ALL_WAVE_IDS:+$ALL_WAVE_IDS:}$IDS"')

    submit_lines = _append_postprocess_bridge(
        submit_lines, hpc_cfg, linux_config, linux_jobs_dir, local_jobs_dir, linux_resolved_config,
    )

    with open(submit_waves_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_waves = len(scripts_by_wave)
    n_scripts = sum(len(v) for v in scripts_by_wave.values())
    n_jobs = sum(len(items) for _, _, items in batches)
    print(
        f"\nDone. {n_waves} wave(s) with remaining work, {n_scripts} sbatch script(s), "
        f"{n_jobs} total remaining job(s) written to {local_jobs_dir}"
    )


if "snakemake" in dir():
    generate_wave_dispatch(
        model_outputs=snakemake.params.model_outputs,  # noqa: F821
        tile_ids=snakemake.params.tile_ids,  # noqa: F821
        hop_by_tile=snakemake.params.hop_by_tile,  # noqa: F821
        batches=snakemake.params.batches,  # noqa: F821
        return_periods=snakemake.params.return_periods,  # noqa: F821
        waterlevel_names=snakemake.params.waterlevel_names,  # noqa: F821
        raster_config=snakemake.params.raster_config,  # noqa: F821
        hpc_cfg=snakemake.params.hpc_cfg,  # noqa: F821
        base_config_path=Path(snakemake.params.base_config_path),  # noqa: F821
        script_paths=snakemake.output.scripts,  # noqa: F821
        resolved_config_path=snakemake.output.resolved_config,  # noqa: F821
        submit_waves_path=snakemake.output.submit_waves,  # noqa: F821
    )
