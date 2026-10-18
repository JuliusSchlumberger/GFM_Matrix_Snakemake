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
from hpc_common import _stage_configfile_lines, _stage_tile_grid_lines, sbatch_header  # noqa: E402


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
        *sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs", cpus_per_task=1),
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


def _write_wave_scripts(
    hpc_cfg: dict,
    base_config_path: Path,
    batches: list,
    script_paths: list,
    resolved_config_path: str,
    submit_waves_path: str,
    wave_sbatch_cfg: dict | None,
    name_prefix: str,
    tiles_per_batch: list[int] | None = None,
) -> tuple[dict[int, list[str]], Path]:
    """Write the per-batch sbatch + triples files, resolved_config.yml and the submit script.

    Shared by the fresh and the resume dispatch, which differ only in how their
    `(wave, batch_id, [(tile_id, rp, slr), ...])` batches are formed and in the
    `name_prefix` of the scripts. `tiles_per_batch` only adds a tile count to the
    progress line. Returns the Linux script paths per wave and the local jobs dir.
    """
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

    for i, ((wave, batch_id, items), script_path) in enumerate(zip(batches, script_paths)):
        sbatch_cfg = wave_sbatch_cfg[wave] if wave_sbatch_cfg and wave in wave_sbatch_cfg else hpc_cfg["sbatch"]
        name = f"{name_prefix}wave{wave}_batch_{batch_id}"

        local_triples_path = Path(script_path).parent / f"{name}_triples.csv"
        linux_triples_path = f"{linux_jobs_dir}/{name}_triples.csv"
        with open(local_triples_path, "w", encoding="utf-8", newline="\n") as f:
            writer = csv.writer(f)
            for tile_id, rp, slr in items:
                writer.writerow([tile_id, rp, slr])

        lines = [
            # not -e: one failed (tile,rp,slr) must not abort the rest of this node's
            # batch, so each setup step below exits explicitly on failure instead.
            *sbatch_header(name, sbatch_cfg, f"{linux_jobs_dir}/logs", errexit=False),
            *_stage_configfile_lines(),
            *_stage_tile_grid_lines(),
            f'FAIL_LOG="{linux_jobs_dir}/logs/{name}_failures.txt"',
            ': > "$FAIL_LOG"',
            f'stage_configfile_locally "{linux_resolved_config}" LOCAL_CONFIGFILE || exit 1',
            f'stage_tile_grid_locally "{linux_tile_grid_path}" LOCAL_TILE_GRID || exit 1',
            "python -c 'import sys, yaml; cfg_path, new_path = sys.argv[1], sys.argv[2]; "
            "cfg = yaml.safe_load(open(cfg_path, encoding=\"utf-8\")); cfg[\"tile_grid\"][\"path\"] = new_path; "
            "yaml.safe_dump(cfg, open(cfg_path, \"w\", encoding=\"utf-8\"))' "
            # Unchecked, a failed rewrite would silently read the tile grid off P:\ again.
            '"$LOCAL_CONFIGFILE" "$LOCAL_TILE_GRID" || exit 1',
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
        tiles_note = f"{tiles_per_batch[i]} tiles, " if tiles_per_batch is not None else ""
        print(f"  wrote {script_path} (wave {wave}, {tiles_note}{len(items)} jobs, "
              f"{sbatch_cfg['cpus_per_task']}-way concurrent on {sbatch_cfg['partition']})")

    # afterany: a failed tile only fails its own hop>=1 neighbours (no output,
    # picked up by --resume), so it must not block the rest of the next wave.
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
    return scripts_by_wave, local_jobs_dir


def generate_wave_dispatch(
    batches: list,
    return_periods: list,
    waterlevel_names: list,
    hpc_cfg: dict,
    base_config_path: Path,
    script_paths: list,
    resolved_config_path: str,
    submit_waves_path: str,
    wave_sbatch_cfg: dict | None = None,
) -> None:
    """Write every wave's sbatch scripts + resolved_config.yml + submit_waves.sh.

    `batches` are `(wave, batch_id, tile_ids)` in the same order as `script_paths`.
    `wave_sbatch_cfg` is a `{wave: sbatch_cfg}` override of `hpc_cfg["sbatch"]`.
    """
    # Empty-boundary wave-0 jobs are not pre-filtered: run_eikonal_cli.py
    # checks that inline instead of ~110,000 file reads here.
    triple_batches = [
        (wave, batch_id, [(tile_id, rp, slr) for tile_id in batch_tiles
                          for rp in return_periods for slr in waterlevel_names])
        for wave, batch_id, batch_tiles in batches
    ]
    scripts_by_wave, local_jobs_dir = _write_wave_scripts(
        hpc_cfg, base_config_path, triple_batches, script_paths, resolved_config_path,
        submit_waves_path, wave_sbatch_cfg, name_prefix="",
        tiles_per_batch=[len(batch_tiles) for _, _, batch_tiles in batches],
    )

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
    scripts_by_wave, local_jobs_dir = _write_wave_scripts(
        hpc_cfg, base_config_path, batches, script_paths, resolved_config_path,
        submit_waves_path, wave_sbatch_cfg, name_prefix="resume_",
    )

    n_waves = len(scripts_by_wave)
    n_scripts = sum(len(v) for v in scripts_by_wave.values())
    n_jobs = sum(len(items) for _, _, items in batches)
    print(
        f"\nDone. {n_waves} wave(s) with remaining work, {n_scripts} sbatch script(s), "
        f"{n_jobs} total remaining job(s) written to {local_jobs_dir}"
    )


if "snakemake" in dir():
    generate_wave_dispatch(
        batches=snakemake.params.batches,  # noqa: F821
        return_periods=snakemake.params.return_periods,  # noqa: F821
        waterlevel_names=snakemake.params.waterlevel_names,  # noqa: F821
        hpc_cfg=snakemake.params.hpc_cfg,  # noqa: F821
        base_config_path=Path(snakemake.params.base_config_path),  # noqa: F821
        script_paths=snakemake.output.scripts,  # noqa: F821
        resolved_config_path=snakemake.output.resolved_config,  # noqa: F821
        submit_waves_path=snakemake.output.submit_waves,  # noqa: F821
    )
