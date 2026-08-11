"""Generate per-wave sbatch scripts to run the flood solver in parallel on HPC.

The actual generation logic lives in `generate_wave_dispatch()` below, shared
by two entry points:
  - `rule generate_aqueduct_jobs` (hpc_dispatch.smk) - runs this as a
    Snakemake `script:`, gated on `input: _PREPROCESS_OUTPUTS` (every
    preprocessing output across the whole domain), so it only starts once
    Snakemake has verified that full dependency list - slow over a network
    mount at real scale (confirmed live 2026-08-10: multi-hour DAG build for
    ~2578 tiles x ~45 scenarios each).
  - `scripts/generate_hpc_simulation_jobs.py` (2026-08, new) - a standalone
    CLI that computes the same wave/batch partition directly from the tile
    grid, with NO Snakemake DAG/dependency check at all - for regenerating
    wave scripts quickly (e.g. after an hpc.n_nodes change) without waiting
    through that DAG build again. It's the caller's own responsibility to
    know preprocessing is actually done first (e.g. via
    check_preprocess_progress.py) - unlike the Snakemake path, this one
    does not verify it.

Does two path resolutions side by side, since it runs on the local
preprocessing machine but writes instructions meant for a Linux HPC:
  - `model_outputs` (this machine's own, already-expanded view) - currently
    only used to build sbatch script paths locally.
  - `config_utils.load_config(..., extra_override="config_hpc.yml")` is
    loaded separately to get the Linux-expanded view (model_outputs,
    code_root) embedded into the generated sbatch scripts and
    resolved_config.yml - see config_hpc.yml.example.

Every (tile, rp, slr) combo - wave-0 and hop_distance>=1 alike - is included
in the generated sbatch scripts unconditionally; NEITHER is pre-filtered at
generation time (2026-08 - wave-0 empty-boundaries jobs used to be resolved
here, reading every wave-0 tile's boundaries file to check `.empty` before
writing the sbatch scripts; removed as a needless ~110,000-file-read,
~90min pass on a real production grid - see run_aqueduct_cli.py's own
docstring for where that check now lives instead, inline with the read it
needs anyway). Hop_distance >= 1 tiles were NEVER pre-excludable this way to
begin with: whether a hinterland tile has any upstream flooding to seed
from can only be known once its lower-hop neighbour has actually run
(run_aqueduct_cli.py discovers "no seeds yet" live and writes its own
zero-waterdepth result) - pre-filtering them here on a same-run-as-wave-0
basis would incorrectly exclude every hop>=1 tile, since a hinterland tile
has no nearby boundary stations by definition. This is also why waves must
be submitted as SLURM dependency barriers (see submit_waves.sh below): a
hop>=1 job started before its neighbour's job has finished would read
"no neighbour output yet" as "no flooding", not as "not run yet".

`batches` (built in hpc_dispatch.smk / generate_hpc_simulation_jobs.py from
the tile grid's hop_distance column AND each tile's estimated pixel count,
node budget split proportionally between size classes via
config_utils.split_batches_proportionally) already carries the full (wave,
size_class, batch_id, tile_ids) partition - this script only turns each
batch into one sbatch script (using hpc.sbatch or the bigger-RAM
hpc.sbatch_large, per the batch's size_class) and groups them into
submit_waves.sh's per-wave dependency chain (both size classes within a
wave submit together - size only picks a batch's partition, not when it
starts).
"""

import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, retry_transient_io  # noqa: E402


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
) -> None:
    """Write every wave's sbatch scripts + resolved_config.yml + submit_waves.sh.

    `script_paths` must be in the same order as `batches` (one path per
    (wave, size_class, batch_id, tile_ids) entry) - both the Snakemake rule
    and the standalone CLI build them from the same `batches` list, so the
    order always matches by construction.
    """
    # Every (tile, rp, slr) combo is included unconditionally - wave-0 jobs
    # whose boundaries turn out empty are no longer pre-filtered here (2026-08
    # - this used to re-read every wave-0 tile's boundaries file at
    # generation time just to check .empty, ~110,000 individual file reads
    # on a real production grid, ~90min, for no benefit over letting
    # run_aqueduct_cli.py check it inline as part of the read it needs
    # anyway for a real, non-empty scenario - see that script's own
    # docstring, and run_aqueduct.py, which already worked this way).
    jobs_by_tile: dict[str, list[tuple[str, str]]] = {
        str(tid): [(rp, slr) for rp in return_periods for slr in waterlevel_names]
        for tid in tile_ids
    }

    # ── Linux view: fully Linux-expanded config, for the sbatch scripts ─────────
    linux_config = load_config(base_config_path, extra_override=base_config_path.parent / "config_hpc.yml")
    linux_aqueduct_cli = f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts/run_aqueduct_cli.py"
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    local_jobs_dir = Path(hpc_cfg["jobs_dir"])
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(linux_config, f)

    # ── Write one sbatch script per (wave, size_class, batch), in the same
    #    order as script_paths (caller built both from the same `batches`
    #    list, so the order always matches).
    scripts_by_wave: dict[int, list[str]] = {}

    for (wave, size_class, batch_id, batch_tiles), script_path in zip(batches, script_paths):
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

    with open(submit_waves_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_waves = len(scripts_by_wave)
    n_scripts = sum(len(v) for v in scripts_by_wave.values())
    print(f"\nDone. {n_waves} wave(s), {n_scripts} sbatch script(s) + resolved_config.yml + submit_waves.sh written to {local_jobs_dir}")
    print(f"Fill in {base_config_path.parent / 'config_hpc.yml'} and hpc.sbatch.*/hpc.sbatch_large.* placeholders before submitting for real.")


def generate_resume_dispatch(
    hpc_cfg: dict,
    base_config_path: Path,
    batches: list,
    script_paths: list,
    resolved_config_path: str,
    submit_waves_path: str,
) -> None:
    """Write sbatch scripts for a RESUME/rebalance dispatch (2026-08, new).

    Unlike generate_wave_dispatch (whole-tile-per-batch - every scenario for
    an included tile), each entry in `batches` here is `(wave, size_class,
    batch_id, items)` where `items` is a flat list of individual
    `(tile_id, rp, slr)` triples - built by the caller
    (generate_hpc_simulation_jobs.py's --resume mode) from whatever's still
    missing on disk, then re-split evenly across the full node budget
    without regard to the ORIGINAL whole-tile batch boundaries. This is what
    lets a rebalance recover from batches that finished their own (uneven)
    original tile lists early and went idle (see conversation 2026-08-11) -
    the new batches are sized by remaining WORK ITEMS, not by how many whole
    tiles happened to land in a batch originally.

    Each generated job is still exactly one `run_aqueduct_cli.py` call per
    (tile, rp, slr), identical to a fresh dispatch - combined with that
    script's own idempotency check (`_output_already_done`), submitting
    these is safe even if a listed item finishes via some other path before
    its job actually runs, though the intended usage is to cancel the
    original batches first (see module docstring) so there's no double-
    processing window at all.

    A wave with zero remaining items is skipped entirely (no key in
    `scripts_by_wave`) - submit_waves.sh's PREV_IDS then correctly carries
    forward from the last wave that DID have batches, so a fully-finished
    early wave doesn't break the dependency chain into whatever wave
    genuinely has remaining work first.
    """
    linux_config = load_config(base_config_path, extra_override=base_config_path.parent / "config_hpc.yml")
    linux_aqueduct_cli = f"{linux_config['paths']['code_root']}/snakemake_workflow/scripts/run_aqueduct_cli.py"
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_resolved_config = f"{linux_jobs_dir}/resolved_config.yml"

    local_jobs_dir = Path(hpc_cfg["jobs_dir"])
    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(linux_config, f)

    scripts_by_wave: dict[int, list[str]] = {}

    for (wave, size_class, batch_id, items), script_path in zip(batches, script_paths):
        sbatch_cfg = hpc_cfg["sbatch_large"] if size_class == "large" else hpc_cfg["sbatch"]
        name = f"resume_wave{wave}_{size_class}_batch_{batch_id}"
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
        for tile_id, rp, slr in items:
            lines.append(f"run_job {tile_id} {rp} {slr}")
        lines.append("")

        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        scripts_by_wave.setdefault(wave, []).append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} (wave {wave}, {size_class}, {len(items)} jobs)")

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

    with open(submit_waves_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    n_waves = len(scripts_by_wave)
    n_scripts = sum(len(v) for v in scripts_by_wave.values())
    n_jobs = sum(len(items) for _, _, _, items in batches)
    print(
        f"\nDone. {n_waves} wave(s) with remaining work, {n_scripts} sbatch script(s), "
        f"{n_jobs} total remaining job(s) written to {local_jobs_dir}"
    )


# ── Snakemake entry point (rule generate_aqueduct_jobs, hpc_dispatch.smk) ──
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
