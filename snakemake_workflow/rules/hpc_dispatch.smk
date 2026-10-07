"""Rule generating per-wave HPC sbatch scripts to run the flood solver in parallel across nodes.

Wave-based hinterland forcing (hop_distance computed by
src/connectivity_tiling.compute_hop_distances, rules/simulation.smk's own
docstring): a hop_distance>=1 tile is seeded from
a strictly-lower-hop_distance neighbour's own output for the SAME scenario
(src/boundaries.collect_neighbor_wave_seeds), so wave N+1 cannot start until
every wave-N job has reached a terminal state - not just "started". Tiles
are therefore grouped by hop_distance FIRST, then split into node batches
within each wave (same whole-tile-per-batch, even-split logic as before),
so this rule generates one set of sbatch scripts per wave instead of one
flat set across all tiles. The generated submit_waves.sh driver (written by
generate_aqueduct_jobs.py) submits wave 0 with no dependency, then each
subsequent wave with `sbatch --dependency=afterany:<every prior-wave job
id>`, guaranteeing a hop>=1 tile's neighbour output actually exists on disk
before it runs. run_aqueduct_cli.py itself is unchanged - it already
resolves hop_distance per tile and reads/writes accordingly; this rule only
controls WHEN each tile's job is allowed to start.

(2026-10-08: tiles used to also be split within each wave by estimated size,
routing anything above hpc.large_tile_pixel_threshold to a separate,
bigger-RAM hpc.sbatch_large partition - a real need under the OLD, pre-
2026-10 greedy-covering tiling pipeline, which could produce much larger
domains. The connectivity-first pipeline's own hard_ceiling_cells caps every
tile's size directly, and a live check against the real production grid
found only one tile anywhere near that old concern - see config.yml's own
comment on hpc.sbatch for the numbers - so this size-class split, along
with hpc.sbatch_large itself, was removed as dead weight from a retired
pipeline generation.)
"""

_hop_by_tile = {
    str(tid): int(hop)
    for tid, hop in zip(_tile_gdf["tile_id"], _tile_gdf["hop_distance"])
}

_HPC_WAVES: dict[int, list[str]] = {}
for _tid in TILE_IDS:
    _HPC_WAVES.setdefault(_hop_by_tile[str(_tid)], []).append(str(_tid))

# One or more node batches per wave, whole tiles per batch, split as evenly
# as possible across up to hpc.n_nodes batches (never more batches than the
# wave has tiles to put in them).
_HPC_BATCHES = []  # [(wave, batch_id, [tile_id, ...]), ...] in generation order
for _wave in sorted(_HPC_WAVES):
    _wave_tiles = _HPC_WAVES[_wave]
    _n_nodes = min(config["hpc"]["n_nodes"], len(_wave_tiles))
    _k, _m = divmod(len(_wave_tiles), _n_nodes)
    for _i in range(_n_nodes):
        _batch_tiles = _wave_tiles[_i * _k + min(_i, _m): (_i + 1) * _k + min(_i + 1, _m)]
        _HPC_BATCHES.append((_wave, f"{_i:03d}", _batch_tiles))

_HPC_SCRIPT_PATHS = [
    os.path.join(config["hpc"]["jobs_dir"], f"wave{_wave}_batch_{_batch_id}.sbatch")
    for _wave, _batch_id, _ in _HPC_BATCHES
]


rule generate_aqueduct_jobs:
    """Generate one sbatch script per (wave, node batch), once preprocessing is done.

    Replaces run_aqueduct as the way to actually execute the flood solver
    for a full run: tiles are grouped whole (a tile's full return_period x
    waterlevel_name set stays on one node), split per-wave, across up to
    hpc.n_nodes sbatch scripts per wave. Written to hpc.jobs_dir together
    with a resolved_config.yml (Linux-path-expanded, via config_hpc.yml -
    see scripts/generate_aqueduct_jobs.py) and a submit_waves.sh driver that
    submits every wave to SLURM in order, each wave depending on the full
    previous wave. The user runs that driver manually (or
    generate_hpc_preprocess_job.py chains it automatically); run_aqueduct
    itself is untouched and still usable for local/small/debug runs.
    """
    input:
        _PREPROCESS_OUTPUTS,
    output:
        scripts=_HPC_SCRIPT_PATHS,
        resolved_config=os.path.join(config["hpc"]["jobs_dir"], "resolved_config.yml"),
        submit_waves=os.path.join(config["hpc"]["jobs_dir"], "submit_waves.sh"),
    params:
        hpc_cfg=config["hpc"],
        # KNOWN TRAP (2026-09): this is a hardcoded literal path to
        # production config.yml, NOT derived from whatever --configfile
        # built the live `config` dict above - generate_wave_dispatch()
        # re-reads THIS path from scratch to build resolved_config.yml (the
        # file every compute node actually loads), so a scenario/calibration
        # --configfile silently has no effect on resolved_config.yml even
        # though `params` below correctly reflects it. Calibration runs
        # bypass this rule entirely (generate_hpc_preprocess_job.py's
        # --calibration flag calls generate_hpc_simulation_jobs.py directly
        # instead, which threads --config through correctly) rather than
        # fixing it here - see that script's module docstring.
        base_config_path=os.path.join(workflow.basedir, "snakemake_workflow", "config", "config.yml"),
        model_outputs=config["simulation"]["model_outputs"],
        tile_ids=TILE_IDS,
        hop_by_tile=_hop_by_tile,
        batches=_HPC_BATCHES,
        return_periods=RETURN_PERIODS,
        waterlevel_names=WATERLEVEL_NAMES,
        raster_config=config["raster_format"],
    script:
        "../scripts/generate_aqueduct_jobs.py"
