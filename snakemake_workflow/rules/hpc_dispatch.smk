"""Generate per-wave HPC sbatch scripts that run the flood solver across nodes.

A hop_distance>=1 tile is seeded from a strictly-lower-hop neighbour's output for
the SAME scenario, so wave N+1 must wait until every wave-N job is terminal.
Tiles are grouped by hop_distance first, then split into node batches per wave;
submit_waves.sh chains waves with `sbatch --dependency=afterany:<prior-wave ids>`.
"""

_hop_by_tile = {
    str(tid): int(hop)
    for tid, hop in zip(_tile_gdf["tile_id"], _tile_gdf["hop_distance"])
}

_HPC_WAVES: dict[int, list[str]] = {}
for _tid in TILE_IDS:
    _HPC_WAVES.setdefault(_hop_by_tile[str(_tid)], []).append(str(_tid))

# Whole tiles per batch, at most hpc.n_nodes batches and never more than tiles.
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


rule generate_eikonal_jobs:
    """Generate one sbatch script per (wave, node batch), plus submit_waves.sh.

    A tile's full return_period x waterlevel_name set stays on one node.
    resolved_config.yml is Linux-path-expanded via config_hpc.yml.
    """
    input:
        _PREPROCESS_OUTPUTS,
    output:
        scripts=_HPC_SCRIPT_PATHS,
        resolved_config=os.path.join(config["hpc"]["jobs_dir"], "resolved_config.yml"),
        submit_waves=os.path.join(config["hpc"]["jobs_dir"], "submit_waves.sh"),
    params:
        hpc_cfg=config["hpc"],
        # KNOWN TRAP: hardcoded path to production config.yml, not the live
        # --configfile; generate_wave_dispatch() re-reads it to build
        # resolved_config.yml, so a scenario --configfile has no effect there.
        # Calibration runs bypass this rule (generate_hpc_simulation_jobs.py).
        base_config_path=os.path.join(workflow.basedir, "snakemake_workflow", "config", "config.yml"),
        model_outputs=config["simulation"]["model_outputs"],
        tile_ids=TILE_IDS,
        hop_by_tile=_hop_by_tile,
        batches=_HPC_BATCHES,
        return_periods=RETURN_PERIODS,
        waterlevel_names=WATERLEVEL_NAMES,
        raster_config=config["raster_format"],
    script:
        "../scripts/generate_eikonal_jobs.py"
