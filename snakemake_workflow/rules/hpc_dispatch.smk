"""Rule generating per-wave HPC sbatch scripts to run the flood solver in parallel across nodes.

Wave-based hinterland forcing (src/tile_chunking.compute_run_order,
rules/simulation.smk's own docstring): a hop_distance>=1 tile is seeded from
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

Within each wave, tiles are ALSO split by estimated size (2026-08 - Hydrax's
regular partitions all scale RAM at a fixed 8GB/vCPU, e.g. 1vcpu=8GB,
16vcpu=128GB): a tile at/above hpc.large_tile_pixel_threshold goes to a
separate 'large' batch using hpc.sbatch_large (a bigger partition) instead
of hpc.sbatch (tuned for the common case, e.g. 1vcpu) - see
aqueduct_runner.estimate_aqueduct_mem_mb for the memory model this threshold
is chosen against. Pixel count is estimated from each tile's own bounding
box (area_deg2 * 3600**2 - DeltaDTM is ~1 arcsec native resolution, and
what matters here is relative compute/memory cost, not exact physical area,
so no latitude correction is needed - same proxy already used by
tests/select_calibration_tiles.py), not by reading the actual DEM raster,
since this must be computable before any preprocessing has run.
"""

_hop_by_tile = {
    str(tid): int(hop)
    for tid, hop in zip(_tile_gdf["tile_id"], _tile_gdf["hop_distance"])
}

_tile_bounds = _tile_gdf.geometry.bounds
_approx_pixels_by_tile = {
    str(tid): float(maxx - minx) * float(maxy - miny) * 3600.0 * 3600.0
    for tid, minx, miny, maxx, maxy in zip(
        _tile_gdf["tile_id"], _tile_bounds["minx"], _tile_bounds["miny"], _tile_bounds["maxx"], _tile_bounds["maxy"],
    )
}
_LARGE_TILE_PIXEL_THRESHOLD = config["hpc"]["large_tile_pixel_threshold"]
# 2026-09: large tiles were regularly not finishing within hpc.sbatch_large's
# time limit even after that was raised - splitting large tiles across MORE,
# smaller batches (fewer tiles serially processed per node) cuts each node's
# own wall-clock need directly, on top of the longer time limit. Applied ON
# TOP of split_batches_proportionally's own shared-budget result for "large"
# specifically (not carved out of "small"'s share) - deliberately allows a
# wave with large tiles to use more than hpc.n_nodes nodes at once, since the
# whole point is more parallelism for the class that was timing out, not
# redistributing a fixed budget away from the (already-fine) small class.
_LARGE_TILE_BATCH_MULTIPLIER = config["hpc"]["large_tile_batch_multiplier"]

_HPC_WAVES: dict[int, list[str]] = {}
for _tid in TILE_IDS:
    _HPC_WAVES.setdefault(_hop_by_tile[str(_tid)], []).append(str(_tid))

# One or more node batches per (wave, size class), whole tiles per batch,
# split as evenly as possible. hpc.n_nodes is a SHARED budget across both
# size classes within a wave (split proportionally to each class's own
# tile count via split_batches_proportionally - 2026-08 fix; previously
# each class got its own independent min(n_nodes, len(class_tiles)),
# letting a wave with both classes present claim up to 2x n_nodes at once -
# found live, config.yml's own n_nodes=20 silently meaning up to 40 nodes
# for wave 0). Never more batches than a class has tiles to put in them,
# and a (wave, class) combination with zero tiles is skipped entirely
# (e.g. most waves will have no 'large' tiles at all).
_HPC_BATCHES = []  # [(wave, size_class, batch_id, [tile_id, ...]), ...] in generation order
for _wave in sorted(_HPC_WAVES):
    _wave_tiles = _HPC_WAVES[_wave]
    _tiles_by_class = {"small": [], "large": []}
    for _t in _wave_tiles:
        _is_large = _approx_pixels_by_tile[_t] >= _LARGE_TILE_PIXEL_THRESHOLD
        _tiles_by_class["large" if _is_large else "small"].append(_t)
    _class_n_nodes = split_batches_proportionally(
        {c: len(ts) for c, ts in _tiles_by_class.items()}, config["hpc"]["n_nodes"],
    )
    for _size_class, _class_tiles in _tiles_by_class.items():
        if not _class_tiles:
            continue
        _n_nodes = _class_n_nodes[_size_class]
        if _size_class == "large":
            _n_nodes = min(len(_class_tiles), _n_nodes * _LARGE_TILE_BATCH_MULTIPLIER)
        _k, _m = divmod(len(_class_tiles), _n_nodes)
        for _i in range(_n_nodes):
            _batch_tiles = _class_tiles[_i * _k + min(_i, _m): (_i + 1) * _k + min(_i + 1, _m)]
            _HPC_BATCHES.append((_wave, _size_class, f"{_i:03d}", _batch_tiles))

_HPC_SCRIPT_PATHS = [
    os.path.join(config["hpc"]["jobs_dir"], f"wave{_wave}_{_size_class}_batch_{_batch_id}.sbatch")
    for _wave, _size_class, _batch_id, _ in _HPC_BATCHES
]


rule generate_aqueduct_jobs:
    """Generate one sbatch script per (wave, size class, node batch), once preprocessing is done.

    Replaces run_aqueduct as the way to actually execute the flood solver
    for a full run: tiles are grouped whole (a tile's full return_period x
    waterlevel_name set stays on one node), split per-wave, then per size
    class (small tiles on hpc.sbatch, large ones on the bigger-RAM
    hpc.sbatch_large), across up to hpc.n_nodes sbatch scripts per group.
    Written to hpc.jobs_dir together with a resolved_config.yml
    (Linux-path-expanded, via config_hpc.yml - see
    scripts/generate_aqueduct_jobs.py) and a submit_waves.sh driver that
    submits every wave to SLURM in order (both size classes together within
    a wave - size only picks which partition a batch targets, not
    ordering), each wave depending on the full previous wave. The user runs
    that driver manually (or generate_hpc_preprocess_job.py chains it
    automatically); run_aqueduct itself is untouched and still usable for
    local/small/debug runs.
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
