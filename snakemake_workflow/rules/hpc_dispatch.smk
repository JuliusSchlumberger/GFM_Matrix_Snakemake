"""Rule generating HPC sbatch scripts to run Aqueduct in parallel across nodes."""

_HPC_BATCH_IDS = [f"{i:03d}" for i in range(config["hpc"]["n_nodes"])]


rule generate_aqueduct_jobs:
    """Generate one sbatch script per HPC node batch, once preprocessing is done.

    Aqueduct can only run one instance at a time PER MACHINE (see
    run_aqueduct's docstring in simulation.smk) - not a global constraint -
    so separate HPC nodes can each run their own serial stream of Aqueduct
    invocations concurrently with every other node. This rule replaces
    run_aqueduct as the way to actually execute Aqueduct for a full run:
    tiles are grouped whole (a tile's full return_period x waterlevel_name
    set stays on one node) and split evenly across hpc.n_nodes sbatch
    scripts, written to hpc.jobs_dir together with a resolved_config.yml
    (Linux-path-expanded, via config_hpc.yml - see
    scripts/generate_aqueduct_jobs.py) and a submit_all.sh convenience
    wrapper. The user submits these to SLURM manually; run_aqueduct itself
    is untouched and still usable for local/small/debug runs.
    """
    input:
        _PREPROCESS_OUTPUTS,
    output:
        scripts=expand(
            os.path.join(config["hpc"]["jobs_dir"], "aqueduct_batch_{batch_id}.sbatch"),
            batch_id=_HPC_BATCH_IDS,
        ),
        resolved_config=os.path.join(config["hpc"]["jobs_dir"], "resolved_config.yml"),
        submit_all=os.path.join(config["hpc"]["jobs_dir"], "submit_all.sh"),
    params:
        hpc_cfg=config["hpc"],
        base_config_path=os.path.join(workflow.basedir, "snakemake_workflow", "config", "config.yml"),
        model_outputs=config["simulation"]["model_outputs"],
        tile_ids=TILE_IDS,
        return_periods=RETURN_PERIODS,
        waterlevel_names=WATERLEVEL_NAMES,
        raster_config=config["raster_format"],
    script:
        "../scripts/generate_aqueduct_jobs.py"
