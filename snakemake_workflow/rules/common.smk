"""Shared setup for the GFM Aqueduct Snakemake workflow."""


wildcard_constraints:
    tile_id=r"\d+",
    waterlevel_name="|".join(config["boundary_conditions"]["slr_scenarios"]),
