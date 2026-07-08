"""Shared setup for the GFM Aqueduct Snakemake workflow."""


wildcard_constraints:
    tile_id=r"\d+",
    waterlevel_name="|".join(WATERLEVEL_NAMES),
    return_period="|".join(f"RP{rp}" for rp in config["boundary_conditions"]["return_periods"]),
