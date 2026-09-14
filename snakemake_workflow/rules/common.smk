"""Shared setup for the GFM Aqueduct Snakemake workflow."""


wildcard_constraints:
    tile_id=r"\d+",
    chunk_id=r"[NS]\d{2}[EW]\d{3}",
    waterlevel_name="|".join(WATERLEVEL_NAMES),
    return_period="|".join(f"RP{rp}" for rp in config["boundary_conditions"]["return_periods"]),
