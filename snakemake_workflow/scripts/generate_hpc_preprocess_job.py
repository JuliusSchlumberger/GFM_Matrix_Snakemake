"""Generate HPC sbatch scripts that run preprocessing across hpc.n_nodes
PARALLEL nodes, then - once every node's batch has finished - one more job
that runs `snakemake generate_aqueduct_jobs` (wave sbatch generation) and
submits submit_waves.sh itself. So a full run (preprocessing AND
simulation) can be launched from a single driver script, with preprocessing
itself spread across the same number of nodes as the simulation waves use.

Preprocessing has no wave/hop_distance ordering constraint (every tile's
DEM/mask/friction/boundaries are independent of every other tile) - unlike
simulation, where hop>=1 tiles must wait for their neighbours. So instead of
one monolithic `snakemake generate_aqueduct_jobs --cores N` run on a single
node, this splits tiles by estimated size FIRST (same bbox-area pixel-count
proxy and hpc.large_tile_pixel_threshold hpc_dispatch.smk uses for
simulation batching - tiles at/above it use hpc.sbatch_large, the rest use
hpc.sbatch), then splits each size class's target files evenly across up to
hpc.n_nodes batches, writes one `snakemake --cores N <explicit target file
list>` sbatch script per batch, and submits all of them with NO dependency
between them (fully parallel) - then submits ONE more job with
`--dependency=afterany:<every batch job id>` that runs generate_aqueduct_jobs
(fast, since every preprocessing output already exists by then) and chains
into submit_waves.sh. This is the exact same afterany-join-multiple-jobs
pattern generate_aqueduct_jobs.py's own submit_waves.sh already uses between
simulation waves - just one more phase in front of wave 0.

Target file paths are reconstructed directly (not via `rules.X.output.Y`
references, since this is a standalone script, not a Snakemake `script:`)
using the exact same path templates preprocessing.smk's rules declare -
mirrors the same approach generate_aqueduct_jobs.py already uses for its
own local-view boundaries check.

Uses the same local-view/Linux-view path resolution as
generate_aqueduct_jobs.py (config_hpc.yml, if present), so it can be run
either from the Windows preprocessing machine (writing scripts meant for
Hydrax) or natively ON Hydrax (config_hpc.yml unnecessary in that case).

Usage:
    python generate_hpc_preprocess_job.py [--config path/to/config.yml]
    bash <printed submit_preprocess_and_dispatch.sh path>
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config_utils import load_config, merged_slr_scenarios, retry_transient_io  # noqa: E402


def _account_line(sbatch_cfg: dict) -> list[str]:
    if sbatch_cfg.get("account"):  # optional - Hydrax jobs don't require one
        return [f"#SBATCH --account={sbatch_cfg['account']}"]
    return []


def _target_paths(tile_dir: str, return_periods: list[str], waterlevel_names: list[str]) -> list[str]:
    paths = [f"{tile_dir}/inputs/dem.tif", f"{tile_dir}/inputs/mask.tif", f"{tile_dir}/inputs/friction.tif"]
    for rp in return_periods:
        for slr in waterlevel_names:
            paths.append(f"{tile_dir}/inputs/boundaries_{rp}_{slr}.gpkg")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config" / "config.yml"
    parser.add_argument("--config", default=str(default_config))
    args = parser.parse_args()

    config_path = Path(args.config)
    local_config = load_config(config_path)
    linux_config = load_config(config_path, extra_override=config_path.parent / "config_hpc.yml")

    local_jobs_dir = Path(local_config["hpc"]["jobs_dir"])
    linux_jobs_dir = linux_config["hpc"]["jobs_dir"]
    linux_code_root = linux_config["paths"]["code_root"]
    linux_model_outputs = linux_config["simulation"]["model_outputs"]
    hpc_cfg = linux_config["hpc"]
    n_nodes = hpc_cfg["n_nodes"]
    large_pixel_threshold = hpc_cfg["large_tile_pixel_threshold"]

    retry_transient_io(local_jobs_dir.mkdir, parents=True, exist_ok=True)
    retry_transient_io((local_jobs_dir / "logs").mkdir, parents=True, exist_ok=True)

    tile_gdf = retry_transient_io(gpd.read_file, linux_config["tile_grid"]["path"])
    return_periods = [f"RP{rp}" for rp in linux_config["boundary_conditions"]["return_periods"]]
    waterlevel_names = merged_slr_scenarios(linux_config["boundary_conditions"], linux_config["adaptation"])

    # Same bbox-area pixel-count proxy hpc_dispatch.smk uses for simulation
    # batching (area_deg2 * 3600**2 - DeltaDTM's ~1 arcsec native
    # resolution) - computable from tile geometry alone, no DEM read
    # needed, so it works before any preprocessing has run.
    bounds = tile_gdf.geometry.bounds
    tiles_by_class: dict[str, list[str]] = {"small": [], "large": []}
    for tile_id, minx, miny, maxx, maxy in zip(
        tile_gdf["tile_id"], bounds["minx"], bounds["miny"], bounds["maxx"], bounds["maxy"],
    ):
        approx_pixels = (maxx - minx) * (maxy - miny) * 3600.0 * 3600.0
        size_class = "large" if approx_pixels >= large_pixel_threshold else "small"
        tiles_by_class[size_class].append(str(int(tile_id)))
    for size_class in tiles_by_class:
        tiles_by_class[size_class].sort(key=int)

    # Node budget split PROPORTIONALLY to each class's share of tiles
    # (e.g. ~15% large / ~85% small on the real production grid), rather
    # than each class independently getting up to n_nodes - the latter
    # would let preprocessing use up to 2x n_nodes total (n_nodes for
    # small + n_nodes for large) even though "large" is a small minority
    # of the actual work.
    present_classes = [c for c in ("small", "large") if tiles_by_class[c]]
    total_tiles = sum(len(tiles_by_class[c]) for c in present_classes)
    class_n_nodes: dict[str, int] = {}
    if len(present_classes) == 1:
        only = present_classes[0]
        class_n_nodes[only] = min(n_nodes, len(tiles_by_class[only]))
    else:
        for c in present_classes:
            class_n_nodes[c] = max(1, round(n_nodes * len(tiles_by_class[c]) / total_tiles))
        drift = n_nodes - sum(class_n_nodes.values())
        if drift:
            # absorb rounding drift into whichever class has more tiles
            biggest = max(present_classes, key=lambda c: len(tiles_by_class[c]))
            class_n_nodes[biggest] = max(1, class_n_nodes[biggest] + drift)
        for c in present_classes:
            class_n_nodes[c] = min(class_n_nodes[c], len(tiles_by_class[c]))

    batches = []  # [(size_class, batch_id, [tile_id, ...]), ...]
    for size_class, class_tiles in tiles_by_class.items():
        if not class_tiles:
            continue
        n_batches = class_n_nodes[size_class]
        k, m = divmod(len(class_tiles), n_batches)
        for i in range(n_batches):
            batch_tiles = class_tiles[i * k + min(i, m): (i + 1) * k + min(i + 1, m)]
            batches.append((size_class, f"{i:03d}", batch_tiles))

    batch_script_paths = []
    for size_class, batch_id, batch_tiles in batches:
        sbatch_cfg = hpc_cfg["sbatch_large"] if size_class == "large" else hpc_cfg["sbatch"]
        name = f"preprocess_{size_class}_batch_{batch_id}"
        targets = [
            p for tile_id in batch_tiles
            for p in _target_paths(f"{linux_model_outputs}/{tile_id}", return_periods, waterlevel_names)
        ]

        # Target list written to its own file (thousands of paths per
        # batch) and expanded at runtime via $(cat ...), rather than
        # inlining every path as a literal CLI argument in the sbatch
        # script body - keeps the script itself short and diffable.
        targets_path = local_jobs_dir / f"{name}_targets.txt"
        with open(targets_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(targets) + "\n")

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={name}",
            f"#SBATCH --partition={sbatch_cfg['partition']}",
            *_account_line(sbatch_cfg),
            f"#SBATCH --time={sbatch_cfg['time']}",
            f"#SBATCH --mem={sbatch_cfg['mem']}",
            f"#SBATCH --cpus-per-task={sbatch_cfg['cpus_per_task']}",
            f"#SBATCH --output={linux_jobs_dir}/logs/{name}_%j.out",
            f"#SBATCH --error={linux_jobs_dir}/logs/{name}_%j.err",
            "",
            "set -euo pipefail",
            sbatch_cfg["env_activate_cmd"],
            "",
            f'cd "{linux_code_root}"',
            f'echo "=== Preprocessing batch {size_class}/{batch_id}: {len(batch_tiles)} tiles ==="',
            f'snakemake --cores {sbatch_cfg["cpus_per_task"]} $(cat "{linux_jobs_dir}/{name}_targets.txt")',
            "",
        ]
        script_path = local_jobs_dir / f"{name}.sbatch"
        with open(script_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        batch_script_paths.append(f"{linux_jobs_dir}/{name}.sbatch")
        print(f"  wrote {script_path} ({size_class}, {len(batch_tiles)} tiles, {len(targets)} target files)")

    # Phase 2: once every preprocessing batch above has finished, generate
    # the wave sbatch scripts (fast - every input already exists) and
    # submit them. Lightweight (just calls snakemake + a shell script), so
    # it uses hpc.sbatch (the smaller of the two) rather than needing its
    # own dedicated config.
    dispatch_cfg = hpc_cfg["sbatch"]
    dispatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=gfm_generate_jobs_and_dispatch",
        f"#SBATCH --partition={dispatch_cfg['partition']}",
        *_account_line(dispatch_cfg),
        f"#SBATCH --time={dispatch_cfg['time']}",
        f"#SBATCH --mem={dispatch_cfg['mem']}",
        "#SBATCH --cpus-per-task=1",
        f"#SBATCH --output={linux_jobs_dir}/logs/generate_jobs_and_dispatch_%j.out",
        f"#SBATCH --error={linux_jobs_dir}/logs/generate_jobs_and_dispatch_%j.err",
        "",
        "set -euo pipefail",
        dispatch_cfg["env_activate_cmd"],
        "",
        f'cd "{linux_code_root}"',
        'echo "=== Generating wave sbatch scripts ==="',
        "snakemake generate_aqueduct_jobs --cores 1",
        "",
        'echo "=== Submitting simulation waves ==="',
        f'bash "{linux_jobs_dir}/submit_waves.sh"',
        "",
    ]
    dispatch_script_path = local_jobs_dir / "generate_jobs_and_dispatch.sbatch"
    with open(dispatch_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(dispatch_lines))
    print(f"  wrote {dispatch_script_path}")

    # Master driver: submit every preprocessing batch in PARALLEL (no
    # dependency between them - tiles are fully independent at this stage),
    # then the dispatch job depending on ALL of them via afterany.
    submit_lines = ["#!/bin/bash", "set -euo pipefail", "", 'IDS=""']
    for script in batch_script_paths:
        submit_lines += [
            f'JID=$(sbatch --parsable "{script}")',
            f'echo "submitted {script} -> job $JID"',
            'IDS="${IDS:+$IDS:}$JID"',
        ]
    submit_lines += [
        "",
        f'JID=$(sbatch --parsable --dependency=afterany:$IDS "{linux_jobs_dir}/generate_jobs_and_dispatch.sbatch")',
        (
            f'echo "submitted {linux_jobs_dir}/generate_jobs_and_dispatch.sbatch -> job $JID '
            f'(depends on all {len(batch_script_paths)} preprocessing batches)"'
        ),
    ]
    submit_script_path = local_jobs_dir / "submit_preprocess_and_dispatch.sh"
    with open(submit_script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(submit_lines) + "\n")

    node_summary = ", ".join(f"{size_class}={class_n_nodes[size_class]}" for size_class in present_classes)
    print(
        f"\nDone. {len(batch_script_paths)} preprocessing batch(es) ({node_summary} nodes, "
        f"{n_nodes} total budget) + 1 dispatch job written to {local_jobs_dir}"
    )
    print(f"Submit on Hydrax with: bash {linux_jobs_dir}/submit_preprocess_and_dispatch.sh")


if __name__ == "__main__":
    main()
